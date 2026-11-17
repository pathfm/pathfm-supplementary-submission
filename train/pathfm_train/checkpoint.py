"""Atomic training checkpoints, EMA weights, and independent RNG streams."""

import copy
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generators(device, seed):
    return {
        name: torch.Generator(device=device).manual_seed(seed + offset)
        for name, offset in (("noise", 10), ("time", 20), ("dropout", 60))
    }


def capture_rng(streams, device):
    return {
        "generators": {
            name: stream.get_state().cpu() for name, stream in streams.items()
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu()
        if str(device).startswith("cuda")
        else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state, streams, device):
    for name, stream in streams.items():
        stream.set_state(state["generators"][name].cpu())
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    parameter.detach().float(), alpha=1 - self.decay
                )

    def state_dict(self):
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {name: value.cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state):
        if state["decay"] != self.decay or set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA configuration or parameter names differ")
        self.updates = state["updates"]
        for name, value in state["shadow"].items():
            self.shadow[name].copy_(value)

    def weights(self, model):
        return {
            name: self.shadow.get(name, value).detach().cpu().contiguous().clone()
            for name, value in model.state_dict().items()
        }


def atomic_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def export(model, config, ema, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "model.safetensors.tmp"
    save_file(ema.weights(model), str(temporary), metadata={"format": "pt"})
    os.replace(temporary, destination / "model.safetensors")
    config = {**config, "weights_sha256": digest(destination / "model.safetensors")}
    temporary = destination / "config.json.tmp"
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    os.replace(temporary, destination / "config.json")


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != "pathfm_training_v1":
        raise ValueError("Expected a full checkpoint produced by this training package")
    required = {
        "model",
        "model_config",
        "optimizer",
        "ema",
        "sampler",
        "rng",
        "recipe",
        "step",
        "stage",
    }
    if not required <= saved.keys():
        raise ValueError("Checkpoint lacks complete training state")
    return saved


def cpu_state(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def compatible_recipe(saved, recipe):
    """A resume keeps the full recipe, including every schedule endpoint."""
    before, after = copy.deepcopy(saved["recipe"]), copy.deepcopy(recipe)
    if before != after:
        changed = [
            key for key in set(before) | set(after) if before.get(key) != after.get(key)
        ]
        raise ValueError(f"Resume changes the training recipe: {sorted(changed)}")
