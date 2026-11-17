"""Pretraining, WSD continuation, and PDD with a fixed global batch."""

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from pathfm.loading import build_model, load_model
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import (
    EMA,
    atomic_save,
    capture_rng,
    compatible_recipe,
    cpu_state,
    digest,
    export,
    generators,
    load_checkpoint,
    restore_rng,
    seed_everything,
)
from .data import PandaOnlineSampler, TrajectoryStore
from .model import initialize, initialize_student, optimizer
from .objectives import Objective, draw_inputs


def learning_rate_factor(recipe, stage, step):
    settings = recipe[stage]
    if stage == "pretrain":
        return min(step / max(settings["warmup_steps"], 1), 1.0)
    if stage == "wsd":
        return max(1.0 - step / settings["steps"], 0.0)
    progress = min(max((step - 1) / max(settings["steps"] - 1, 1), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def batch_to_tensors(batch, device):
    groups, channels, length = batch["context"].shape
    return {
        "context": torch.from_numpy(
            batch["context"].reshape(groups * channels, length)
        ).to(device),
        "target": torch.from_numpy(batch["target"].reshape(groups * channels, -1)).to(
            device
        ),
        "group_ids": torch.arange(groups, device=device).repeat_interleave(channels),
        "groups": groups,
        "channels": channels,
    }


def partition(tensors, inputs, rank, world):
    groups, channels = tensors["groups"], tensors["channels"]
    if groups < world:
        raise ValueError(
            "Each GPU needs at least one complete trajectory; increase batch_rows"
        )
    first, last = groups * rank // world, groups * (rank + 1) // world
    rows = slice(first * channels, last * channels)
    local = {
        "context": tensors["context"][rows],
        "target": tensors["target"][rows],
        "group_ids": tensors["group_ids"][rows] - first,
        "groups": last - first,
        "channels": channels,
    }
    return (
        local,
        {name: value[:, rows] for name, value in inputs.items()},
        (last - first) / groups,
    )


def teacher_from(path, device, recipe):
    path = Path(path)
    if path.is_dir():
        teacher, config = load_model(path, device=device)
        identity = digest(path / "model.safetensors")
    else:
        saved = load_checkpoint(path)
        if saved["stage"] != "wsd" or saved["step"] != recipe["wsd"]["steps"]:
            raise ValueError("PDD requires a completed WSD checkpoint")
        teacher = build_model(saved["model_config"]).to(device)
        weights = dict(saved["model"])
        weights.update(saved["ema"]["shadow"])
        teacher.load_state_dict(weights, strict=True)
        config = saved["model_config"]
        identity = digest(path)
    if (
        teacher.parallel_knots is not None
        or teacher.source_scale != recipe["source_scale"]
        or teacher.num_blocks != recipe["num_layers"]
        or teacher.max_horizon != recipe["prediction_length"]
        or teacher.backbone.config.d_ff != recipe["ffn_dim"]
    ):
        raise ValueError("Teacher architecture differs from this recipe")
    return teacher.eval().requires_grad_(False), config, identity


def run(args):
    recipe = json.loads(args.config.read_text())
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("These recipes support a single host")
    device = args.device
    if world > 1 and device.startswith("cuda"):
        device = f"cuda:{local_rank}"
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(
            "nccl" if device.startswith("cuda") else "gloo",
            timeout=timedelta(minutes=30),
        )
    try:
        train(args, recipe, device, rank, world)
    finally:
        if world > 1:
            dist.destroy_process_group()


def train(args, recipe, device, rank, world):
    seed_everything(recipe["seed"])
    stage = args.stage
    total = recipe[stage]["steps"]
    endpoint = total if args.stop_after is None else args.stop_after
    if not 0 < endpoint <= total:
        raise ValueError("--stop-after must lie within this stage's fixed schedule")
    streams = generators(device, recipe["seed"])
    store = TrajectoryStore(args.data_root)
    sampler = PandaOnlineSampler(
        store,
        split=recipe["train_split"],
        seed=recipe["seed"],
        augmentation_rate=recipe["augmentation_rate"],
        weighting="system",
        sampling_mode="online",
        prediction_length=recipe["prediction_length"],
        batch_rows=recipe["batch_rows"],
    )
    data_hash = digest(args.data_root / "manifest.json")
    saved = load_checkpoint(args.resume) if args.resume else None
    teacher, teacher_identity = None, None
    initial = args.init
    if saved:
        compatible_recipe(saved, recipe)
        if saved["stage"] != stage:
            raise ValueError(
                "--resume must continue the same stage; use --init for WSD/PDD"
            )
        if saved["world_size"] != world or saved["data_manifest_sha256"] != data_hash:
            raise ValueError(
                "Resume requires the original world size and dataset manifest"
            )
        initial = initial or (
            Path(saved["initial_checkpoint"]) if saved["initial_checkpoint"] else None
        )
    if stage == "pdd":
        if initial is None:
            raise ValueError(
                "PDD requires --init pointing to WSD EMA weights or a completed WSD checkpoint"
            )
        teacher, teacher_config, teacher_identity = teacher_from(
            initial, device, recipe
        )
        if saved and saved["teacher_sha256"] != teacher_identity:
            raise ValueError("The PDD resume teacher changed")
    if saved:
        model_config = saved["model_config"]
        model = build_model(model_config).to(device).train()
        model.load_state_dict(saved["model"], strict=True)
    elif stage == "pretrain":
        if initial is not None:
            raise ValueError(
                "Pretraining starts from the pinned Chronos-2 embeddings; use --resume to continue"
            )
        model, model_config = initialize(recipe, device, args.backbone)
    elif stage == "wsd":
        if initial is None or not Path(initial).is_file():
            raise ValueError(
                "WSD requires --init pointing to the full pretraining checkpoint"
            )
        saved = load_checkpoint(initial)
        compatible_recipe(saved, recipe)
        if saved["stage"] != "pretrain" or saved["step"] != recipe["pretrain"]["steps"]:
            raise ValueError("WSD requires the completed pretraining stage")
        if saved["world_size"] != world or saved["data_manifest_sha256"] != data_hash:
            raise ValueError("WSD preserves the pretraining world size and data stream")
        model_config = saved["model_config"]
        model = build_model(model_config).to(device).train()
        model.load_state_dict(saved["model"], strict=True)
    else:
        model, model_config = initialize_student(
            teacher, teacher_config, recipe, device
        )
    model.gradient_checkpointing = recipe["gradient_checkpointing"]
    opt = optimizer(model, recipe, stage)
    ema = EMA(model, recipe["ema_decay"])
    completed = 0
    if saved:
        opt.load_state_dict(saved["optimizer"])
        ema.load_state_dict(saved["ema"])
        sampler.load_state_dict(saved["sampler"])
        completed = saved["step"] if args.resume else 0
    if completed >= endpoint:
        raise ValueError("The requested endpoint must exceed the completed stage step")
    amp = bool(recipe["amp"] and device.startswith("cuda"))
    objective = Objective(model, teacher, amp=amp)
    if world > 1:
        objective = DistributedDataParallel(
            objective,
            device_ids=[torch.device(device).index]
            if device.startswith("cuda")
            else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    # DDP construction and model loading must not advance the resumed streams.
    if saved:
        restore_rng(saved["rng"][rank], streams, device)
    if rank == 0:
        if args.resume:
            if args.output.resolve() != Path(args.resume).resolve().parent:
                raise ValueError(
                    "Resume into the checkpoint's original output directory"
                )
        else:
            args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    if world > 1:
        dist.barrier()
    log = (args.output / "metrics.jsonl").open("a") if rank == 0 else None
    try:
        for step in range(completed + 1, endpoint + 1):
            started = time.perf_counter()
            base_lr = recipe[stage]["learning_rate"] * learning_rate_factor(
                recipe, stage, step
            )
            for group in opt.param_groups:
                group["lr"] = base_lr * (
                    recipe["pdd"]["head_lr_multiplier"]
                    if group["name"] == "heads"
                    else 1
                )
            tensors = batch_to_tensors(sampler.sample(), device)
            inputs = draw_inputs(model, tensors, recipe, stage, streams)
            local, inputs, fraction = partition(tensors, inputs, rank, world)
            opt.zero_grad(set_to_none=True)
            loss = objective(local, inputs)
            (loss * world * fraction).backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), recipe["max_grad_norm"], error_if_nonfinite=True
            )
            global_loss = loss.detach() * fraction
            if world > 1:
                dist.all_reduce(global_loss)
            if not torch.isfinite(global_loss):
                raise FloatingPointError("Nonfinite training loss")
            opt.step()
            ema.update(model)
            if rank == 0 and (step % args.log_every == 0 or step == endpoint):
                record = {
                    "stage": stage,
                    "step": step,
                    "loss": float(global_loss),
                    "grad_norm": float(norm),
                    "learning_rate": base_lr,
                    "seconds": time.perf_counter() - started,
                }
                line = json.dumps(record, allow_nan=False)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            if step % args.checkpoint_every == 0 or step == endpoint:
                state = capture_rng(streams, device)
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, state)
                else:
                    states[0] = state
                if rank == 0:
                    payload = {
                        "format": "pathfm_training_v1",
                        "stage": stage,
                        "step": step,
                        "recipe": recipe,
                        "model_config": model_config,
                        "model": cpu_state(model),
                        "optimizer": opt.state_dict(),
                        "ema": ema.state_dict(),
                        "sampler": sampler.state_dict(),
                        "rng": states,
                        "world_size": world,
                        "data_manifest_sha256": data_hash,
                        "initial_checkpoint": str(initial.resolve())
                        if initial
                        else None,
                        "teacher_sha256": teacher_identity,
                    }
                    atomic_save(payload, args.output / "checkpoint.pt")
                    del payload
                    if step == endpoint:
                        export(model, model_config, ema, args.output / "export")
                if world > 1:
                    dist.barrier()
    finally:
        if log:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pretrain", "wsd", "pdd"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, required=True, help="Prepared trajectories/ directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--init",
        type=Path,
        help="Full pretraining checkpoint for WSD; WSD weights for PDD",
    )
    parser.add_argument(
        "--resume", type=Path, help="Full checkpoint of this same stage"
    )
    parser.add_argument(
        "--backbone", help="Local pinned Chronos-2 snapshot for offline pretraining"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stage step at which to stop; preserves the full LR schedule",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.log_every < 1 or args.checkpoint_every < 1:
        parser.error("Log and checkpoint intervals must be positive")
    run(args)
