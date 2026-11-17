"""Initialization and interval readouts for the released chaotic architecture."""

import copy
from pathlib import Path

import torch
from chronos import Chronos2Pipeline
from chronos.chronos2.model import Chronos2Model
from huggingface_hub import snapshot_download
from pathfm.loading import build_model
from pathfm.model import PathFM, QKNormProjection
from torch import nn


def reinitialize(backbone, seed):
    """Apply Chronos-2's fan-in initialization after clearing HF init guards."""
    for tensor in (*backbone.parameters(), *backbone.buffers()):
        tensor._is_hf_initialized = False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in reversed(list(backbone.modules())):
            backbone._init_weights(module)


def model_config(model, recipe, *, parallel=False):
    backbone = model.backbone.config.to_dict()
    for key in (
        "_name_or_path",
        "name_or_path",
        "transformers_version",
        "architectures",
    ):
        backbone.pop(key, None)
    return {
        "format_version": 1,
        "model_type": "pathfm",
        "name": recipe["model_name"],
        "family": "chaotic",
        "backbone": backbone,
        "input_embedding_hidden_dim": model.backbone.input_patch_embedding.hidden_layer.out_features,
        "architecture": {
            "max_output_patches": recipe["prediction_length"] // 16,
            "flow_entry_block": 2,
            "source_scale": recipe["source_scale"],
            "instance_scale_floor": 0.0001,
            "parallel_knots": list(model.parallel_knots) if parallel else None,
            "parallel_block_menu": list(model.parallel_block_menu)
            if parallel
            else None,
        },
        "inference_defaults": {
            "steps": 2 if parallel else 10,
            "sampler_method": "parallel" if parallel else "euler",
            "num_samples": 16,
            "source_sampling": "iid",
            "token_budget": 32768,
            "seed": recipe["seed"],
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def initialize(recipe, device, backbone_path=None):
    """Reinitialize the transformer, retaining the released input embeddings."""
    source = backbone_path or recipe["backbone"]
    if not Path(source).is_dir():
        source = snapshot_download(
            repo_id=source,
            revision=recipe["backbone_revision"],
            allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
        )
    backbone = Chronos2Pipeline.from_pretrained(source, device_map=device).model
    if (backbone.config.d_model, backbone.config.num_heads) != (768, 12):
        raise ValueError(
            "The recipes require the pinned 768-dimensional Chronos-2 backbone"
        )
    rebuilt = recipe["ffn_dim"] != backbone.config.d_ff
    if rebuilt:
        released = backbone
        config = copy.deepcopy(released.config)
        config.d_ff = recipe["ffn_dim"]
        config.num_layers = recipe["num_layers"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(recipe["seed"])
            backbone = Chronos2Model(config)
        reinitialize(backbone, recipe["seed"])
        backbone.input_patch_embedding = copy.deepcopy(released.input_patch_embedding)
        backbone.shared.load_state_dict(released.shared.state_dict())
    else:
        del backbone.encoder.block[recipe["num_layers"] :]
        backbone.config.num_layers = recipe["num_layers"]
        preserved = {
            name: value.detach().clone()
            for name, value in backbone.state_dict().items()
            if name.startswith(("input_patch_embedding.", "shared."))
        }
        reinitialize(backbone, recipe["seed"])
        backbone.load_state_dict(preserved, strict=False)
    backbone.config._attn_implementation = "sdpa"
    backbone.config.dropout_rate = 0.0
    for module in backbone.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if isinstance(getattr(module, "dropout", None), (int, float)):
            module.dropout = 0.0
    model = PathFM(
        backbone,
        max_output_patches=recipe["prediction_length"] // 16,
        flow_entry_block=2,
        source_scale=recipe["source_scale"],
        instance_scale_floor=0.0001,
    )
    for layer in (
        model.gate_projection,
        model.final_modulation,
        model.adaln_shared,
        model.clean_output_layer,
        model.clean_residual_layer,
    ):
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
    nn.init.constant_(model.gate_projection.bias, 0.1)
    nn.init.zeros_(model.adaln_block_offsets)
    for module in model.modules():
        if isinstance(module, QKNormProjection):
            nn.init.constant_(module.temperature, module.head_dim**-0.25)
    return model.to(device).train(), model_config(model, recipe)


def initialize_student(teacher, config, recipe, device):
    """Tile the WSD EMA velocity head onto the 20-interval PDD grid."""
    if teacher.parallel_knots is not None:
        raise ValueError("PDD requires an undistilled WSD teacher")
    config = copy.deepcopy(config)
    grid_size = recipe["pdd"]["grid_size"]
    config["architecture"].update(
        parallel_knots=[index / grid_size for index in range(grid_size + 1)],
        parallel_block_menu=recipe["pdd"]["block_menu"],
    )
    student = build_model(config).to(device)
    state = {
        name: value
        for name, value in teacher.state_dict().items()
        if not name.startswith(("clean_output_layer.", "clean_residual_layer."))
    }
    missing, unexpected = student.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith("parallel_") for name in missing):
        raise ValueError("Teacher and student architectures differ")
    with torch.no_grad():
        for clean, parallel in (
            (teacher.clean_output_layer, student.parallel_output_layer),
            (teacher.clean_residual_layer, student.parallel_residual_layer),
        ):
            parallel.weight.copy_(clean.weight.repeat(grid_size, 1))
            parallel.bias.copy_(clean.bias.repeat(grid_size))
    teacher.eval().requires_grad_(False)
    return student.train(), model_config(student, recipe, parallel=True)


def parallel_heads(model, states, times, packed):
    hidden = model.encode_packed_states(states, times, packed)
    rows, draws = hidden.shape[0], packed["num_draws"]
    forecast = hidden[:, packed["num_context_patches"] :].reshape(
        rows, draws, packed["appended_per_draw"], hidden.shape[-1]
    )[:, :, 1:]
    trunk = model.backbone.output_patch_embedding
    latent = trunk.act(trunk.hidden_layer(forecast))
    prediction = trunk.dropout(
        model.parallel_output_layer(latent)
    ) + model.parallel_residual_layer(forecast)
    return (
        prediction.reshape(
            rows,
            draws,
            packed["num_output_patches"],
            model.parallel_grid_size,
            model.patch_length,
        )
        .permute(1, 0, 3, 2, 4)
        .reshape(draws, rows, model.parallel_grid_size, packed["future_length"])
    )


def optimizer(model, recipe, stage):
    heads, fresh, backbone = [], [], []
    # Match the original groups; inactive quantile heads are absent in this package.
    fresh_modules = [model.flow_time_embedding]
    if model.parallel_knots is None:
        fresh_modules += [model.clean_output_layer, model.clean_residual_layer]
    fresh_modules += [model.gate_projection, model.adaln_shared, model.final_modulation]
    for module in fresh_modules:
        fresh.extend(module.parameters())
    fresh.append(model.adaln_block_offsets)
    for module in model.modules():
        if isinstance(module, QKNormProjection):
            fresh.append(module.temperature)
    if stage == "pdd":
        heads = list(model.parallel_output_layer.parameters()) + list(
            model.parallel_residual_layer.parameters()
        )
    new_ids = {id(parameter) for parameter in [*fresh, *heads]}
    backbone = [
        parameter for parameter in model.parameters() if id(parameter) not in new_ids
    ]
    lr = recipe[stage]["learning_rate"]
    groups = [
        {"params": backbone, "name": "backbone", "lr": lr},
        {"params": fresh, "name": "flow", "lr": lr},
    ]
    if heads:
        groups.append(
            {
                "params": heads,
                "name": "heads",
                "lr": lr * recipe["pdd"]["head_lr_multiplier"],
            }
        )
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=tuple(recipe["adam_betas"]),
        weight_decay=recipe["weight_decay"],
    )
