"""Offline conversion, stage-transition, resume, and export smoke test.

Uses a random fixture and a reduced architecture, batch and horizon.
The production recipes are read but never modified.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pathfm import PathFMPipeline
from pathfm_train.build_trajectory_store import convert, parent_systems, split_for
from pathfm_train.data import PandaOnlineSampler, TrajectoryStore, VALIDATION_PARENTS


def equal(left, right, location="state"):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=location)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=location)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), location
        for key in left:
            equal(left[key], right[key], location + "/" + str(key))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), location
        for index, (a, b) in enumerate(zip(left, right)):
            equal(a, b, location + "/" + str(index))
    else:
        assert left == right, (location, left, right)


def prepare_fixture(work):
    rng = np.random.default_rng(42)
    parts = []
    expected = []
    for part in ("skew_mixedp_ic16", "base_mixedp_ic16", "base40"):
        values, shapes, directories, names = [], [], [], []
        for channels in (3, 4, 5, 6, 10):
            for system in ("Lorenz_Rossler-pp1", "Chua_Rossler-pp2"):
                value = rng.normal(size=(channels, 4096)).astype(np.float32)
                values.append(value.ravel().tolist())
                expected.append(value)
                shapes.append([channels, 4096])
                directories.append(system)
                names.append(f"{part}-{system}-c{channels}.npy")
                assert (split_for(system) == "validation") == bool(parent_systems(system) & VALIDATION_PARENTS)
        path = work / f"{part}.parquet"
        pq.write_table(pa.table({
            "target": values, "target._np_shape": shapes,
            "_source_directory": directories, "_source_filename": names,
        }), path)
        parts.append(f"{part}={path}")
    convert(SimpleNamespace(parquet=parts, dataset_root=work / "data", batch_size=4, seed=42))
    root = work / "data/trajectories"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["rows"] == 30
    assert manifest["split_counts"] == {"development_train": 15, "validation": 15}
    for channels in (3, 4, 5, 6, 10):
        stored = np.load(root / f"c{channels}_values.npy")
        indices = np.load(root / f"c{channels}_row_index.npy")
        for row, index in zip(stored, indices):
            np.testing.assert_array_equal(row, expected[index])
    sampler = PandaOnlineSampler(
        TrajectoryStore(root), split="development_train", seed=42,
        augmentation_rate=0.2, weighting="system", batch_rows=10, prediction_length=32,
    )
    seen = set()
    for _ in range(100):
        batch = sampler.sample()
        seen.add(batch["num_channels"])
    state = copy.deepcopy(sampler.state_dict())
    first = sampler.sample()
    sampler.load_state_dict(state)
    equal(first, sampler.sample(), "sampler_resume")
    assert seen == {3, 4, 5, 6, 10}
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--config", type=Path, default=Path("train/configs/75m.json"))
    args = parser.parse_args()
    if not (args.backbone / "config.json").is_file():
        parser.error("--backbone must be the local pinned Chronos-2 snapshot")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    data = prepare_fixture(work)
    recipe = json.loads(args.config.read_text())
    recipe.update(
        model_name="PathFM-Chaotic-Smoke", num_layers=3, ffn_dim=128,
        prediction_length=32, batch_rows=10, flow_draws=1, amp=False,
    )
    recipe["pretrain"].update(steps=2, warmup_steps=1)
    recipe["wsd"]["steps"] = 2
    recipe["pdd"]["steps"] = 2
    config = work / "smoke-config.json"
    config.write_text(json.dumps(recipe, indent=2) + "\n")
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads), HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
    )
    checked = []
    for stage in ("pretrain", "wsd", "pdd"):
        for branch in ("full", "resumed"):
            output = work / branch / stage
            command = [
                sys.executable, "-m", "pathfm_train", stage,
                "--config", str(config), "--data-root", str(data),
                "--output", str(output), "--device", args.device,
                "--log-every", "1", "--checkpoint-every", "1",
            ]
            if stage == "pretrain":
                command += ["--backbone", str(args.backbone.resolve())]
            else:
                previous = "pretrain" if stage == "wsd" else "wsd"
                command += ["--init", str(work / branch / previous / "checkpoint.pt")]
            first = command + (["--stop-after", "1"] if branch == "resumed" else [])
            with (work / f"{stage}-{branch}.log").open("w") as log:
                subprocess.run(first, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if branch == "resumed":
                with (work / f"{stage}-{branch}-resume.log").open("w") as log:
                    subprocess.run(
                        command + ["--resume", str(output / "checkpoint.pt")],
                        env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                    )
        left = torch.load(work / "full" / stage / "checkpoint.pt", map_location="cpu", weights_only=False)
        right = torch.load(work / "resumed" / stage / "checkpoint.pt", map_location="cpu", weights_only=False)
        for key in ("model", "optimizer", "ema", "sampler", "rng", "step"):
            equal(left[key], right[key], stage + "/" + key)
        assert all(torch.isfinite(value).all() for value in left["model"].values())
        del left, right
        gc.collect()
        model = PathFMPipeline.from_pretrained(
            work / "full" / stage / "export", device=args.device,
            local_files_only=True, num_samples=2, source_sampling="stratified",
        )
        history = torch.sin(torch.arange(64, dtype=torch.float32) / 10)[None, None]
        result = model.predict_samples(history, prediction_length=16)[0]
        assert result.shape == (1, 2, 16) and torch.isfinite(result).all()
        checked.append({"stage": stage, "resume_bitwise_exact": True, "export_inference_finite": True})
        print(json.dumps(checked[-1]), flush=True)
        del model
        gc.collect()
    result = {
        "status": "passed", "device": args.device,
        "fixture_trajectories": 30, "channel_widths": [3, 4, 5, 6, 10],
        "conversion_values_and_order_exact": True, "sampler_resume_exact": True,
        "stages": checked, "world_size": 1,
        "scope": "Reduced architecture, horizon, batch and step counts; production recipes unchanged.",
        "full_training_rerun": False,
    }
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
