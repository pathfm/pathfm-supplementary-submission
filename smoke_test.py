"""Check local checkpoint loading, both sampling modes, and independent rollout."""

import argparse
import json
from pathlib import Path

import torch
from pathfm import PathFMPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    time = torch.arange(64, dtype=torch.float32)
    history = torch.stack([torch.sin(time / 7), torch.cos(time / 11)])[None]
    modes = {}
    for sampling in ("stratified", "iid"):
        model = PathFMPipeline.from_pretrained(
            args.model, device=args.device, local_files_only=True,
            source_sampling=sampling, num_samples=4, seed=42,
        )
        model.set_stream(0)
        samples = model.predict_samples(history, prediction_length=16)
        assert len(samples) == 1 and samples[0].shape == (2, 4, 16)
        assert torch.isfinite(samples[0]).all()
        model.set_stream(0)
        repeated = model.predict_samples(history, prediction_length=16)
        torch.testing.assert_close(samples[0], repeated[0], rtol=0, atol=0)
        quantiles, points = model.predict_quantiles(
            history, prediction_length=16, quantile_levels=[0.1, 0.5, 0.9],
        )
        assert quantiles[0].shape == (2, 16, 3)
        assert points[0].shape == (2, 16)
        assert torch.isfinite(quantiles[0]).all() and torch.isfinite(points[0]).all()
        # Cross the checkpoint's native boundary to exercise path-specific feedback.
        rollout_horizon = model.flow.max_horizon + 16
        rollout = model.rollout(history, prediction_length=rollout_horizon, num_samples=4, seed=42)
        assert rollout.samples.shape == (1, 4, 2, rollout_horizon)
        assert torch.isfinite(rollout.samples).all()
        modes[sampling] = {
            "sample_shape": list(samples[0].shape),
            "quantile_shape": list(quantiles[0].shape),
            "rollout_shape": list(rollout.samples.shape),
            "stream_replay_exact": True,
            "steps": model.ode_steps,
            "sampler": model.sampler_method,
        }
    result = {"status": "passed", "model": args.model.name, "device": args.device, "sampling": modes}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
