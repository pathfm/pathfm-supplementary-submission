# PathFM anonymous release

We release the PathFM training and inference code, together with forecasting benchmarks.

## Install

Use Python 3.11 or 3.12 from this directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m pip install ./train
```

For the benchmark entry points and interactive notebook:

```bash
python -m pip install ".[benchmarks,notebooks]"
```

`requirements.txt` records the CUDA 13 environment used for the prepared release.
The package installation above also supports CPU inference.
Installing dependencies and downloading the public datasets used in the examples requires network access.
Inference uses local model folders in the examples below.

## Checkpoints and quick inference

Checkpoint configurations are included on `main`.
Weight files are stored separately using Git LFS on the [checkpoints branch](https://github.com/pathfm/pathfm-supplementary-submission/tree/checkpoints).
A normal clone checks out `main` without downloading the checkpoint weights.
Download the desired `model.safetensors` from the table below and save it in the matching local folder.
Training also produces loadable model folders under each run's `export/` directory.

| Local folder | Model | Default decoder | Weight size | Download |
|---|---|---|---:|---|
| `weights/general-125m` | Updated PathFM-125M | Parallel, 1 evaluation | 500 MB | [Weights](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/general-125m/model.safetensors) |
| `weights/general-300m` | Updated PathFM-300M | Parallel, 1 evaluation | 1.18 GB | [Weights](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/general-300m/model.safetensors) |
| `weights/chaotic-75m` | PathFM-Chaotic-75M | Euler, 10 evaluations | 294.6 MB | [Weights](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/chaotic-75m/model.safetensors) |
| `weights/chaotic-125m` | PathFM-Chaotic-125M | Euler, 10 evaluations | 495.3 MB | [Weights](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/chaotic-125m/model.safetensors) |

The links use the repository's access permissions.
The [checkpoint guide](https://github.com/pathfm/pathfm-supplementary-submission/tree/checkpoints#readme) also provides instructions for downloading all models with Git LFS.

```bash
python smoke_test.py --model weights/general-125m --device cpu
```

```python
import torch
from pathfm import PathFMPipeline

torch.manual_seed(42)
model = PathFMPipeline.from_pretrained(
    "weights/general-125m", device="cpu", local_files_only=True,
    source_sampling="stratified", num_samples=32, seed=42,
)
context = torch.sin(torch.arange(512, dtype=torch.float32) / 10)[None, None, :]
paths = model.predict_samples(context, prediction_length=128)
print(paths[0].shape)  # [channels, paths, horizon]
totals = paths[0].sum(dim=-1)
peaks = paths[0].amax(dim=-1)
```

Inputs have shape `[batch, channels, context]`.
Native forecast limits are 480 steps for general models and 256 for chaotic models.
`rollout` continues each sampled path independently for longer horizons.
Both `source_sampling="stratified"` and `source_sampling="iid"` are supported.
The FEV/FPS benchmark entry point uses stratified ensembles.

For chaotic forecasts and long continuations:

```bash
python examples/chaotic_rollout.py --model weights/chaotic-125m   --context context.npy --horizon 4096 --device cuda --seed 42 --output rollout.npz
jupyter lab notebooks/chaotic_forecasting.ipynb
```

The notebook includes a three-channel Rössler example with a 512-step history and a 4,096-step observed continuation.

## Training code

The training code covers optimization, flow matching, PDD, EMA weights, checkpointing, and inference export.
The [training guide](train/README.md) demonstrates the workflow on chaotic-system forecasting.
Run the example from this directory:

```bash
python -m pathfm_train.prepare_data --output datasets/panda-train

torchrun --standalone --nproc_per_node=2 -m pathfm_train pretrain   --config train/configs/75m.json   --data-root datasets/panda-train/trajectories --output runs/75m/pretrain
```

The preparer downloads a pinned public PANDA-72M reproduction corpus and verifies the three Parquet checksums.
It preserves trajectory identities and holds out the ten parent-system families specified in `pathfm_train/data.py`.
The prepared corpus has 190,006 trajectories, of which 156,784 belong to `development_train`.
In this example, pretraining preserves the pinned Chronos-2 input embeddings and reinitializes the transformer, as specified in the training instructions.
The example data preparation, sampler, and model configurations are included alongside the training code.

## Benchmarks

```bash
python -m pathfm.benchmarks.download fev --output datasets/fev
python -m pathfm.benchmarks.fev --model weights/general-125m   --data-root datasets/fev --output outputs/fev-125m --device cuda --save-samples
python -m pathfm.benchmarks.fps --fev-output outputs/fev-125m

python -m pathfm.benchmarks.download panda --output datasets/panda-test
python -m pathfm.benchmarks.panda --model weights/chaotic-125m   --data-root datasets/panda-test --output outputs/panda-125m --device cuda
```

FPS scores the saved FEV paths using the empirical ensemble estimators implemented in this release.
Its standard command produces the 94-task common-support score; this command is distinct from additional rebuttal sensitivity analyses.
PANDA short-horizon evaluation uses the pinned test splits, 512-step histories, 16 stratified paths, and ten-step Euler sampling.
TIME and GIFT are available through `pathfm.benchmarks.time`, `pathfm.benchmarks.gift`, and the corresponding download subcommands.
Use `--help` on each entry point for its full interface.
Use a separate output directory for each model and protocol.

## File structure

- `src/pathfm/` contains the model, sampler, and benchmark implementations.
- `train/` contains the PathFM training code and the chaotic-system example recipes.
- `CHECKPOINTS.json` records the exported weight hashes.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) provides the bundled protocol licenses.
