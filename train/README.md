# PathFM training code

We release the PathFM training code for flow-matching pretraining, WSD continuation, PDD, and checkpoint resume.
We use chaotic-system forecasting as a worked example, with 75M and 125M configurations.
Training uses the shared model implementation in `../src/pathfm/`.

## Example setup and data

From this release's root directory, use Python 3.11 or 3.12:

```bash
pip install -r requirements.txt
pip install . ./train
python -m pathfm_train.prepare_data --output datasets/panda-train
```

The preparer downloads a pinned PANDA-72M corpus revision and verifies all three Parquet checksums.
It preserves the original 190,006 trajectories, their order, and the held-out parent-system split; training uses the 156,784 `development_train` trajectories.
To use an existing download, add `--parquet-root /path/to/dataset-panda72M-repro`.

## Pretrain → WSD → PDD

Run these commands from the release root.
Use `GPUS=2 MODEL=75m` for the 75M recipe or `GPUS=4 MODEL=125m` for 125M.

```bash
GPUS=2
MODEL=75m
CONFIG=train/configs/$MODEL.json
DATA=datasets/panda-train/trajectories
RUN=runs/$MODEL

torchrun --standalone --nproc_per_node="$GPUS" -m pathfm_train pretrain \
  --config "$CONFIG" --data-root "$DATA" --output "$RUN/pretrain"

torchrun --standalone --nproc_per_node="$GPUS" -m pathfm_train wsd \
  --config "$CONFIG" --data-root "$DATA" --output "$RUN/wsd" \
  --init "$RUN/pretrain/checkpoint.pt"

torchrun --standalone --nproc_per_node="$GPUS" -m pathfm_train pdd \
  --config "$CONFIG" --data-root "$DATA" --output "$RUN/pdd" \
  --init "$RUN/wsd/checkpoint.pt"
```

Each stage writes a resumable `checkpoint.pt`, a compact `metrics.jsonl`, and inference-ready EMA weights under `export/`.
The released chaotic models use the WSD export with 10-step Euler; PDD exports default to two decoder calls.
PDD can also start directly from an undistilled model folder with `--init weights/chaotic-75m`.

## Example recipes

| Setting | Pretraining | WSD | PDD |
|---|---:|---:|---:|
| Steps | 2,000,000 | 100,000 | 100,000 |
| Learning rate | 0.0001 | 0.0001 → 0 | 0.00003 → 0 |
| Schedule | 500-step warmup, constant | Linear decay | Cosine decay |
| Training source scale | 1.0 | 1.0 | 0.86 |
| PDD head LR multiplier | — | — | 10 |

All stages use seed 42, context 512, horizon 256, a global budget of 384 channel rows, four flow draws, AdamW β=(0.9, 0.99), zero weight decay, gradient clipping at 1, and EMA decay 0.999.
Pretraining preserves the pinned Chronos-2 input embeddings and reinitializes the transformer.
Flow times are shared across each trajectory's channels and follow logit-normal(−0.8, 0.8).
Online augmentation has probability 0.2, with uniform sampling over source systems.
PDD uses 20 uniform intervals, block lengths {1, 2, 5, 10, 20}, and on-policy Euler teacher labels.

The example uses unit-scale noise for pretraining/WSD and a source scale of 0.86 for inference/PDD.
WSD carries forward raw weights, optimizer, EMA, sampler, and RNG state.
PDD starts a fresh optimizer and sampler from the WSD EMA weights.

## Code organization

`pathfm_train/engine.py` implements optimization, stage transitions, and resume.
`pathfm_train/objectives.py` implements flow matching and PDD.
`pathfm_train/checkpoint.py` handles EMA weights, optimizer and RNG state, and inference export.
The model architecture and inference sampler are shared with `src/pathfm/`.

The chaotic-system example uses the PANDA data preparer and online sampler in `pathfm_train/prepare_data.py`, `pathfm_train/build_trajectory_store.py`, and `pathfm_train/data.py`.
Its model and optimization settings are in `configs/75m.json` and `configs/125m.json`.
To use another dataset, adapt the data preparation and sampler to produce the context and target batches consumed by `engine.py`, and set the corresponding training configuration.

## Resume and inference

Resume with the same config, dataset, GPU count, and output directory:

```bash
torchrun --standalone --nproc_per_node="$GPUS" -m pathfm_train pretrain \
  --config "$CONFIG" --data-root "$DATA" --output "$RUN/pretrain" \
  --resume "$RUN/pretrain/checkpoint.pt"
```

Use the corresponding stage name and checkpoint for WSD or PDD.
`--stop-after N` stops at stage step N while preserving the full learning-rate schedule.
For an offline pretraining start, pass `--backbone /path/to/the/pinned/chronos-2/snapshot`.
The default checkpoint interval is 10,000 steps.

```python
from pathfm import PathFMPipeline

model = PathFMPipeline.from_pretrained("runs/75m/wsd/export", device="cuda")
```
