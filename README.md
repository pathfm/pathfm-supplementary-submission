# PathFM checkpoints

This branch stores the PathFM checkpoints using Git LFS.
Training and inference code are on the [main branch](https://github.com/pathfm/pathfm-supplementary-submission/tree/main).
A normal clone checks out `main` and does not download the checkpoint payloads.
The four weights total approximately 2.47 GB.

| Model | Weight size | Download | Configuration |
|---|---:|---|---|
| Updated PathFM-125M | 500.0 MB | [Weights](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/general-125m/model.safetensors) | [Configuration](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/general-125m/config.json) |
| Updated PathFM-300M | 1,180.0 MB | [Weights](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/general-300m/model.safetensors) | [Configuration](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/general-300m/config.json) |
| PathFM-Chaotic-75M | 294.6 MB | [Weights](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/chaotic-75m/model.safetensors) | [Configuration](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/chaotic-75m/config.json) |
| PathFM-Chaotic-125M | 495.3 MB | [Weights](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/chaotic-125m/model.safetensors) | [Configuration](https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/chaotic-125m/config.json) |

Download the chosen `model.safetensors` into its matching `weights/` folder in the main-branch checkout.
The matching configurations are already included on `main`.

For example, from the main checkout:

```bash
curl --fail --location \
  https://github.com/pathfm/pathfm-supplementary-submission/raw/refs/heads/checkpoints/weights/general-125m/model.safetensors \
  --output weights/general-125m/model.safetensors
```

The links use the repository's access permissions.
You can also retrieve all four models by cloning this branch with Git LFS installed:

```bash
git lfs install
git clone --single-branch --branch checkpoints \
  https://github.com/pathfm/pathfm-supplementary-submission.git pathfm-checkpoints
```

`CHECKPOINTS.json` records model settings and SHA-256 hashes.
After downloading all files, `sha256sum -c SHA256SUMS` verifies their contents.
