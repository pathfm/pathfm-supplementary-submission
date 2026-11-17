"""Download the pinned PANDA corpus and preserve its original split and row order."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import snapshot_download

from .build_trajectory_store import convert
from .checkpoint import digest

REPOSITORY = "GilpinLab/dataset-panda72M-repro"
REVISION = "f2c867ac28d0e99451a06c73ea994b4da1249fbb"
PARTS = (
    (
        "skew_mixedp_ic16",
        "skew_mixedp_ic16/train-part1.parquet",
        "a7e96ef5243fdd417bf06a5a9b453af2eda153031f0d664522119682f7549f7a",
    ),
    (
        "base_mixedp_ic16",
        "base_mixedp_ic16/train-part1.parquet",
        "a18210ea3127094b22f28ffd5b0fcfc86920e7b0fc775f75bd0aabaf38968e42",
    ),
    (
        "base40",
        "base40/train.parquet",
        "1aa89327b42f04a60d8c5d52ef058bb675e02b0424843afafc5095fc31d68377",
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parquet-root", type=Path, help="Already-downloaded repository snapshot"
    )
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    source = args.parquet_root or Path(
        snapshot_download(
            REPOSITORY,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=[filename for _, filename, _ in PARTS],
            cache_dir=args.cache_dir,
        )
    )
    parts = []
    for name, filename, expected in PARTS:
        file = source / filename
        if digest(file) != expected:
            raise ValueError(f"Training corpus checksum mismatch: {filename}")
        parts.append(f"{name}={file}")
    convert(
        SimpleNamespace(
            parquet=parts,
            dataset_root=args.output,
            batch_size=64,
            seed=42,
        )
    )


if __name__ == "__main__":
    main()
