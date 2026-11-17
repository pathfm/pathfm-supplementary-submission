"""Convert the three pinned PANDA Parquets without changing their row order."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .checkpoint import digest as sha256
from .data import VALIDATION_PARENTS

TIMEPOINTS = 4096


def parse_parquets(values: list[str]) -> list[tuple[str, Path]]:
    parts = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PARQUET, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or name in seen:
            raise ValueError(f"Invalid or duplicate part name {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parts.append((name, path))
        seen.add(name)
    return parts


def parent_systems(source_directory: str) -> frozenset[str]:
    """Base-system names of a base (one parent) or skew (two) directory."""

    trimmed = re.sub(r"-pp\d+$", "", source_directory)
    parents = frozenset(trimmed.split("_"))
    if not 1 <= len(parents) <= 2 or any(not parent for parent in parents):
        raise ValueError(f"Invalid source directory {source_directory!r}")
    return parents


def split_for(source_directory: str) -> str:
    if parent_systems(source_directory) & VALIDATION_PARENTS:
        return "validation"
    return "development_train"


def read_metadata(path: Path) -> tuple[list[tuple[int, int]], list[str], list[str]]:
    table = pq.read_table(
        path,
        columns=["target._np_shape", "_source_directory", "_source_filename"],
    )
    shapes = [tuple(value) for value in table["target._np_shape"].to_pylist()]
    directories = table["_source_directory"].to_pylist()
    filenames = table["_source_filename"].to_pylist()
    return shapes, directories, filenames


def convert(args) -> None:
    if args.seed != 42:
        raise ValueError("PANDA-72M repro preparation is locked to seed 42")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    parts = parse_parquets(args.parquet)
    output_root = args.dataset_root / "trajectories"
    if output_root.exists():
        raise FileExistsError(output_root)

    metadata = {name: read_metadata(path) for name, path in parts}
    all_shapes = [shape for shapes, _, _ in metadata.values() for shape in shapes]
    all_directories = [
        directory
        for _, directories, _ in metadata.values()
        for directory in directories
    ]
    all_filenames = [
        filename for _, _, filenames in metadata.values() for filename in filenames
    ]
    if any(shape[1] != TIMEPOINTS for shape in all_shapes):
        raise ValueError(f"Every repro trajectory must have length {TIMEPOINTS}")
    counts = Counter(int(shape[0]) for shape in all_shapes)

    output_root.mkdir(parents=True)
    writers = {}
    positions = Counter()
    for channels, count in sorted(counts.items()):
        writers[channels] = {
            "values": np.lib.format.open_memmap(
                output_root / f"c{channels}_values.npy",
                mode="w+",
                dtype=np.float32,
                shape=(count, channels, TIMEPOINTS),
            ),
            "row_index": np.lib.format.open_memmap(
                output_root / f"c{channels}_row_index.npy",
                mode="w+",
                dtype=np.int64,
                shape=(count,),
            ),
            "source_directory": np.lib.format.open_memmap(
                output_root / f"c{channels}_source_directory.npy",
                mode="w+",
                dtype=f"<U{max(map(len, all_directories))}",
                shape=(count,),
            ),
            "source_filename": np.lib.format.open_memmap(
                output_root / f"c{channels}_source_filename.npy",
                mode="w+",
                dtype=f"<U{max(map(len, all_filenames))}",
                shape=(count,),
            ),
            "split": np.lib.format.open_memmap(
                output_root / f"c{channels}_split.npy",
                mode="w+",
                dtype="<U17",
                shape=(count,),
            ),
        }

    part_ranges = {}
    row_index = 0
    for name, path in parts:
        shapes, directories, filenames = metadata[name]
        part_start = row_index
        parquet = pq.ParquetFile(path)
        local_row = 0
        for batch in parquet.iter_batches(
            batch_size=args.batch_size,
            columns=["target"],
        ):
            for flattened in batch.column(0).to_pylist():
                channels, timepoints = shapes[local_row]
                local_index = positions[channels]
                values = np.asarray(flattened, dtype=np.float32).reshape(
                    channels,
                    timepoints,
                )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"{name} row {local_row} contains non-finite values"
                    )
                writer = writers[channels]
                writer["values"][local_index] = values
                writer["row_index"][local_index] = row_index
                writer["source_directory"][local_index] = directories[local_row]
                writer["source_filename"][local_index] = filenames[local_row]
                writer["split"][local_index] = split_for(directories[local_row])
                positions[channels] += 1
                row_index += 1
                local_row += 1
                if row_index % 5000 == 0:
                    print(f"converted {row_index}/{len(all_shapes)}", flush=True)
        if local_row != len(shapes):
            raise RuntimeError(f"{name} row count changed during conversion")
        part_ranges[name] = {
            "rows": local_row,
            "row_index_start": part_start,
            "row_index_stop": row_index,
        }
    if row_index != len(all_shapes):
        raise RuntimeError("Total row count changed during conversion")
    for writer in writers.values():
        for array in writer.values():
            array.flush()

    split_counts = Counter(split_for(d) for d in all_directories)
    system_counts = {
        split: len({d for d in all_directories if split_for(d) == split})
        for split in split_counts
    }
    shards = {}
    for channels, count in sorted(counts.items()):
        files = {name: f"c{channels}_{name}.npy" for name in writers[channels]}
        shards[str(channels)] = {
            **files,
            "trajectories": count,
            "timepoints": TIMEPOINTS,
            "sha256": {
                name: sha256(output_root / path) for name, path in files.items()
            },
        }
    manifest = {
        "status": "complete",
        "schema": "panda_trajectory_store",
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "corpus": "GilpinLab/dataset-panda72M-repro",
        "parquets": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                **part_ranges[name],
            }
            for name, path in parts
        },
        "rows": len(all_shapes),
        "timepoints": TIMEPOINTS,
        "channel_counts": dict(sorted(counts.items())),
        "split_counts": dict(split_counts),
        "system_counts": system_counts,
        "validation_parents": sorted(VALIDATION_PARENTS),
        "shards": shards,
        "command": [sys.executable, *sys.argv],
    }
    with (output_root / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    print(
        "store complete:",
        json.dumps(
            {
                "rows": manifest["rows"],
                "channel_counts": manifest["channel_counts"],
                "split_counts": manifest["split_counts"],
                "system_counts": manifest["system_counts"],
            }
        ),
        flush=True,
    )
