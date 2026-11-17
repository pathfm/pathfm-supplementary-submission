"""PANDA-style online trajectory sampling."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np

CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 128
FIXED_CHANNELS = 3
MAX_CHANNELS = 16
VALIDATION_PARENTS = frozenset(
    {
        "BickleyJet",
        "Chua",
        "LidDrivenCavityFlow",
        "LiuChen",
        "PehlivanWei",
        "RayleighBenard",
        "SprottC",
        "SprottH",
        "SprottN",
        "VallisElNino",
    }
)
StreamName = Literal[
    "trajectory",
    "augmentation",
    "channel",
    "origin",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def safe_standardize(values: np.ndarray) -> np.ndarray:
    """Match PANDA's per-channel full-trajectory standardization."""
    mean = np.nanmean(values, axis=-1, keepdims=True)
    scale = np.nanstd(values, axis=-1, keepdims=True)
    scale = np.where(scale < 1e-10, 1e-10, scale)
    return (values - mean) / scale


def random_takens_embedding(
    values: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    lag = int(generator.integers(1, 11))
    channel = int(generator.integers(0, values.shape[0]))
    length = values.shape[1] - lag * (values.shape[0] - 1)
    return np.stack(
        [
            values[channel, index * lag : index * lag + length]
            for index in range(values.shape[0])
        ]
    )


def random_convex_combination(
    values: np.ndarray,
    generator: np.random.Generator,
    *,
    minimum_dimensions: int = 3,
) -> np.ndarray:
    dimensions = int(
        generator.integers(
            max(3, minimum_dimensions),
            max(9, minimum_dimensions + 1),
        )
    )
    coefficients = generator.dirichlet(
        np.ones(values.shape[0]),
        size=dimensions,
    )
    return coefficients @ values


def random_affine_transform(
    values: np.ndarray,
    generator: np.random.Generator,
    *,
    minimum_dimensions: int = 3,
) -> np.ndarray:
    dimensions = int(
        generator.integers(
            max(3, minimum_dimensions),
            max(9, minimum_dimensions + 1),
        )
    )
    transform = generator.normal(size=(dimensions, 1 + values.shape[0])) / np.sqrt(
        dimensions
    )
    return transform[:, :-1] @ values + transform[:, -1, None]


def transform_trajectory(
    values: np.ndarray,
    *,
    augmentation_generator: np.random.Generator,
    channel_generator: np.random.Generator,
    origin_generator: np.random.Generator,
    augmentation_rate: float,
    num_channels: int = FIXED_CHANNELS,
    prediction_length: int = PREDICTION_LENGTH,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply PANDA transforms and draw one 512-to-``prediction_length``
    training instance."""
    if prediction_length < 1:
        raise ValueError("prediction_length must be positive")
    if not 1 <= num_channels <= MAX_CHANNELS:
        raise ValueError(f"num_channels must lie between 1 and {MAX_CHANNELS}")
    if values.shape[0] < num_channels:
        raise ValueError(
            f"Source has {values.shape[0]} channels, fewer than "
            f"requested {num_channels}"
        )
    augmentation = "none"
    transformed = np.asarray(values)
    if augmentation_generator.random() < augmentation_rate:
        augmentation_index = int(augmentation_generator.integers(0, 3))
        if augmentation_index == 0:
            augmentation = "takens"
            transformed = random_takens_embedding(
                transformed,
                augmentation_generator,
            )
        elif augmentation_index == 1:
            augmentation = "convex"
            transformed = random_convex_combination(
                transformed,
                augmentation_generator,
                minimum_dimensions=num_channels,
            )
        else:
            augmentation = "affine"
            transformed = random_affine_transform(
                transformed,
                augmentation_generator,
                minimum_dimensions=num_channels,
            )
    transformed = safe_standardize(transformed)
    transformed_channels = int(transformed.shape[0])
    selected_channels = channel_generator.choice(
        transformed_channels,
        size=num_channels,
        replace=False,
    )
    transformed = transformed[selected_channels]
    latest_origin = transformed.shape[-1] - prediction_length
    if latest_origin < CONTEXT_LENGTH:
        raise ValueError("Transformed trajectory is too short for the task")
    origin = int(
        origin_generator.integers(
            CONTEXT_LENGTH,
            latest_origin + 1,
        )
    )
    context = transformed[:, origin - CONTEXT_LENGTH : origin]
    target = transformed[:, origin : origin + prediction_length]
    metadata = {
        "augmentation": augmentation,
        "selected_channels": selected_channels.astype(np.int64),
        "origin": origin,
        "transformed_channels": transformed_channels,
        "transformed_length": int(transformed.shape[-1]),
        "num_channels": num_channels,
    }
    return (
        context.astype(np.float32, copy=False),
        target.astype(np.float32, copy=False),
        metadata,
    )


class TrajectoryStore:
    """Read immutable channel-count shards of a trajectory corpus Parquet."""

    def __init__(self, root: Path):
        self.root = root
        with (root / "manifest.json").open() as file:
            self.manifest = json.load(file)
        if self.manifest.get("schema") != "panda_trajectory_store":
            raise ValueError(f"{root} is not a PANDA trajectory store")
        self.shards: dict[int, dict[str, np.ndarray]] = {}
        self.entries: list[tuple[int, int]] = []
        self.split_entries: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.system_entries: dict[
            str,
            dict[str, list[tuple[int, int]]],
        ] = defaultdict(lambda: defaultdict(list))
        for key, metadata in sorted(
            self.manifest["shards"].items(),
            key=lambda item: int(item[0]),
        ):
            channels = int(key)
            shard = {
                "values": np.load(root / metadata["values"], mmap_mode="r"),
                "row_index": np.load(
                    root / metadata["row_index"],
                    mmap_mode="r",
                ),
                "source_directory": np.load(
                    root / metadata["source_directory"],
                    mmap_mode="r",
                ),
                "source_filename": np.load(
                    root / metadata["source_filename"],
                    mmap_mode="r",
                ),
                "split": np.load(root / metadata["split"], mmap_mode="r"),
            }
            if shard["values"].shape != (
                metadata["trajectories"],
                channels,
                metadata["timepoints"],
            ):
                raise ValueError(f"Shard c{channels} has the wrong shape")
            self.shards[channels] = shard
            for local_index in range(metadata["trajectories"]):
                entry = (channels, local_index)
                split = str(shard["split"][local_index])
                system = str(shard["source_directory"][local_index])
                self.entries.append(entry)
                self.split_entries[split].append(entry)
                self.system_entries[split][system].append(entry)
                if split != "all_train":
                    self.split_entries["all_train"].append(entry)
                    self.system_entries["all_train"][system].append(entry)
        self.width_entries: dict[
            str,
            dict[int, tuple[tuple[int, int], ...]],
        ] = defaultdict(dict)
        self.width_system_entries: dict[
            str,
            dict[int, dict[str, tuple[tuple[int, int], ...]]],
        ] = defaultdict(dict)
        for split, entries in self.split_entries.items():
            by_width: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for entry in entries:
                by_width[entry[0]].append(entry)
            for width, width_entries in sorted(by_width.items()):
                by_system: dict[str, list[tuple[int, int]]] = defaultdict(list)
                for entry in width_entries:
                    by_system[
                        str(self.shards[entry[0]]["source_directory"][entry[1]])
                    ].append(entry)
                self.width_entries[split][width] = tuple(width_entries)
                self.width_system_entries[split][width] = {
                    system: tuple(system_entries)
                    for system, system_entries in by_system.items()
                }

    def values(self, entry: tuple[int, int]) -> np.ndarray:
        channels, local_index = entry
        return self.shards[channels]["values"][local_index]

    def metadata(self, entry: tuple[int, int]) -> dict[str, Any]:
        channels, local_index = entry
        shard = self.shards[channels]
        return {
            "row_index": int(shard["row_index"][local_index]),
            "source_directory": str(shard["source_directory"][local_index]),
            "source_filename": str(shard["source_filename"][local_index]),
            "source_channels": channels,
            "split": str(shard["split"][local_index]),
        }


class PandaOnlineSampler:
    """Stateful PANDA-style sampler with independent reproducible streams."""

    STREAM_OFFSETS: dict[StreamName, int] = {
        "trajectory": 0,
        "augmentation": 1,
        "channel": 2,
        "origin": 3,
    }

    def __init__(
        self,
        store: TrajectoryStore,
        *,
        split: str,
        seed: int = 42,
        augmentation_rate: float = 0.2,
        weighting: Literal["system", "trajectory"] = "system",
        sampling_mode: Literal["online", "fixed"] = "online",
        prediction_length: int = PREDICTION_LENGTH,
        batch_rows: int | None = None,
    ):
        if prediction_length < 1:
            raise ValueError("prediction_length must be positive")
        if batch_rows is not None and batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        if not 0 <= augmentation_rate <= 1:
            raise ValueError("augmentation_rate must be between zero and one")
        if weighting not in ("system", "trajectory"):
            raise ValueError("weighting must be system or trajectory")
        if sampling_mode not in ("online", "fixed"):
            raise ValueError("sampling_mode must be online or fixed")
        if sampling_mode == "fixed" and augmentation_rate != 0:
            raise ValueError("fixed sampling requires augmentation_rate=0")
        if not store.split_entries.get(split):
            raise ValueError(f"Split {split!r} is empty")
        self.store = store
        self.split = split
        self.seed = seed
        self.augmentation_rate = augmentation_rate
        self.weighting = weighting
        self.sampling_mode = sampling_mode
        self.prediction_length = int(prediction_length)
        self.batch_rows = None if batch_rows is None else int(batch_rows)
        self.channel_widths = tuple(sorted(store.width_entries[split]))
        self.generators = {
            name: np.random.default_rng(seed + offset)
            for name, offset in self.STREAM_OFFSETS.items()
        }
        self.systems = np.asarray(
            sorted(store.system_entries[split]),
            dtype=np.str_,
        )
        self.entries = tuple(store.split_entries[split])
        self.samples = 0
        self.augmentation_counts: Counter[str] = Counter()
        self.source_channel_counts: Counter[int] = Counter()
        self.batch_channel_widths: Counter[int] = Counter()
        self.system_counts: Counter[str] = Counter()
        self.fixed_examples: dict[
            tuple[tuple[int, int], int],
            tuple[np.ndarray, np.ndarray, dict[str, Any]],
        ] = {}

    def _draw_entry(self, num_channels: int | None = None) -> tuple[int, int]:
        """Draw a trajectory, optionally restricted to one channel width.

        With ``num_channels=None`` the draw ranges over the whole split, so a
        width is chosen implicitly in proportion to its trajectories (under
        trajectory weighting) or its source systems (under system weighting).
        """

        generator = self.generators["trajectory"]
        if self.weighting == "trajectory":
            entries = (
                self.store.split_entries[self.split]
                if num_channels is None
                else self.store.width_entries[self.split][num_channels]
            )
            return entries[int(generator.integers(len(entries)))]
        if num_channels is None:
            system_entries = self.store.system_entries[self.split]
        else:
            system_entries = self.store.width_system_entries[self.split][num_channels]
        systems = tuple(sorted(system_entries))
        system = systems[int(generator.integers(len(systems)))]
        entries = system_entries[system]
        return entries[int(generator.integers(len(entries)))]

    def sample(self, batch_size: int | None = None) -> dict[str, Any]:
        """Draw one batch, sized either in trajectories or in channel rows.

        A batch is filled from a single native width, so a trajectory count
        means a different amount of work at every width. Constructing with
        ``batch_rows`` instead spends a row budget: the trajectory count is
        whatever that width affords, and activation memory is the same
        whichever width is drawn.
        """

        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size is None and self.batch_rows is None:
            raise ValueError(
                "sample() needs a trajectory count, or a sampler built with batch_rows"
            )
        first_entry = self._draw_entry()
        num_channels = first_entry[0]
        if self.batch_rows is not None:
            affordable = max(1, self.batch_rows // num_channels)
            batch_size = (
                affordable if batch_size is None else min(batch_size, affordable)
            )
        contexts = []
        targets = []
        rows = []
        systems = []
        origins = []
        augmentations = []
        selected_channels = []
        source_channels = []
        transformed_channels = []
        transformed_lengths = []
        for position in range(batch_size):
            entry = first_entry if position == 0 else self._draw_entry(num_channels)
            source = self.store.metadata(entry)
            if self.sampling_mode == "online":
                context, target, metadata = transform_trajectory(
                    self.store.values(entry),
                    augmentation_generator=self.generators["augmentation"],
                    channel_generator=self.generators["channel"],
                    origin_generator=self.generators["origin"],
                    augmentation_rate=self.augmentation_rate,
                    num_channels=num_channels,
                    prediction_length=self.prediction_length,
                )
            else:
                fixed_key = (entry, num_channels)
                if fixed_key not in self.fixed_examples:
                    row_seed = self.seed + 10_000 + source["row_index"] * 4
                    self.fixed_examples[fixed_key] = transform_trajectory(
                        self.store.values(entry),
                        augmentation_generator=np.random.default_rng(row_seed + 1),
                        channel_generator=np.random.default_rng(row_seed + 2),
                        origin_generator=np.random.default_rng(row_seed + 3),
                        augmentation_rate=0.0,
                        num_channels=num_channels,
                        prediction_length=self.prediction_length,
                    )
                context, target, metadata = self.fixed_examples[fixed_key]
            contexts.append(context)
            targets.append(target)
            rows.append(source["row_index"])
            systems.append(source["source_directory"])
            origins.append(metadata["origin"])
            augmentations.append(metadata["augmentation"])
            selected_channels.append(metadata["selected_channels"])
            source_channels.append(source["source_channels"])
            transformed_channels.append(metadata["transformed_channels"])
            transformed_lengths.append(metadata["transformed_length"])
            self.samples += 1
            self.augmentation_counts[metadata["augmentation"]] += 1
            self.source_channel_counts[source["source_channels"]] += 1
            self.batch_channel_widths[num_channels] += 1
            self.system_counts[source["source_directory"]] += 1
        return {
            "context": np.stack(contexts),
            "target": np.stack(targets),
            "row_index": np.asarray(rows, dtype=np.int64),
            "source_directory": np.asarray(systems, dtype=np.str_),
            "origin": np.asarray(origins, dtype=np.int64),
            "augmentation": np.asarray(augmentations, dtype=np.str_),
            "selected_channels": np.stack(selected_channels),
            "source_channels": np.asarray(
                source_channels,
                dtype=np.int64,
            ),
            "transformed_channels": np.asarray(
                transformed_channels,
                dtype=np.int64,
            ),
            "transformed_length": np.asarray(
                transformed_lengths,
                dtype=np.int64,
            ),
            "num_channels": num_channels,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "panda_online_sampler_state",
            "schema_version": 2,
            "split": self.split,
            "seed": self.seed,
            "augmentation_rate": self.augmentation_rate,
            "weighting": self.weighting,
            "sampling_mode": self.sampling_mode,
            "channel_widths": list(self.channel_widths),
            "samples": self.samples,
            "generators": {
                name: deepcopy(generator.bit_generator.state)
                for name, generator in self.generators.items()
            },
            "augmentation_counts": dict(self.augmentation_counts),
            "source_channel_counts": dict(self.source_channel_counts),
            "batch_channel_widths": dict(self.batch_channel_widths),
            "system_counts": dict(self.system_counts),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schema": "panda_online_sampler_state",
            "schema_version": 2,
            "split": self.split,
            "seed": self.seed,
            "augmentation_rate": self.augmentation_rate,
            "weighting": self.weighting,
            "sampling_mode": self.sampling_mode,
            "channel_widths": list(self.channel_widths),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Sampler state {key} must be {value!r}, got {state.get(key)!r}"
                )
        for name, generator_state in state["generators"].items():
            self.generators[name].bit_generator.state = generator_state
        self.samples = int(state["samples"])
        self.augmentation_counts = Counter(state.get("augmentation_counts", {}))
        self.source_channel_counts = Counter(
            {
                int(key): value
                for key, value in state.get(
                    "source_channel_counts",
                    {},
                ).items()
            }
        )
        self.batch_channel_widths = Counter(
            {
                int(key): value
                for key, value in state.get(
                    "batch_channel_widths",
                    {},
                ).items()
            }
        )
        self.system_counts = Counter(state.get("system_counts", {}))

    def diagnostics(self) -> dict[str, Any]:
        augmented = self.samples - self.augmentation_counts.get("none", 0)
        return _jsonable(
            {
                "samples": self.samples,
                "augmentation_rate_observed": (
                    augmented / self.samples if self.samples else None
                ),
                "augmentation_counts": dict(self.augmentation_counts),
                "source_channel_counts": dict(self.source_channel_counts),
                "batch_channel_widths": dict(self.batch_channel_widths),
                "unique_systems_seen": len(self.system_counts),
                "system_sample_count_min": (
                    min(self.system_counts.values()) if self.system_counts else None
                ),
                "system_sample_count_max": (
                    max(self.system_counts.values()) if self.system_counts else None
                ),
            }
        )
