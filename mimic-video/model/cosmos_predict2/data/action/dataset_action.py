import collections
import contextlib
import copy
import hashlib
import inspect
import json
import os
import pathlib
from functools import partial
from operator import methodcaller

import numpy as np
import threadpoolctl
import torch
import tqdm

import cosmos_predict2.data.action.types as data_spec
from cosmos_predict2.data.action import chunk_reader
from cosmos_predict2.data.action import data_transforms as data_transforms_mod
from cosmos_predict2.data.action.data_transforms import apply_data_transforms, make_data_transforms
from cosmos_predict2.data.action.types import S_TO_NS, LieRepr, NormalizationType, ObsMeta, ObsType
from cosmos_predict2.data.action.utils import dict_apply, get_paths
from cosmos_predict2.module import normalizer
from cosmos_predict2.module.normalizer import array_to_stats


class MimicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        *,
        timestep_anchor: str,
        data_components: dict,
        data_transforms: list[dict],
        policy_io: dict,
        source_component_names: dict,
        should_include_padded_tails: bool,
        seed: int = 42,
        num_val_episodes: int = 1,
        val_episode_ranges: list[dict] | None = None,
        val_ratio: float | None = None,
        train: bool = True,
        verbose: bool = False,
    ) -> None:
        self._data_dir = pathlib.Path(data_dir)
        self._episode_paths = get_paths(self._data_dir, verbose=verbose)

        def get_source_component(key: str, spec: dict, prefix: str) -> tuple[str, ObsMeta]:
            source_name = source_component_names.get(f"{prefix}/{key}", key)
            return f"{prefix}/{source_name}", spec | data_components[source_name]

        self._data_components = dict(
            get_source_component(key, spec, category)
            for category, values in policy_io.items()
            for key, spec in values.items()
        )

        self._normalized_data_components = {
            dc.split("/")[1]
            for dc, meta in self._data_components.items()
            if meta["normalization_type"] is not NormalizationType.NONE
        }

        self._data_transforms = list(make_data_transforms(data_transforms, self._data_components))

        reader_data_components = copy.deepcopy(self._data_components)

        val_episode_ranges_ns = self.normalize_val_episode_ranges(val_episode_ranges)
        if val_episode_ranges_ns:
            val_episode_names = set(val_episode_ranges_ns)
            available_episode_names = {path.name for path in self._episode_paths}
            missing_episode_names = sorted(val_episode_names - available_episode_names)
            if missing_episode_names:
                raise ValueError(f"Validation range episodes are missing from {self._data_dir}: {missing_episode_names}")
            val_mask = np.array([path.name in val_episode_names for path in self._episode_paths], dtype=bool)
            episode_time_ranges_ns = None if train else val_episode_ranges_ns
        elif val_ratio is not None and val_ratio > 0:
            # Hash-based stable split keyed on episode basename, same scheme as
            # dataset_video.Dataset: deterministic across runs and across the
            # multi-source mix (each `sourceNNN_episode_XXX.zarr` hashes the
            # same way every time).
            val_mask = self.get_val_mask_by_hash(self._episode_paths, val_ratio)
            episode_time_ranges_ns = None
        else:
            val_mask = self.get_val_mask(
                n_episodes=len(self._episode_paths), n_val_episodes=num_val_episodes, seed=seed
            )
            episode_time_ranges_ns = None
        val_mask = ~val_mask if train else val_mask
        stats_val_episode_ranges = tuple(
            (episode_name, tuple(ranges)) for episode_name, ranges in sorted(val_episode_ranges_ns.items())
        )

        self._stats_id = hashlib.sha256(
            "".join(
                (
                    str(self._data_components),
                    str(data_transforms),
                    *map(str, self._episode_paths),
                    str(timestep_anchor),
                    str(should_include_padded_tails),
                    str(seed),
                    str(num_val_episodes),
                    str(stats_val_episode_ranges),
                    inspect.getsource(chunk_reader),
                    inspect.getsource(data_transforms_mod),
                    inspect.getsource(data_spec),
                    inspect.getsource(normalizer),
                )
            ).encode("utf-8")
        ).hexdigest()

        self._chunk_reader = chunk_reader.ChunkReader(
            self._episode_paths,
            data_components=reader_data_components,
            timestep_anchor=timestep_anchor,
            should_include_padded_tails=should_include_padded_tails,
            episode_mask=val_mask,
            episode_time_ranges_ns=episode_time_ranges_ns,
            verbose=verbose,
            stats_id=self._stats_id,
            data_dir=self.data_dir,
        )

        self._threadpool_limits_is_applied = False
        self._should_ignore_transforms_for_norm = False

    @property
    def data_dir(self) -> pathlib.Path:
        return self._data_dir

    @property
    def stats_id(self) -> str:
        return self._stats_id

    @contextlib.contextmanager
    def restrict_chunk_reader(self):
        self._chunk_reader.restrict_keys(self._normalized_data_components)
        try:
            yield self
        finally:
            self._chunk_reader.restrict_keys(None)

    @contextlib.contextmanager
    def ignore_transforms(self):
        should_ignore_transform_ops = self._should_ignore_transforms_for_norm
        self._should_ignore_transforms_for_norm = True
        try:
            yield self
        finally:
            self._should_ignore_transforms_for_norm = should_ignore_transform_ops

    def _compute_statistics(self) -> dict:
        data_cache = collections.defaultdict(list)

        with self.restrict_chunk_reader() as normed_keys, normed_keys.ignore_transforms() as dataset:
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=(os.cpu_count() or 4) // 4,
                num_workers=(os.cpu_count() or 4) // 4,
            )
            for batch in tqdm.tqdm(dataloader, desc="Iterating dataset to get normalization"):
                for key, values in batch.items():
                    data_cache[key].append(copy.deepcopy(values))
        for key, vals in data_cache.items():
            data_cache[key] = np.concatenate(vals)

        return {key: array_to_stats(values) for key, values in data_cache.items()}

    def get_statistics(self) -> dict:
        cached_stats = pathlib.Path(self.data_dir) / ".statistics_cache" / self._stats_id

        try:
            with cached_stats.open() as f:
                stats = json.load(f)
                return dict_apply(stats, partial(np.array, dtype=np.float32))
        except FileNotFoundError:
            stats = self._compute_statistics()

        raw_stats = dict_apply(stats, methodcaller("tolist"))
        cached_stats.parent.mkdir(parents=True, exist_ok=True)
        with cached_stats.open("w") as f:
            json.dump(raw_stats, f)

        return stats

    def __len__(self):
        return len(self._chunk_reader)

    def __getitem__(self, idx: int) -> dict:  # ty:ignore[invalid-method-override]
        if not self._threadpool_limits_is_applied:
            threadpoolctl.threadpool_limits(1)
            self._threadpool_limits_is_applied = True

        data = self._chunk_reader.read_chunk(idx)

        return apply_data_transforms(
            data,
            self._data_transforms,
            should_ignore_transforms_for_norm=self._should_ignore_transforms_for_norm,
        )

    @staticmethod
    def get_val_mask(n_episodes, n_val_episodes, seed=0) -> np.ndarray:
        val_mask = np.zeros(n_episodes, dtype=bool)
        if n_val_episodes <= 0:
            return val_mask

        rng = np.random.default_rng(seed=seed)
        val_idxs = rng.choice(n_episodes, size=n_val_episodes, replace=False)
        val_mask[val_idxs] = True
        return val_mask

    @staticmethod
    def get_val_mask_by_hash(episode_paths, val_ratio: float, denom: int = 10_000) -> np.ndarray:
        # Hash the *stem* (no extension) so the IDM val split and the cosmos
        # video val split agree on which episodes are held out across the
        # `.mp4` / `.zarr` companion files.
        thresh = round(val_ratio * denom)
        return np.array(
            [
                int(hashlib.md5(path.stem.encode("utf-8")).hexdigest(), 16) % denom < thresh
                for path in episode_paths
            ],
            dtype=bool,
        )

    @staticmethod
    def normalize_val_episode_ranges(val_episode_ranges: list[dict] | None) -> dict[str, list[tuple[int, int]]]:
        normalized: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for entry in val_episode_ranges or []:
            if not hasattr(entry, "keys"):
                raise ValueError("Each validation episode range must be a mapping.")
            if set(entry.keys()) != {"episode", "start_s", "end_s"}:
                raise ValueError("Each validation episode range must have exactly: episode, start_s, end_s.")

            episode = int(entry["episode"])
            start_s = float(entry["start_s"])
            end_s = float(entry["end_s"])
            if episode < 0:
                raise ValueError(f"Validation episode id must be non-negative, got {episode}.")
            if start_s < 0 or end_s <= start_s:
                raise ValueError(f"Invalid validation range for episode {episode}: start_s={start_s}, end_s={end_s}.")

            normalized[f"episode_{episode:06d}.zarr"].append((round(start_s * S_TO_NS), round(end_s * S_TO_NS)))

        return dict(normalized)
