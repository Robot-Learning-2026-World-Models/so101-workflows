# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
import pickle
import traceback
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torch.utils.data import Dataset as _Dataset

from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
from imaginaire.utils import log


def _stable_hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


class Dataset(_Dataset):
    def __init__(
        self,
        dataset_dir,
        num_frames,
        video_size,
        is_val: bool,
        include_only_with_substrings: list[str] | None = None,
        exclude_with_substring: str | None = None,
        data_fps: int | None = None,
        effective_fps: float = 10.0,
        obs_history: int = 5,
        is_multi_img: bool = False,
        val_ratio: float = 0.0,
        seed: int | None = None,
        use_precomputed_latents: bool = False,
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()

        # this dataset impl is very sloppy.
        # an index corresponds to a video and then we sample a chunk.
        # this first of all means chunks don't have uniform probability :/
        # we sample by keeping this rng in the dataset. this is also sloppy and will have different behavior
        # depending on persistent_workers (if not kept persistent, we reset the rng state every epoch).
        # but since every epoch every dataset object copy will probably access different videos in a different order
        # thanks to the sampler shuffling, i don't think there is any actual big issue with this.
        # in my opinion dataset getitem should be deterministic and the index should reflect the chunk.
        self._seed = seed
        self._is_val = is_val
        self.rng: np.random.Generator | None = None
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.state_t = 1 + (num_frames - 1) // 4  # latent frames, e.g. 7 for num_frames=25

        latent_dir = os.path.join(dataset_dir, "vae_latents")
        latent_dir_present = os.path.isdir(latent_dir)
        if use_precomputed_latents and latent_dir_present:
            self.latent_dir = latent_dir
            log.info(f"Using precomputed VAE latents from {self.latent_dir}")
        else:
            self.latent_dir = None
            if latent_dir_present and not use_precomputed_latents:
                # Stale cache silently changing semantics is a footgun. Warn loudly.
                log.warning(
                    f"vae_latents directory exists at {latent_dir} but "
                    "use_precomputed_latents=False — ignoring it. Remove the dir or "
                    "set use_precomputed_latents=True to enable it."
                )

        include_only_with_substrings = include_only_with_substrings or []

        video_dir = os.path.join(self.dataset_dir, "video")
        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")

        video_paths = sorted(
            [
                os.path.join(video_dir, f)
                for f in os.listdir(video_dir)
                if f.endswith(".mp4")
                and all(substring in f for substring in include_only_with_substrings)
                and (exclude_with_substring is None or exclude_with_substring not in f)
            ]
        )
        # remove video paths that does not have t5_embedding
        for video_path in video_paths:
            assert os.path.exists(
                os.path.join(
                    self.t5_dir,
                    os.path.basename(video_path).removesuffix(".mp4")[:-1] + ".pickle"
                    if is_multi_img
                    else os.path.basename(video_path).replace(".mp4", ".pickle"),
                )
            ), os.path.basename(video_path)

        log.info(f"{len(video_paths)} videos in total")

        # make val set deterministic even if list of paths changes minorly
        denom = 10_000
        thresh = round(val_ratio * denom)
        if is_val:
            self.video_paths = [p for p in video_paths if _stable_hash_int(os.path.splitext(os.path.basename(p))[0]) % denom < thresh]
        else:
            self.video_paths = [p for p in video_paths if _stable_hash_int(os.path.splitext(os.path.basename(p))[0]) % denom >= thresh]

        log.info(f"{len(self.video_paths)} videos in {'val' if is_val else 'train'}.")

        self.wrong_number = 0
        self.video_size = video_size
        self.obs_history = obs_history
        self.data_fps = data_fps
        self.effective_fps = effective_fps
        self.is_multi_img = is_multi_img

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _ensure_rng(self) -> None:
        if self.rng is not None:
            return
        if self._seed is None:
            self.rng = np.random.default_rng()
            return
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        split_offset = 1 if self._is_val else 0
        self.rng = np.random.default_rng(self._seed + split_offset * 999_983 + worker_id * 100_003)

    def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=0)
        n = len(vr)
        if n == 0:
            raise ValueError(f"Empty video: {video_path}")

        self._ensure_rng()
        fps = self.data_fps or vr.get_avg_fps()
        step = fps / self.effective_fps

        T = self.sequence_length
        # Center the window so the conditioning prefix (obs_history frames) lands at
        # absolute index `i` and the future suffix extends (T - obs_history) frames
        # forward. Restrict `i` to the range where no element of `idx` would clip.
        i_min = int(np.ceil((self.obs_history - 1) * step))
        i_max = int(np.floor((n - 1) - (T - self.obs_history) * step))
        if i_max >= i_min:
            i = int(self.rng.integers(i_min, i_max + 1))  # ty:ignore[no-matching-overload]
        else:
            # Video too short for a clip-free window; fall back to clipped sampling.
            i = max(0, (n - 1) // 2)

        k = np.arange(T, dtype=np.float64) - (self.obs_history - 1)
        idx = np.rint(i + k * step).astype(np.int64)
        idx = np.clip(idx, 0, n - 1)

        uniq, inverse = np.unique(idx, return_inverse=True)

        frames_np = vr.get_batch(uniq).asnumpy()  # [Tv,H,W,3]

        del vr

        x = torch.from_numpy(frames_np).permute(3, 0, 1, 2).contiguous()  # [C,Tv,H,W]
        C, _Tv, H, W = x.shape
        T = self.sequence_length

        out_h, out_w = self.video_size
        x = F.interpolate(
            x.float().view(-1, 1, H, W),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(C, -1, out_h, out_w)
        x = x.round().clamp_(0, 255).to(torch.uint8)

        j = torch.from_numpy(inverse).to(torch.long)
        x = x.index_select(1, j)  # [C,T,H,W]

        # Conditioning fps token matches the effective sample rate of this clip.
        # For Cosmos Predict2 video2world, constants.py asserts fps == 10.
        return x, int(self.effective_fps)

    def __getitem__(self, index) -> dict | Any:
        try:
            data = dict()
            video_path = self.video_paths[index]
            t5_embedding_path = os.path.join(
                self.t5_dir,
                (
                    os.path.basename(video_path).removesuffix(".mp4")[:-1] + ".pickle"
                    if self.is_multi_img
                    else os.path.basename(video_path)
                ).replace(".mp4", ".pickle"),
            )
            data["video_name"] = {
                "video_path": video_path,
                "t5_embedding_path": t5_embedding_path,
            }

            if self.latent_dir is not None:
                self._ensure_rng()
                latent_path = os.path.join(
                    self.latent_dir, os.path.basename(video_path).replace(".mp4", ".pt")
                )
                full_latents = torch.load(latent_path, weights_only=True)  # [16, T_lat, 60, 80]
                n_lat = full_latents.shape[1]
                max_start = max(0, n_lat - self.state_t)
                lat_start = int(self.rng.integers(0, max_start + 1))
                data["video_latent"] = full_latents[:, lat_start : lat_start + self.state_t].float()
                h, w = 480, 640
            else:
                video, fps = self._get_frames(video_path)
                data["video"] = video
                _, _, h, w = video.shape

            # Just add these to fit the interface
            with open(t5_embedding_path, "rb") as f:
                t5_embedding_raw = pickle.load(f)
                assert isinstance(t5_embedding_raw, list)
                assert len(t5_embedding_raw) == 1
                t5_embedding = t5_embedding_raw[0]  # [n_tokens, CosmosTextEncoderConfig.EMBED_DIM]
                assert isinstance(t5_embedding, np.ndarray)
                assert len(t5_embedding.shape) == 2
            n_tokens = t5_embedding.shape[0]
            if n_tokens < CosmosTextEncoderConfig.NUM_TOKENS:
                t5_embedding = np.concatenate(
                    [
                        t5_embedding,
                        np.zeros(
                            (CosmosTextEncoderConfig.NUM_TOKENS - n_tokens, CosmosTextEncoderConfig.EMBED_DIM),
                            dtype=np.float32,
                        ),
                    ],
                    axis=0,
                )
            t5_text_mask = torch.zeros(CosmosTextEncoderConfig.NUM_TOKENS, dtype=torch.int64)
            t5_text_mask[:n_tokens] = 1

            data["obs/language_embedding"] = torch.from_numpy(t5_embedding)
            data["t5_text_mask"] = t5_text_mask
            data["fps"] = int(self.effective_fps)  # match clip's actual sample rate
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except Exception:
            warnings.warn(  # noqa: B028
                f"Invalid data encountered: {self.video_paths[index]}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")  # noqa: B028
            warnings.warn(traceback.format_exc())  # noqa: B028
            self.wrong_number += 1
            log.info(str(self.wrong_number), rank0_only=False)
            return self[np.random.randint(len(self))]


class MultiDataset(_Dataset):
    def __init__(self, **datasets: Dataset) -> None:
        self._datasets = list(datasets.values())
        self._lens = [len(ds) for ds in self._datasets]
        log.info(f"Dataset mix has {len(self)} videos.")

    def __len__(self):
        return sum(self._lens)

    def __getitem__(self, index):
        for i, len_ in enumerate(self._lens):
            if index < len_:
                return self._datasets[i][index]
            index -= len_
