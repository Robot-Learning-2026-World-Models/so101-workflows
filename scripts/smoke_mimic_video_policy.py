from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot_policy_mimic_video.configuration_mimic_video import MimicVideoPolicyConfig
from lerobot_policy_mimic_video.modeling_mimic_video import MimicVideoPolicy


def _load_image(path: Path | None, width: int, height: int) -> np.ndarray:
    if path is None:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = 32
        image[..., 1] = 32
        image[..., 2] = 32
        return image

    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mimic-video policy inference and discard the action.")
    parser.add_argument("--video-model-path", type=Path, required=True)
    parser.add_argument("--action-model-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--prompt-embedding-path", type=Path)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--prompt", default="the robot arm gently nudges the white object towards the target circle")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-sampling-steps", type=int)
    parser.add_argument("--stop-video-denoising-step", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MimicVideoPolicyConfig(
        video_model_path=args.video_model_path,
        action_model_path=args.action_model_path,
        stats_path=args.stats_path,
        training_config=args.training_config,
        prompt_embedding_path=args.prompt_embedding_path,
        prompt=args.prompt,
        device=args.device,
        seed=args.seed,
        use_cuda_graphs=args.use_cuda_graphs,
    )
    if args.num_sampling_steps is not None:
        config.num_sampling_steps = args.num_sampling_steps
    if args.stop_video_denoising_step is not None:
        config.stop_video_denoising_step = args.stop_video_denoising_step

    policy = MimicVideoPolicy(config).eval()
    image = _load_image(args.image_path, width=1280, height=720)
    state = np.zeros((max(1, policy.config.lowdim_horizon), policy.config.action_dim), dtype=np.float32)
    batch = {
        policy.config.camera_key: torch.from_numpy(image),
        policy.config.state_key: torch.from_numpy(state),
    }

    with torch.inference_mode():
        action = policy.select_action(batch)

    print("mimic-video smoke pass complete")
    print(f"action shape: {tuple(action.shape)}")
    print(f"action dtype: {action.dtype}")
    print(f"action: {action.detach().cpu().tolist()}")
    print("action discarded; no robot was connected")


if __name__ == "__main__":
    main()
