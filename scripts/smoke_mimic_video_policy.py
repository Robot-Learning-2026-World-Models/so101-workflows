from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot_policy_mimic_video.configuration_mimic_video import MimicVideoPolicyConfig
from lerobot_policy_mimic_video.modeling_mimic_video import MimicVideoPolicy


_VIDEO_SUFFIXES = {".avi", ".m4v", ".mov", ".mp4", ".webm"}
_SO101_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _load_image(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _load_video(path: Path, width: int, height: int, max_frames: int, start_frame: int | None) -> np.ndarray:
    import imageio.v2 as imageio

    frames: deque[np.ndarray] = deque(maxlen=max_frames)
    with imageio.get_reader(path) as reader:
        for frame_index, frame in enumerate(reader):
            if start_frame is not None and frame_index < start_frame:
                continue
            image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            frames.append(np.asarray(image, dtype=np.uint8))
            if start_frame is not None and len(frames) == max_frames:
                break

    if not frames:
        raise ValueError(f"No frames found in seed video: {path}")
    return np.stack(list(frames), axis=0)


def _load_visual_context(
    path: Path | None,
    width: int,
    height: int,
    max_video_frames: int,
    video_start_frame: int | None,
) -> np.ndarray:
    if path is None:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = 32
        image[..., 1] = 32
        image[..., 2] = 32
        return image

    if path.suffix.lower() in _VIDEO_SUFFIXES:
        return _load_video(path, width=width, height=height, max_frames=max_video_frames, start_frame=video_start_frame)
    return _load_image(path, width=width, height=height)


def _load_state(path: Path | None, *, action_dim: int, lowdim_horizon: int) -> np.ndarray:
    if path is None:
        return np.zeros((max(1, lowdim_horizon), action_dim), dtype=np.float32)

    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload.get("state", payload)
    if isinstance(state, dict):
        state = [state[name] for name in _SO101_JOINT_NAMES]
    arr = np.asarray(state, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != action_dim:
        raise ValueError(f"Expected state shape ({action_dim},), got {arr.shape} from {path}.")

    joint_names = payload.get("joint_names") if isinstance(payload, dict) else None
    if joint_names is not None and list(joint_names) != _SO101_JOINT_NAMES:
        raise ValueError(f"Unexpected joint order in {path}: {joint_names}")

    return np.repeat(arr[None, :], max(1, lowdim_horizon), axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mimic-video policy inference and discard the action.")
    parser.add_argument("--video-model-path", type=Path, required=True)
    parser.add_argument("--action-model-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--prompt-embedding-path", type=Path)
    parser.add_argument("--image-path", type=Path, help="Optional seed image or video path.")
    parser.add_argument("--video-start-frame", type=int, help="Zero-based frame index for the seed video window.")
    parser.add_argument("--state-path", type=Path, help="Optional JSON state seed in SO101 joint degrees.")
    parser.add_argument("--prompt", default="the robot arm gently nudges the white object towards the target circle")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-sampling-steps", type=int)
    parser.add_argument("--stop-video-denoising-step", type=int)
    parser.add_argument("--denoised-video-save-dir", type=Path)
    parser.add_argument("--denoised-video-save-fps", type=int, default=10)
    parser.add_argument("--denoised-video-save-prefix", default="smoke")
    parser.add_argument("--trajectory-save-dir", type=Path)
    parser.add_argument("--trajectory-save-prefix", default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _save_trajectory(
    action_trajectory: torch.Tensor,
    *,
    path: Path,
    policy: MimicVideoPolicy,
    args: argparse.Namespace,
) -> None:
    action_cpu = action_trajectory.detach().cpu()
    payload = {
        "format": "so101_mimic_video_action_trajectory",
        "version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "joint_names": _SO101_JOINT_NAMES,
        "action_frequency": policy.config.action_frequency,
        "trajectory": action_cpu.tolist(),
        "shape": list(action_cpu.shape),
        "dtype": str(action_cpu.dtype),
        "prompt": args.prompt,
        "seed": args.seed,
        "image_path": None if args.image_path is None else str(args.image_path),
        "video_start_frame": args.video_start_frame,
        "state_path": None if args.state_path is None else str(args.state_path),
        "denoised_video_path": None
        if policy.last_denoised_video_path is None
        else str(policy.last_denoised_video_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
        denoised_video_save_dir=args.denoised_video_save_dir,
        denoised_video_save_fps=args.denoised_video_save_fps,
        denoised_video_save_prefix=args.denoised_video_save_prefix,
    )
    if args.num_sampling_steps is not None:
        config.num_sampling_steps = args.num_sampling_steps
    if args.stop_video_denoising_step is not None:
        config.stop_video_denoising_step = args.stop_video_denoising_step
    if args.video_start_frame is not None and args.video_start_frame < 0:
        raise ValueError("--video-start-frame must be non-negative.")

    policy = MimicVideoPolicy(config).eval()
    step = policy.config.input_frequency // policy.config.image_frequency
    max_video_frames = 1 + (policy.config.image_horizon - 1) * step
    image = _load_visual_context(
        args.image_path,
        width=1280,
        height=720,
        max_video_frames=max_video_frames,
        video_start_frame=args.video_start_frame,
    )
    state = _load_state(
        args.state_path,
        action_dim=policy.config.action_dim,
        lowdim_horizon=policy.config.lowdim_horizon,
    )
    batch = {
        policy.config.camera_key: torch.from_numpy(image),
        policy.config.state_key: torch.from_numpy(state),
    }

    with torch.inference_mode():
        action_trajectory = policy.predict_action_chunk(batch)[0]

    print("mimic-video smoke pass complete")
    print(f"action trajectory shape: {tuple(action_trajectory.shape)}")
    print(f"action trajectory dtype: {action_trajectory.dtype}")
    print(f"state seed: {state[0].tolist()}")
    print(f"action trajectory: {action_trajectory.detach().cpu().tolist()}")
    print(f"first action: {action_trajectory[0].detach().cpu().tolist()}")
    if policy.last_denoised_video_path is not None:
        print(f"video saved: {policy.last_denoised_video_path}")
    elif policy.config.denoised_video_save_dir is not None:
        print(f"video save requested under: {policy.config.denoised_video_save_dir.expanduser()}")
    if args.trajectory_save_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_path = args.trajectory_save_dir.expanduser() / f"{args.trajectory_save_prefix}_{timestamp}.json"
        _save_trajectory(action_trajectory, path=trajectory_path, policy=policy, args=args)
        print(f"trajectory saved: {trajectory_path}")
    print("action discarded; no robot was connected")


if __name__ == "__main__":
    main()
