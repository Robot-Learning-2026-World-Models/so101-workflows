from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


SO101_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a saved SO101 action trajectory to a tiny LeRobot dataset.")
    parser.add_argument("trajectory_path", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/lerobot_trajectory_datasets"))
    parser.add_argument("--repo-id", help="LeRobot repo_id to store in metadata. Defaults to local/<trajectory-stem>.")
    parser.add_argument("--fps", type=int, help="Override the trajectory action frequency.")
    parser.add_argument("--task", help="Task text stored in the one-episode dataset.")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the generated LeRobot dataset to Hugging Face.")
    parser.add_argument("--private", action="store_true", help="Create the Hugging Face dataset repo as private.")
    parser.add_argument("--branch", help="Optional Hugging Face branch to push to.")
    parser.add_argument("--license", default="apache-2.0", help="Dataset card license used when pushing.")
    parser.add_argument("--no-tag-version", action="store_true", help="Do not tag the repo with the LeRobot codebase version.")
    parser.add_argument("--include-video", action="store_true", help="Include a video stream for web visualizers.")
    parser.add_argument("--video-source", type=Path, help="Optional mp4/image source for the video stream.")
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=180)
    return parser.parse_args()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return slug or "trajectory"


def _next_dataset_root(output_root: Path, stem: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / stem
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = output_root / f"{stem}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused dataset directory under {output_root}")


def _load_trajectory(path: Path) -> tuple[np.ndarray, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = np.asarray(payload["trajectory"], dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 6:
        raise ValueError(f"Expected trajectory shape (T, 6), got {trajectory.shape}.")
    return trajectory, payload


def _load_video_frames(path: Path, *, width: int, height: int) -> list[np.ndarray]:
    import imageio.v2 as imageio

    frames = []
    with imageio.get_reader(path) as reader:
        for frame in reader:
            image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            frames.append(np.asarray(image, dtype=np.uint8))
    return frames


def _load_image_frame(path: Path, *, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _make_video_frames(
    *,
    payload: dict,
    source: Path | None,
    length: int,
    width: int,
    height: int,
) -> list[np.ndarray]:
    if source is None and payload.get("denoised_video_path"):
        source = Path(payload["denoised_video_path"]).expanduser()

    if source is not None and source.exists():
        if source.suffix.lower() in {".avi", ".m4v", ".mov", ".mp4", ".webm"}:
            source_frames = _load_video_frames(source, width=width, height=height)
        else:
            source_frames = [_load_image_frame(source, width=width, height=height)]
    else:
        source_frames = []

    if not source_frames:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = 24
        frame[..., 1] = 24
        frame[..., 2] = 24
        source_frames = [frame]

    if len(source_frames) == length:
        return source_frames

    indices = np.linspace(0, len(source_frames) - 1, num=length)
    return [source_frames[int(round(index))] for index in indices]


def main() -> None:
    args = parse_args()
    trajectory_path = args.trajectory_path.expanduser()
    trajectory, payload = _load_trajectory(trajectory_path)

    fps = args.fps or int(payload.get("action_frequency") or 30)
    if fps <= 0:
        raise ValueError("--fps must be positive.")

    joint_names = payload.get("joint_names") or SO101_JOINT_NAMES
    if joint_names != SO101_JOINT_NAMES:
        raise ValueError(f"Expected SO101 joint names {SO101_JOINT_NAMES}, got {joint_names}.")

    stem = _slug(trajectory_path.stem)
    repo_id = args.repo_id or f"local/{stem}"
    dataset_root = _next_dataset_root(args.output_root.expanduser().resolve(), stem)
    task = args.task or payload.get("prompt") or "mimic-video SO101 smoke trajectory"

    features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": SO101_JOINT_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": SO101_JOINT_NAMES,
        },
    }
    video_frames = None
    if args.include_video:
        features["observation.images.front"] = {
            "dtype": "video",
            "shape": (args.video_height, args.video_width, 3),
            "names": ["height", "width", "channels"],
        }
        video_frames = _make_video_frames(
            payload=payload,
            source=args.video_source.expanduser() if args.video_source is not None else None,
            length=len(trajectory),
            width=args.video_width,
            height=args.video_height,
        )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=dataset_root,
        robot_type="so101_follower",
        use_videos=args.include_video,
        vcodec="h264" if args.include_video else "libsvtav1",
    )

    zero_state = np.zeros(6, dtype=np.float32)
    for index, action in enumerate(trajectory):
        frame = {
            "action": action.astype(np.float32, copy=False),
            "observation.state": zero_state,
            "task": task,
        }
        if video_frames is not None:
            frame["observation.images.front"] = video_frames[index]
        dataset.add_frame(frame)
    dataset.save_episode()
    dataset.finalize()

    print(f"dataset root: {dataset_root}")
    print(f"repo id: {repo_id}")
    print(f"episode index: 0")
    print(
        "visualize with: "
        f".venv/bin/lerobot-dataset-viz --repo-id {repo_id} --root {dataset_root} --episode-index 0"
    )
    if args.push_to_hub:
        print(f"pushing to Hugging Face: {repo_id}")
        dataset.push_to_hub(
            branch=args.branch,
            private=args.private,
            license=args.license,
            tag_version=not args.no_tag_version,
            push_videos=args.include_video,
        )
        revision_suffix = "" if args.branch is None else f"@{args.branch}"
        print(f"pushed: {repo_id}{revision_suffix}")


if __name__ == "__main__":
    main()
