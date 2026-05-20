from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping.")
    return value


def _require_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required in the training/policy_io config.")
    return int(value)


def resolve_action_experiment(name: str) -> str:
    if name.startswith("w2a_"):
        return name
    return f"w2a_{name}_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"


@dataclass(frozen=True)
class PolicyIOConfig:
    image_horizon: int
    image_action_horizon: int
    lowdim_horizon: int
    action_horizon: int
    image_frequency: int | None
    image_action_frequency: int | None
    lowdim_frequency: int | None
    action_frequency: int | None


@dataclass(frozen=True)
class MimicVideoDeploymentConfig:
    prompt: str
    camera_key: str
    crop_xywh: tuple[int, int, int, int]
    image_height: int
    image_width: int
    experiment_name: str
    xattn_layer_idx: int
    action_dim: int
    max_horizon: int
    action_denoising_steps: int
    video_model_path: Path | None
    action_model_path: Path | None
    image_horizon: int
    lowdim_horizon: int
    action_horizon: int
    image_frequency: int
    lowdim_frequency: int
    action_frequency: int
    video_pipe_state_t: int
    network_config: dict[str, Any]
    stats_path: Path


def load_training_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().open() as f:
        data = yaml.safe_load(f)
    return _require_mapping(data, str(path))


def _repo_root_from_training_config(training_config: Path) -> Path:
    config_path = training_config.expanduser().resolve()
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "model/cosmos_predict2/configs/dataloading/policy_io").is_dir():
            return parent
    raise FileNotFoundError(
        f"Could not locate mimic-video repo root from training_config={config_path}; "
        "expected model/cosmos_predict2/configs/dataloading/policy_io under an ancestor."
    )


def load_policy_io(training_config: Path, dataset_name: str) -> PolicyIOConfig:
    policy_io_path = (
        _repo_root_from_training_config(training_config)
        / "model/cosmos_predict2/configs/dataloading/policy_io"
        / f"{dataset_name}.yaml"
    )
    with policy_io_path.open() as f:
        data = yaml.safe_load(f)
    policy_io = _require_mapping(data, str(policy_io_path))["policy_io"]
    obs = _require_mapping(policy_io.get("obs"), f"{policy_io_path}: policy_io.obs")
    action = _require_mapping(policy_io.get("action"), f"{policy_io_path}: policy_io.action")
    image_obs = _require_mapping(obs.get("workspace_rgb"), f"{policy_io_path}: policy_io.obs.workspace_rgb")
    state_obs = _require_mapping(obs.get("joint_state_lowdim"), f"{policy_io_path}: policy_io.obs.joint_state_lowdim")
    image_action = _require_mapping(action.get("workspace_rgb"), f"{policy_io_path}: policy_io.action.workspace_rgb")
    joint_action = _require_mapping(action.get("joint_action_lowdim"), f"{policy_io_path}: policy_io.action.joint_action_lowdim")
    return PolicyIOConfig(
        image_horizon=_require_int(image_obs.get("horizon"), "policy_io.obs.workspace_rgb.horizon"),
        image_action_horizon=_require_int(image_action.get("horizon"), "policy_io.action.workspace_rgb.horizon"),
        lowdim_horizon=_require_int(state_obs.get("horizon"), "policy_io.obs.joint_state_lowdim.horizon"),
        action_horizon=_require_int(joint_action.get("horizon"), "policy_io.action.joint_action_lowdim.horizon"),
        image_frequency=None if image_obs.get("target_frequency") is None else int(image_obs.get("target_frequency")),
        image_action_frequency=None if image_action.get("target_frequency") is None else int(image_action.get("target_frequency")),
        lowdim_frequency=None if state_obs.get("target_frequency") is None else int(state_obs.get("target_frequency")),
        action_frequency=None if joint_action.get("target_frequency") is None else int(joint_action.get("target_frequency")),
    )


def resolve_stats_path(*, explicit_stats_path: Path | None, action_data_dir: Path | None) -> Path:
    if explicit_stats_path is not None:
        return explicit_stats_path.expanduser().resolve()

    if action_data_dir is None:
        raise ValueError("Either stats_path or action_data_dir is required.")
    stats_dir = action_data_dir.expanduser().resolve() / ".statistics_cache"
    candidates = sorted(p for p in stats_dir.glob("*") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No statistics cache files found under {stats_dir}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(f"Multiple statistics cache files found under {stats_dir}; pass stats_path explicitly: {names}")
    return candidates[0]


def maybe_download_stats(*, stats_path: Path, hf_repo_id: str | None, hf_filename: str | None, hf_revision: str | None) -> Path:
    if stats_path.exists():
        return stats_path
    if not hf_repo_id or not hf_filename:
        raise FileNotFoundError(f"stats_path does not exist and no HF stats source was provided: {stats_path}")

    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_filename,
            repo_type=os.environ.get("HF_STATS_REPO_TYPE", "dataset"),
            revision=hf_revision or "main",
        )
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_bytes(downloaded.read_bytes())
    return stats_path


def maybe_download_file(
    *,
    path: Path,
    hf_repo_id: str | None,
    hf_filename: str | None,
    hf_revision: str | None,
    repo_type: str | None = "model",
) -> Path:
    if path.exists():
        return path
    if not hf_repo_id or not hf_filename:
        raise FileNotFoundError(f"path does not exist and no HF source was provided: {path}")

    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_filename,
            repo_type=repo_type,
            revision=hf_revision or "main",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(downloaded.read_bytes())
    return path


def _derive_video_pipe_state_t(policy_io: PolicyIOConfig) -> int:
    video_pixel_horizon = policy_io.image_horizon + policy_io.image_action_horizon
    if (video_pixel_horizon - 1) % 4 != 0:
        raise ValueError(
            "policy_io workspace_rgb obs/action horizons must form a Cosmos pixel horizon "
            f"of 4n+1 frames, got {video_pixel_horizon}."
        )
    return 1 + (video_pixel_horizon - 1) // 4


def _require_frequency(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} must be set for SO101 deployment.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return int(value)


def _optional_checkpoint_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = str(value).strip()
    if not path:
        return None
    return Path(path).expanduser()


def build_deployment_config(
    *,
    training_config: Path | None,
    stats_path: Path | None,
    action_data_dir: Path | None,
    hf_stats_repo_id: str | None,
    hf_stats_filename: str | None,
    hf_stats_revision: str | None,
) -> MimicVideoDeploymentConfig:
    if training_config is None:
        raise ValueError("training_config is required; pass the SO101 YAML used for training.")

    data = load_training_yaml(training_config)
    preprocessing = _require_mapping(data.get("preprocessing"), "preprocessing")
    training = _require_mapping(data.get("training"), "training")
    action_cfg = training.get("action_decoder") or {}
    if not isinstance(action_cfg, dict):
        raise ValueError("training.action_decoder must be a mapping when present.")
    model_cfg = action_cfg.get("model") or {}
    network_cfg = action_cfg.get("network") or {}
    scheduler_cfg = action_cfg.get("action_scheduler") or {}
    image_cfg = _require_mapping(preprocessing.get("image"), "preprocessing.image")

    crop_xywh = image_cfg.get("crop_xywh")
    if not isinstance(crop_xywh, list) or len(crop_xywh) != 4:
        raise ValueError("preprocessing.image.crop_xywh must be [x, y, width, height].")
    output_size_hw = image_cfg.get("output_size_hw")
    if not isinstance(output_size_hw, list) or len(output_size_hw) != 2:
        raise ValueError("preprocessing.image.output_size_hw must be [height, width].")

    dataset_name = str(action_cfg.get("experiment") or "so101")
    policy_io = load_policy_io(training_config, dataset_name)
    image_frequency = _require_frequency(policy_io.image_frequency, "policy_io.obs.workspace_rgb.target_frequency")
    image_action_frequency = _require_frequency(
        policy_io.image_action_frequency,
        "policy_io.action.workspace_rgb.target_frequency",
    )
    if image_action_frequency != image_frequency:
        raise ValueError(
            "SO101 deployment expects obs/action video target frequencies to match, got "
            f"{image_frequency} and {image_action_frequency}."
        )
    lowdim_frequency = _require_frequency(policy_io.lowdim_frequency, "policy_io.obs.joint_state_lowdim.target_frequency")
    action_frequency = _require_frequency(policy_io.action_frequency, "policy_io.action.joint_action_lowdim.target_frequency")
    if lowdim_frequency != action_frequency:
        raise ValueError(
            "SO101 deployment expects lowdim obs/action target frequencies to match, got "
            f"{lowdim_frequency} and {action_frequency}."
        )
    derived_video_pipe_state_t = _derive_video_pipe_state_t(policy_io)
    video_pipe_state_t = int(model_cfg.get("video_pipe_state_t", derived_video_pipe_state_t))
    if video_pipe_state_t != derived_video_pipe_state_t:
        raise ValueError(
            "training.action_decoder.model.video_pipe_state_t does not match policy_io workspace_rgb horizons: "
            f"{video_pipe_state_t} vs derived {derived_video_pipe_state_t}."
        )

    resolved_stats_path = resolve_stats_path(explicit_stats_path=stats_path, action_data_dir=action_data_dir)
    resolved_stats_path = maybe_download_stats(
        stats_path=resolved_stats_path,
        hf_repo_id=hf_stats_repo_id,
        hf_filename=hf_stats_filename,
        hf_revision=hf_stats_revision,
    )

    sources = preprocessing.get("sources") or []
    first_source = sources[0] if isinstance(sources, list) and sources and isinstance(sources[0], dict) else {}
    action_dim = int(network_cfg.get("out_channels", network_cfg.get("in_channels", 6)))
    return MimicVideoDeploymentConfig(
        prompt=str(preprocessing.get("instruction") or first_source.get("instruction") or ""),
        camera_key=str(preprocessing.get("video_feature") or first_source.get("video_feature") or "observation.images.front"),
        crop_xywh=tuple(int(x) for x in crop_xywh),
        image_height=int(output_size_hw[0]),
        image_width=int(output_size_hw[1]),
        experiment_name=resolve_action_experiment(dataset_name),
        xattn_layer_idx=int(model_cfg.get("xattn_layer_idx", 20)),
        action_dim=action_dim,
        max_horizon=int(network_cfg.get("max_horizon", policy_io.action_horizon + 1)),
        action_denoising_steps=int(scheduler_cfg.get("num_denoising_steps", 10)),
        video_model_path=_optional_checkpoint_path(model_cfg.get("video_dit_path")),
        action_model_path=_optional_checkpoint_path(model_cfg.get("action_dit_path") or action_cfg.get("checkpoint_path")),
        image_horizon=policy_io.image_horizon,
        lowdim_horizon=policy_io.lowdim_horizon,
        action_horizon=policy_io.action_horizon,
        image_frequency=image_frequency,
        lowdim_frequency=lowdim_frequency,
        action_frequency=action_frequency,
        video_pipe_state_t=video_pipe_state_t,
        network_config=dict(network_cfg),
        stats_path=resolved_stats_path,
    )


def load_stats(path: Path) -> dict[str, Any]:
    with path.open() as f:
        stats = json.load(f)
    required = ("action/joint_action_lowdim", "obs/joint_state_lowdim")
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError(f"Statistics file {path} is missing required keys: {missing}")
    return stats
