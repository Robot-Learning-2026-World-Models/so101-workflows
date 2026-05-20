from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    from lerobot.configs import PreTrainedConfig
except ImportError:
    from lerobot.configs.policies import PreTrainedConfig

try:
    from lerobot.optim import AdamWConfig
except ImportError:
    from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("mimic_video")
@dataclass
class MimicVideoPolicyConfig(PreTrainedConfig):
    """LeRobot config for loading mimic-video directly in PolicyServer."""

    video_model_path: Path | None = None
    action_model_path: Path | None = None
    stats_path: Path | None = None
    action_data_dir: Path | None = None
    hf_stats_repo_id: str | None = None
    hf_stats_filename: str | None = None
    hf_stats_revision: str | None = None
    training_config: Path | None = None
    prompt_embedding_path: Path | None = None
    experiment_name: str = "w2a_so101_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"
    prompt: str = "the robot arm pushes the white object towards the goal circle"
    camera_key: str = "observation.images.front"
    state_key: str = "observation.state"
    action_dim: int = 6
    horizon: int = 60
    n_action_steps: int = 3
    image_width: int = 640
    image_height: int = 480
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 640
    crop_height: int = 480
    image_horizon: int = 5
    lowdim_horizon: int = 1
    image_frequency: int = 10
    lowdim_frequency: int = 30
    action_frequency: int = 30
    input_frequency: int = 30
    num_sampling_steps: int = 35
    stop_video_denoising_step: int = 0
    denoised_video_save_dir: Path | None = None
    denoised_video_save_every_n: int = 1
    denoised_video_save_fps: int = 10
    denoised_video_save_prefix: str = "denoised"
    max_horizon: int = 61
    video_pipe_state_t: int = 11
    xattn_layer_idx: int = 20
    action_denoising_steps: int = 10
    use_cuda_graphs: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps cannot exceed horizon.")
        if self.video_model_path is None:
            raise ValueError("video_model_path is required.")
        if self.action_model_path is None:
            raise ValueError("action_model_path is required.")
        if self.training_config is None:
            raise ValueError("training_config is required; pass the SO101 YAML used for training.")
        if self.stats_path is None and self.action_data_dir is None:
            raise ValueError("Either stats_path or action_data_dir is required.")
        if self.input_frequency <= 0:
            raise ValueError("input_frequency must be positive.")
        if self.image_frequency <= 0:
            raise ValueError("image_frequency must be positive.")
        if self.action_frequency <= 0:
            raise ValueError("action_frequency must be positive.")
        if self.input_frequency % self.image_frequency != 0:
            raise ValueError(
                "input_frequency must be an integer multiple of image_frequency "
                f"for stateless visual downsampling, got {self.input_frequency} and {self.image_frequency}."
            )
        if self.action_frequency % self.input_frequency != 0:
            raise ValueError(
                "action_frequency must be an integer multiple of input_frequency "
                f"for action chunk execution, got {self.action_frequency} and {self.input_frequency}."
            )
        if self.num_sampling_steps <= 0:
            raise ValueError("num_sampling_steps must be positive.")
        if self.stop_video_denoising_step < 0:
            raise ValueError("stop_video_denoising_step must be non-negative.")
        if self.stop_video_denoising_step > self.num_sampling_steps:
            raise ValueError(
                "stop_video_denoising_step cannot exceed num_sampling_steps, got "
                f"{self.stop_video_denoising_step} and {self.num_sampling_steps}."
            )
        if self.denoised_video_save_every_n <= 0:
            raise ValueError("denoised_video_save_every_n must be positive.")
        if self.denoised_video_save_fps <= 0:
            raise ValueError("denoised_video_save_fps must be positive.")

    def validate_features(self) -> None:
        if self.action_feature is not None and self.action_feature.shape[0] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim}, got LeRobot action feature {self.action_feature.shape}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=0.0, weight_decay=0.0)

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
