import logging
import pickle  # nosec
import time
import traceback
from concurrent import futures
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import grpc
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2
from lerobot.transport import services_pb2_grpc

from .configuration_mimic_video import MimicVideoPolicyConfig
from .modeling_mimic_video import MimicVideoPolicy
from .training_config import build_deployment_config, maybe_download_file

try:
    from lerobot.async_inference.helpers import make_lerobot_observation as _raw_to_lerobot_observation
except ImportError:
    from lerobot.async_inference.helpers import raw_observation_to_observation as _raw_to_lerobot_observation


FRAME_WINDOW_SUFFIX = "_frame_window"
RAW_STATE_KEY = "observation.state"


@dataclass
class MimicVideoPolicyServerConfig(PolicyServerConfig):
    video_model_path: Path | None = None
    action_model_path: Path | None = None
    video_model_hf_repo_id: str | None = None
    video_model_hf_filename: str | None = None
    video_model_hf_revision: str | None = None
    video_model_hf_repo_type: str = "model"
    action_model_hf_repo_id: str | None = None
    action_model_hf_filename: str | None = None
    action_model_hf_revision: str | None = None
    action_model_hf_repo_type: str = "model"
    checkpoint_cache_dir: Path = Path("/checkpoints")
    training_config: Path | None = None
    stats_path: Path | None = None
    action_data_dir: Path | None = None
    hf_stats_repo_id: str | None = None
    hf_stats_filename: str | None = None
    hf_stats_revision: str | None = None
    policy_device: str = "cuda"
    prompt: str | None = None
    task: str | None = None
    num_sampling_steps: int = 35
    stop_video_denoising_step: int = 0
    denoised_video_save_dir: Path | None = None
    denoised_video_save_every_n: int = 1
    denoised_video_save_fps: int = 10
    denoised_video_save_prefix: str = "denoised"
    use_cuda_graphs: bool = True
    seed: int = 0


def _raw_camera_key(policy_camera_key: str) -> str:
    prefix = "observation.images."
    if policy_camera_key.startswith(prefix):
        return policy_camera_key[len(prefix) :]
    return policy_camera_key.rsplit(".", 1)[-1]


def _select_prompt(default_prompt: str, *, task: str | None, prompt: str | None) -> str:
    if prompt:
        return prompt
    task_prompts = {
        "task1": "the robot arm gently nudges the white object towards the target circle",
        "task2": "the robot arm gently nudges the white object towards the target circle while avoiding the red cylinder",
    }
    if task:
        key = task.strip().lower().replace("-", "").replace("_", "")
        if key in task_prompts:
            return task_prompts[key]
        return task
    return default_prompt


def _resolve_model_path(
    *,
    explicit_path: Path | None,
    config_path: Path | None,
    cache_dir: Path,
    hf_repo_id: str | None,
    hf_filename: str | None,
    hf_revision: str | None,
    hf_repo_type: str,
    label: str,
) -> Path:
    path = explicit_path or config_path
    if path is None and hf_filename:
        path = cache_dir / hf_filename
    if path is None:
        raise ValueError(
            f"{label} checkpoint is required. Set {label.upper()}_MODEL_PATH, "
            f"put it in the training YAML, or provide {label.upper()}_MODEL_HF_REPO_ID/FILENAME."
        )
    return maybe_download_file(
        path=path.expanduser(),
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        hf_revision=hf_revision,
        repo_type=hf_repo_type,
    ).resolve()


class MimicVideoPolicyServer(PolicyServer):
    """LeRobot async gRPC server that preserves mimic-video's frame-window input.

    The stock LeRobot server converts each raw camera frame to one resized
    `C,H,W` tensor before calling the policy. mimic-video needs the raw temporal
    frame window so it can apply the YAML crop/resize and downsample 30Hz -> 10Hz
    itself.
    """

    config: MimicVideoPolicyServerConfig

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()
        policy_specs = pickle.loads(request.data)  # nosec
        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")
        if policy_specs.policy_type != "mimic_video":
            raise ValueError(f"MimicVideoPolicyServer only supports policy_type='mimic_video', got {policy_specs.policy_type!r}.")

        self.device = self.config.policy_device or policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        deployment = build_deployment_config(
            training_config=self.config.training_config,
            stats_path=self.config.stats_path,
            action_data_dir=self.config.action_data_dir,
            hf_stats_repo_id=self.config.hf_stats_repo_id,
            hf_stats_filename=self.config.hf_stats_filename,
            hf_stats_revision=self.config.hf_stats_revision,
        )
        video_model_path = _resolve_model_path(
            explicit_path=self.config.video_model_path,
            config_path=deployment.video_model_path,
            cache_dir=self.config.checkpoint_cache_dir,
            hf_repo_id=self.config.video_model_hf_repo_id,
            hf_filename=self.config.video_model_hf_filename,
            hf_revision=self.config.video_model_hf_revision,
            hf_repo_type=self.config.video_model_hf_repo_type,
            label="video",
        )
        action_model_path = _resolve_model_path(
            explicit_path=self.config.action_model_path,
            config_path=deployment.action_model_path,
            cache_dir=self.config.checkpoint_cache_dir,
            hf_repo_id=self.config.action_model_hf_repo_id,
            hf_filename=self.config.action_model_hf_filename,
            hf_revision=self.config.action_model_hf_revision,
            hf_repo_type=self.config.action_model_hf_repo_type,
            label="action",
        )
        policy_config = MimicVideoPolicyConfig(
            video_model_path=video_model_path,
            action_model_path=action_model_path,
            stats_path=deployment.stats_path,
            action_data_dir=None,
            hf_stats_repo_id=self.config.hf_stats_repo_id,
            hf_stats_filename=self.config.hf_stats_filename,
            hf_stats_revision=self.config.hf_stats_revision,
            training_config=self.config.training_config,
            n_action_steps=policy_specs.actions_per_chunk,
            input_frequency=self.config.fps,
            num_sampling_steps=self.config.num_sampling_steps,
            stop_video_denoising_step=self.config.stop_video_denoising_step or self.config.num_sampling_steps,
            denoised_video_save_dir=self.config.denoised_video_save_dir,
            denoised_video_save_every_n=self.config.denoised_video_save_every_n,
            denoised_video_save_fps=self.config.denoised_video_save_fps,
            denoised_video_save_prefix=self.config.denoised_video_save_prefix,
            use_cuda_graphs=self.config.use_cuda_graphs,
            seed=self.config.seed,
        )
        policy_config.prompt = _select_prompt(
            policy_config.prompt,
            task=self.config.task,
            prompt=self.config.prompt,
        )
        start = time.perf_counter()
        self.policy = MimicVideoPolicy(policy_config).to(self.device).eval()
        self.preprocessor = None
        self.postprocessor = None
        self.logger.info(
            f"Loaded mimic_video on {self.device} in {time.perf_counter() - start:.4f}s | "
            f"actions_per_chunk={self.actions_per_chunk}"
        )
        return services_pb2.Empty()

    def _get_action_chunk(
        self,
        observation: dict[str, torch.Tensor],
        *,
        observation_timestep: int | None = None,
    ) -> torch.Tensor:
        chunk = self.policy.predict_action_chunk(observation, observation_timestep=observation_timestep)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        start_prepare = time.perf_counter()
        raw_observation: dict[str, Any] = observation_t.get_observation()
        if RAW_STATE_KEY in raw_observation:
            observation = {RAW_STATE_KEY: torch.as_tensor(raw_observation[RAW_STATE_KEY])}
        else:
            observation = _raw_to_lerobot_observation(raw_observation, self.lerobot_features)

        camera_key = self.policy.config.camera_key
        window_key = f"{_raw_camera_key(camera_key)}{FRAME_WINDOW_SUFFIX}"
        if window_key in raw_observation:
            observation[camera_key] = torch.as_tensor(raw_observation[window_key])
        elif camera_key in observation:
            observation[camera_key] = torch.as_tensor(observation[camera_key])
        else:
            raise KeyError(
                f"Missing camera window {window_key!r} and policy camera key {camera_key!r}; "
                f"raw keys={sorted(raw_observation)}, observation keys={sorted(observation)}"
            )

        if "task" in raw_observation:
            observation["task"] = raw_observation["task"]
        self.last_processed_obs = observation_t
        prepare_time = time.perf_counter() - start_prepare

        start_inference = time.perf_counter()
        try:
            action_tensor = self._get_action_chunk(observation, observation_timestep=observation_t.get_timestep())
        except Exception:
            self.logger.error("Mimic-video inference failed:\n%s", traceback.format_exc())
            raise
        inference_time = time.perf_counter() - start_inference
        if action_tensor.ndim != 3:
            raise RuntimeError(f"Expected action chunk shape (B, H, D), got {tuple(action_tensor.shape)}")

        action_tensor = action_tensor.squeeze(0).detach().cpu()
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor),
            observation_t.get_timestep(),
        )
        total_time = time.perf_counter() - start_prepare
        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare: {1000 * prepare_time:.2f}ms | "
            f"Inference: {1000 * inference_time:.2f}ms | "
            f"Total: {1000 * total_time:.2f}ms | "
            f"Action shape: {tuple(action_tensor.shape)}"
        )
        return action_chunk


@draccus.wrap()
def serve(cfg: MimicVideoPolicyServerConfig):
    logging.info(pformat(asdict(cfg)))
    policy_server = MimicVideoPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info(f"MimicVideoPolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
