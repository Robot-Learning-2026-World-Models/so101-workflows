from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from lerobot.policies.pretrained import PreTrainedPolicy

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.types import NormalizationType
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from cosmos_predict2.utils.context_parallel import cat_outputs_cp, split_inputs_cp
from imaginaire.utils import misc
from imaginaire.utils.config_helper import override

from .configuration_mimic_video import MimicVideoPolicyConfig
from .training_config import build_deployment_config, load_stats


LOGGER = logging.getLogger(__name__)


class MimicVideoPolicy(PreTrainedPolicy):
    """LeRobot policy that runs mimic-video directly in the async PolicyServer."""

    config_class = MimicVideoPolicyConfig
    name = "mimic_video"

    def __init__(self, config: MimicVideoPolicyConfig, dataset_stats: dict[str, Any] | None = None):
        self.deployment_config = build_deployment_config(
            training_config=config.training_config,
            stats_path=config.stats_path,
            action_data_dir=config.action_data_dir,
            hf_stats_repo_id=config.hf_stats_repo_id,
            hf_stats_filename=config.hf_stats_filename,
            hf_stats_revision=config.hf_stats_revision,
        )
        self._apply_deployment_config(config)
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.pipeline = self._load_pipeline(config)
        self.prompt_embedding = self._load_prompt_embedding(config)
        self._prediction_index = 0
        self.last_denoised_video_path = None

    def _apply_deployment_config(self, config: MimicVideoPolicyConfig) -> None:
        dc = self.deployment_config
        config.prompt = dc.prompt or config.prompt
        config.camera_key = dc.camera_key
        config.image_height = dc.image_height
        config.image_width = dc.image_width
        config.crop_x, config.crop_y, config.crop_width, config.crop_height = dc.crop_xywh
        config.experiment_name = dc.experiment_name
        config.xattn_layer_idx = dc.xattn_layer_idx
        config.action_dim = dc.action_dim
        config.horizon = dc.action_horizon
        config.image_horizon = dc.image_horizon
        config.lowdim_horizon = dc.lowdim_horizon
        config.image_frequency = dc.image_frequency
        config.lowdim_frequency = dc.lowdim_frequency
        config.action_frequency = dc.action_frequency
        config.max_horizon = dc.max_horizon
        config.video_pipe_state_t = dc.video_pipe_state_t
        config.action_denoising_steps = dc.action_denoising_steps
        config.stats_path = dc.stats_path
        config._mimic_video_network_config = dc.network_config

    def reset(self) -> None:
        return None

    def get_optim_params(self) -> dict:
        return {"params": []}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        prompt = self.config.prompt
        images = self._image_context_from_batch(batch)
        lowdims = self._state_context_from_batch(batch)

        input_vid = torch.from_numpy(images[None]).cuda().to(dtype=torch.bfloat16)
        state_tensor = torch.from_numpy(lowdims[None]).cuda().to(dtype=torch.bfloat16)
        prompt_embedding = self.prompt_embedding if prompt == self.config.prompt else None
        observation_timestep = kwargs.get("observation_timestep")
        prediction_index = self._prediction_index
        self._prediction_index += 1
        save_index = int(observation_timestep) if observation_timestep is not None else prediction_index

        with torch.no_grad():
            if self._should_save_denoised_video(prediction_index):
                actions = self._predict_actions_and_save_denoised_video(
                    input_vid=input_vid,
                    state_tensor=state_tensor,
                    prompt=prompt,
                    prompt_embedding=prompt_embedding,
                    save_index=save_index,
                    prediction_index=prediction_index,
                )[0].float()
            else:
                actions = self.pipeline(
                    input_vid=input_vid,
                    state_B_HO_O=state_tensor,
                    prompt=prompt,
                    prompt_embedding=prompt_embedding,
                    num_sampling_step=self.config.num_sampling_steps,
                    stop_after_step=self.config.stop_video_denoising_step,
                    seed=self.config.seed,
                    use_cuda_graphs=self.config.use_cuda_graphs,
                )[0].float()

        actions = actions[: self.config.horizon].cpu()
        if actions.ndim != 2 or actions.shape[1] != self.config.action_dim:
            raise RuntimeError(
                f"Expected action chunk shape (H, {self.config.action_dim}), got {tuple(actions.shape)}."
            )
        return actions.unsqueeze(0)

    def _should_save_denoised_video(self, prediction_index: int) -> bool:
        return (
            self.config.denoised_video_save_dir is not None
            and prediction_index % self.config.denoised_video_save_every_n == 0
        )

    def _predict_actions_and_save_denoised_video(
        self,
        *,
        input_vid: torch.Tensor,
        state_tensor: torch.Tensor,
        prompt: str,
        prompt_embedding: torch.Tensor | None,
        save_index: int,
        prediction_index: int,
    ) -> torch.Tensor:
        crossattn_emb, video_sigma, video = self._generate_denoised_video_and_context(
            input_vid=input_vid,
            prompt=prompt,
            prompt_embedding=prompt_embedding,
        )
        hidden_state_shape = crossattn_emb.shape
        crossattn_emb = crossattn_emb.reshape(hidden_state_shape[0], -1, hidden_state_shape[-1])
        actions = self.pipeline.world2action_pipeline(
            state_B_HO_O=state_tensor,
            crossattn_emb=crossattn_emb,
            context_timesteps_B_1=video_sigma.unsqueeze(1),
            seed=self.config.seed,
            use_cuda_graphs=self.config.use_cuda_graphs,
        )
        self._save_denoised_video(video, save_index=save_index, prediction_index=prediction_index)
        return actions

    def _generate_denoised_video_and_context(
        self,
        *,
        input_vid: torch.Tensor,
        prompt: str,
        prompt_embedding: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_pipe = self.pipeline.video2world_pipeline
        hidden_state_layer_idx = self.pipeline.world2action_pipeline.config.xattn_layer_idx
        if video_pipe.text_guardrail_runner is not None:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            if not guardrail_presets.run_text_guardrail(prompt, video_pipe.text_guardrail_runner):
                raise RuntimeError("Text guardrail error on prompt.")

        image_context_frames = input_vid.shape[2]
        if image_context_frames not in {1, 5}:
            raise ValueError(
                f"mimic-video expects 1 or 5 image context frames, got {image_context_frames}."
            )
        num_latent_conditional_frames = 1 if image_context_frames == 1 else 2

        data_batch = video_pipe._get_data_batch_input(
            input_vid,
            prompt,
            prompt_embedding,
            "",
            num_latent_conditional_frames=num_latent_conditional_frames,
        )
        video_pipe._normalize_video_databatch_inplace(data_batch)
        video_pipe._augment_image_dim_inplace(data_batch)
        is_image_batch = video_pipe.is_image_batch(data_batch)
        input_key = video_pipe.input_image_key if is_image_batch else video_pipe.input_video_key
        n_sample = data_batch[input_key].shape[0]
        _t, height, width = data_batch[input_key].shape[-3:]
        state_shape = [
            video_pipe.config.state_ch,
            video_pipe.config.state_t,
            height // video_pipe.tokenizer.spatial_compression_factor,
            width // video_pipe.tokenizer.spatial_compression_factor,
        ]
        x0_fn = video_pipe.get_x0_fn_from_batch(
            data_batch,
            guidance=0.0,
            is_negative_prompt=True,
            use_cuda_graphs=self.config.use_cuda_graphs,
        )
        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample, *tuple(state_shape)),
                torch.float32,
                video_pipe.tensor_kwargs["device"],
                self.config.seed,
            )
            * video_pipe.scheduler.config.sigma_max
        )
        if video_pipe.dit.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(
                x=x_sigma_max,
                seq_dim=2,
                cp_group=video_pipe.get_context_parallel_group(),
            )

        scheduler = video_pipe.scheduler
        scheduler.set_timesteps(self.config.num_sampling_steps, device=x_sigma_max.device)
        sample = x_sigma_max.to(dtype=torch.float32)
        x0_prev: torch.Tensor | None = None
        context_step = self.config.stop_video_denoising_step
        crossattn_emb = None
        video_sigma = None

        for i, _ in enumerate(scheduler.timesteps):
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)
            sigma_in = sigma_t.repeat(sample.shape[0])
            return_hidden_states = i == context_step
            # Partial hidden-state returns skip x0 prediction, but saving video
            # still needs x0_pred so the scheduler can finish denoising.
            x0_pred = x0_fn(
                sample,
                sigma_in,
                return_only_hidden_states_up_to=float("inf") if return_hidden_states else None,
            )
            if return_hidden_states:
                x0_pred, hidden_states = x0_pred
                crossattn_emb = hidden_states[hidden_state_layer_idx]
                video_sigma = sigma_in
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )

        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        return_final_hidden_states = context_step == len(scheduler.timesteps)
        samples = x0_fn(
            sample,
            sigma_in,
            return_only_hidden_states_up_to=float("inf") if return_final_hidden_states else None,
        )
        if return_final_hidden_states:
            samples, hidden_states = samples
            crossattn_emb = hidden_states[hidden_state_layer_idx]
            video_sigma = sigma_in

        if video_pipe.dit.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=video_pipe.get_context_parallel_group())
        if crossattn_emb is None or video_sigma is None:
            raise RuntimeError(
                "Failed to capture video context for action decoding at "
                f"denoising step {self.config.stop_video_denoising_step}."
            )
        return crossattn_emb, video_sigma, video_pipe.decode(samples)

    def _save_denoised_video(
        self,
        video: torch.Tensor,
        *,
        save_index: int,
        prediction_index: int,
    ) -> None:
        save_dir = self.config.denoised_video_save_dir
        if save_dir is None:
            return
        try:
            import imageio.v2 as imageio

            save_dir = save_dir.expanduser()
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{self.config.denoised_video_save_prefix}_obs{save_index:06d}"
                f"_pred{prediction_index:06d}.mp4"
            )
            path = save_dir / filename
            frames = self._video_tensor_to_uint8_frames(video)
            imageio.mimsave(path, frames, fps=self.config.denoised_video_save_fps)
            self.last_denoised_video_path = path
            LOGGER.info("Saved denoised mimic-video rollout to %s", path)
        except Exception:
            LOGGER.exception(
                "Failed to save denoised mimic-video rollout; continuing with action inference."
            )

    @staticmethod
    def _video_tensor_to_uint8_frames(video: torch.Tensor) -> np.ndarray:
        if video.ndim == 5:
            video = video[0]
        if video.ndim != 4:
            raise ValueError(
                "Expected denoised video with shape (B, C, T, H, W) or (C, T, H, W), "
                f"got {tuple(video.shape)}."
            )
        if video.shape[0] not in {1, 3}:
            raise ValueError(f"Expected denoised video channels-first shape, got {tuple(video.shape)}.")
        video = video.detach().float().cpu().clamp(-1.0, 1.0)
        video = ((video + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        frames = video.permute(1, 2, 3, 0).numpy()
        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        return frames

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return self.predict_action_chunk(batch, **kwargs)[0, 0]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("mimic_video is an inference-only policy.")

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image.astype(np.uint8), "RGB")
        x, y = self.config.crop_x, self.config.crop_y
        w, h = self.config.crop_width, self.config.crop_height
        pil_image = pil_image.crop((x, y, x + w, y + h))
        pil_image = pil_image.resize((self.config.image_width, self.config.image_height), Image.Resampling.BILINEAR)
        image = np.asarray(pil_image)
        tensor = rearrange(image, "h w c -> c h w")[:, None, :, :]
        return 2.0 * (tensor.astype(np.float32) / 255.0 - 0.5)

    def _image_context_from_batch(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        images = self._images_from_batch(batch)
        step = self.config.input_frequency // self.config.image_frequency
        raw_horizon = 1 + (self.config.image_horizon - 1) * step
        if len(images) < raw_horizon:
            images = [images[0].copy() for _ in range(raw_horizon - len(images))] + images
        selected = images[-raw_horizon::step]
        if len(selected) != self.config.image_horizon:
            raise RuntimeError(
                f"Expected {self.config.image_horizon} selected image frames, got {len(selected)} "
                f"from raw_horizon={raw_horizon}, step={step}."
            )
        return np.concatenate([self._process_image(image) for image in selected], axis=1)

    def _images_from_batch(self, batch: dict[str, torch.Tensor]) -> list[np.ndarray]:
        if self.config.camera_key not in batch:
            raise KeyError(f"Missing camera key {self.config.camera_key}; available keys: {sorted(batch)}")
        image = batch[self.config.camera_key]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            arr = image.numpy()
        else:
            arr = np.asarray(image)

        frames = self._split_image_frames(arr)
        return [self._normalize_image_array(frame) for frame in frames]

    @staticmethod
    def _split_image_frames(arr: np.ndarray) -> list[np.ndarray]:
        arr = np.asarray(arr)
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim == 4:
            if arr.shape[0] == 1 and arr.shape[1] in {1, 3}:
                return [arr[0]]
            return [arr[i] for i in range(arr.shape[0])]
        if arr.ndim == 3:
            return [arr]
        raise ValueError(f"Expected image tensor with 3, 4, or 5 dims, got {arr.shape}.")

    @staticmethod
    def _normalize_image_array(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 3:
            raise ValueError(f"Expected one image frame with 3 dims, got {arr.shape}.")
        if arr.shape[0] in {1, 3}:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB image frame, got shape {arr.shape}.")
        if arr.dtype != np.uint8:
            max_value = float(np.nanmax(arr)) if arr.size else 1.0
            if max_value <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _state_context_from_batch(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        if self.config.state_key not in batch:
            raise KeyError(f"Missing state key {self.config.state_key}; available keys: {sorted(batch)}")
        state = batch[self.config.state_key]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu()
            arr = state.numpy()
        else:
            arr = np.asarray(state)
        arr = np.asarray(arr, dtype=np.float32)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None]
        if arr.ndim != 2 or arr.shape[1] != self.config.action_dim:
            raise ValueError(f"Expected states with shape (T, {self.config.action_dim}), got {arr.shape}.")
        if len(arr) < self.config.lowdim_horizon:
            arr = np.concatenate([np.repeat(arr[:1], self.config.lowdim_horizon - len(arr), axis=0), arr], axis=0)
        return arr[-self.config.lowdim_horizon :]

    @staticmethod
    def _load_prompt_embedding(config: MimicVideoPolicyConfig) -> torch.Tensor:
        if config.prompt_embedding_path is not None:
            if config.prompt_embedding_path.suffix in {".pickle", ".pkl"}:
                with config.prompt_embedding_path.open("rb") as f:
                    embedding = pickle.load(f)
            else:
                embedding = torch.load(config.prompt_embedding_path, map_location="cpu")
            if isinstance(embedding, dict):
                embedding = embedding["prompt_embedding"]
            embedding = torch.as_tensor(embedding)
            return embedding.to(device=config.device, dtype=torch.bfloat16)

        text_pipe_cfg = make_config()
        text_pipe_cfg = override(text_pipe_cfg, ["--", f"experiment={config.experiment_name}"])
        text_pipe_cfg.model.config.video_pipe_config.guardrail_config.enabled = False
        text_pipe_cfg.model.config.video_pipe_config.state_t = config.video_pipe_state_t
        text_pipe = Video2WorldPipeline.from_config(
            config=text_pipe_cfg.model.config.video_pipe_config,
            dit_path=str(config.video_model_path),
            use_text_encoder=True,
            offload_text_encoder=False,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=False,
        )
        with torch.no_grad():
            embedding = text_pipe.encode_prompt(config.prompt).to(device="cuda", dtype=torch.bfloat16)
        del text_pipe
        torch.cuda.empty_cache()
        return embedding

    @staticmethod
    def _load_pipeline(config: MimicVideoPolicyConfig) -> Video2World2ActionPipeline:
        cfg = make_config()
        cfg = override(cfg, ["--", f"experiment={config.experiment_name}"])
        cfg.model.config.video_pipe_config.guardrail_config.enabled = False
        cfg.model.config.video_pipe_config.state_t = config.video_pipe_state_t
        cfg.model.config.pipe_config.xattn_layer_idx = config.xattn_layer_idx
        cfg.model.config.pipe_config.scheduler.num_denoising_steps = config.action_denoising_steps
        network_config = getattr(config, "_mimic_video_network_config", {})
        for key, value in network_config.items():
            if hasattr(cfg.model.config.pipe_config.net, key):
                setattr(cfg.model.config.pipe_config.net, key, value)
        cfg.model.config.pipe_config.net.max_horizon = config.max_horizon
        cfg.model.config.pipe_config.net.in_channels = config.action_dim
        cfg.model.config.pipe_config.net.out_channels = config.action_dim

        video2world_pipe = Video2WorldPipeline.from_config(
            config=cfg.model.config.video_pipe_config,
            dit_path=str(config.video_model_path),
            use_text_encoder=False,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=False,
        )
        world2action_pipe = World2ActionPipeline.from_config(
            config=cfg.model.config.pipe_config,
            dit_path=str(config.action_model_path),
            device="cuda",
            dtype=torch.bfloat16,
        )
        stats = load_stats(config.stats_path)
        world2action_pipe.normalizer.build_from_stats(
            stats,
            normalization_types={
                "action/joint_action_lowdim": NormalizationType.VARIANCE,
                "obs/joint_state_lowdim": NormalizationType.VARIANCE,
            },
            concat_groups={
                "action/lowdim_concat": ["action/joint_action_lowdim"],
                "obs/lowdim_concat": ["obs/joint_state_lowdim"],
            },
            device="cuda",
            dtype=torch.bfloat16,
        )
        return Video2World2ActionPipeline(video2world_pipe, world2action_pipe).cuda().eval()


MimicVideoPolicyPolicy = MimicVideoPolicy
