from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from imaginaire.utils.config_helper import override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a static mimic-video prompt once and save it.")
    parser.add_argument("--video-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--experiment-name",
        default="w2a_so101_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1",
    )
    parser.add_argument("--video-pipe-state-t", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_config()
    cfg = override(cfg, ["--", f"experiment={args.experiment_name}"])
    cfg.model.config.video_pipe_config.guardrail_config.enabled = False
    cfg.model.config.video_pipe_config.state_t = args.video_pipe_state_t

    pipe = Video2WorldPipeline.from_config(
        config=cfg.model.config.video_pipe_config,
        dit_path=str(args.video_model_path),
        use_text_encoder=True,
        offload_text_encoder=False,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
    )
    with torch.inference_mode():
        embedding = pipe.encode_prompt(args.prompt).detach().cpu().to(torch.float16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompt": args.prompt,
            "prompt_embedding": embedding,
            "experiment_name": args.experiment_name,
            "video_pipe_state_t": args.video_pipe_state_t,
        },
        args.output,
    )
    print(f"saved prompt embedding to {args.output}")
    print(f"shape={tuple(embedding.shape)} dtype={embedding.dtype}")


if __name__ == "__main__":
    main()
