#!/usr/bin/env python
"""Create a random ACT checkpoint for fake SO101 async latency tests.

The model is not trained and should never be used on hardware. It exists only to
exercise LeRobot policy loading, preprocessing, inference, postprocessing, and
async action-chunk transport.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/fake_act_so101")
    parser.add_argument("--image-key", default="front")
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_feature = f"{OBS_IMAGES}.{args.image_key}"
    cfg = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            # LeRobot async inference expects visual policy features as CHW.
            image_feature: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, args.image_height, args.image_width),
            ),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
        device=args.device,
        use_amp=False,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        dim_model=128,
        n_heads=4,
        dim_feedforward=512,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        latent_dim=16,
        n_vae_encoder_layers=1,
    )

    torch.manual_seed(101)
    policy = ACTPolicy(cfg)
    policy.save_pretrained(output_dir)

    preprocessor, postprocessor = make_pre_post_processors(cfg)
    preprocessor.save_pretrained(
        output_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor.save_pretrained(
        output_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    )

    print(f"Wrote random fake ACT policy to: {output_dir}")
    print("Use only with fake_so101_follower latency tests.")


if __name__ == "__main__":
    main()

