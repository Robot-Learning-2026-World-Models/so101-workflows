# Smoke Policy How-To

Run smoke inference:

```bash
./commands/smoke_policy.sh
```

This writes:

- predicted video: `outputs/smoke_policy_videos/`
- trajectory JSON: `outputs/smoke_policy_trajectories/`

Visualize the latest trajectory locally:

```bash
./commands/trajectory_to_lerobot_dataset.sh
./commands/visualize_trajectory_dataset.sh
```

Push the latest trajectory to Hugging Face:

```bash
./commands/push_trajectory_to_hf.sh
```

The push command converts the latest trajectory internally and uploads it to
`rl26-world-models/inference-viz-1`; no manual conversion step is needed first.
The uploaded dataset includes a small video stream so the Hugging Face LeRobot
visualizer can open it.
