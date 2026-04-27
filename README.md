# so101-workflows

Helper scripts for SO-101 teleoperation using LeRobot as a third-party dependency.

## 1) Create and activate the conda env

This project uses a conda env named `lerobot`:

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
```

LeRobot installation reference:
[Hugging Face LeRobot Installation](https://huggingface.co/docs/lerobot/en/installation)

## 2) Install uv in that env

```bash
pip install -U uv
```

## 3) Install project dependencies from lockfile

From this repo root:

```bash
uv sync --frozen
```

If `uv.lock` is missing (first-time bootstrap), run:

```bash
uv sync
```

## 4) Configure local robot settings

```bash
cp .env.example .env
```

Then set:

- `FOLLOWER_PORT`
- `LEADER_PORT`
- `FOLLOWER_ID`
- `LEADER_ID`

## 5) Run teleoperation

```bash
bash scripts/teleop.sh
```

Optional flags:

- enable data display: `bash scripts/teleop.sh --display-data`

## 6) Record and upload dataset to Hugging Face (user or org)

Task configs live in `configs/tasks/*.json`, and `--config` is required:

```bash
bash scripts/record.sh --config configs/tasks/<TASK_NAME>.json
```

Choose your config first before recording, especially if you switch datasets
often.

`scripts/record.sh` uploads to the logged-in HF user by default.

To upload to an org (for example `rl26-world-models`), set this in `.env`:

```bash
HF_DATASET_OWNER=rl26-world-models
```

Uploads are private by default. To make uploads public, set this in `.env`:

```bash
HF_DATASET_PRIVATE=false
```

Then run:

```bash
bash scripts/record.sh --config configs/tasks/<TASK_NAME>.json
```

For repeated runs on the same dataset id, prefer resume:

```bash
bash scripts/record.sh --config configs/tasks/<TASK_NAME>.json --resume
```

`--resume` continues from local cache at
`~/.cache/huggingface/lerobot/<owner>/<dataset_name>`.

When resuming, you can update timing/count settings in your task config (for
example `num_episodes`, `episode_time_s`, `reset_time_s`). These changes apply
to newly recorded episodes only; previously recorded episodes are unchanged.

Use overwrite when you deleted/reset the Hub dataset and want to recreate it
with the same dataset id (this clears stale local cache that can cause local
path conflicts):

```bash
bash scripts/record.sh --config configs/tasks/<TASK_NAME>.json --overwrite
```

`--overwrite` only removes local cache. It does not delete data already on the Hub.

To create a new recording preset, copy an existing config file and update:

- `profile_name` (local label)
- `dataset_name` (HF dataset repo name)
- `task`
- `num_episodes`, `episode_time_s`, `reset_time_s`
- camera fields under `camera` (`fallback_index`, `width`, `height`, `fps`)

Keep it minimal before recording:

- verify the camera index with `lerobot-find-cameras opencv`
- use a unique `dataset_name` for each dataset version

## Troubleshooting

- `objc ... AVFFrameReceiver ... implemented in both cv2 and av` warning on macOS:

  Keep `av` installed (`lerobot` imports it). If this warning appears but runs
  are stable, it is usually safe to continue.

- Camera index issues (wrong camera selected or no frames):

  ```bash
  lerobot-find-cameras opencv
  ```

  Use the reported OpenCV camera index for your setup.

- Calibration files location (local, not in git):
  - follower: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<FOLLOWER_ID>.json`
  - leader: `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<LEADER_ID>.json`
