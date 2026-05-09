# so101-workflows

Helper scripts for SO-101 teleoperation using LeRobot as a third-party dependency.

## 1) Install LeRobot with uv

Use Python 3.12, then install LeRobot from PyPI with Feetech motor support:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install 'lerobot[feetech]'
uv sync
```

On macOS, install `ffmpeg` if it is missing:

```bash
brew install ffmpeg
```

LeRobot installation reference:
[Hugging Face LeRobot Installation](https://huggingface.co/docs/lerobot/en/installation?create_venv=uv+%28PyTorch+%3E%3D+2.10+only%29&install_lerobot_pypi=uv&install_lerobot_src=uv&install_ffmpeg=uv+%28PyTorch+%3E%3D+2.10+only%29&activate_venv=uv+%28PyTorch+%3E%3D+2.10+only%29)

## 2) Configure local robot settings

```bash
cp .env.example .env
```

Find the ports for the follower and leader arms:

```bash
uv run lerobot-find-port
```

Then set these values in `.env`:

- `FOLLOWER_PORT`
- `LEADER_PORT`
- `FOLLOWER_ID`
- `LEADER_ID`

## 3) Copy calibration files into the LeRobot cache

After `.env` has the correct IDs, copy the checked-in calibration files to the
cache paths LeRobot expects:

```bash
set -a; source .env; set +a
mkdir -p \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower" \
  "$HOME/.cache/huggingface/lerobot/calibration/teleoperators/so_leader"
cp "configs/calibration/follower/${FOLLOWER_ID}.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/${FOLLOWER_ID}.json"
cp "configs/calibration/leader/${LEADER_ID}.json" \
  "$HOME/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/${LEADER_ID}.json"
```

## 4) Run teleoperation

```bash
bash scripts/teleop.sh
```

Optional flags:

- enable data display: `bash scripts/teleop.sh --display-data`

## 5) Record and upload dataset to Hugging Face (user or org)

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
`~/.cache/huggingface/lerobot/<owner>/<dataset_name>`. On each resume run the
script first **downloads the full dataset from the Hub** into that path to prevent overwriting Hub data with metadata-only copy.

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

## 6) Replay an episode on the follower

Replays stored actions from a LeRobot dataset (local cache under
`~/.cache/huggingface/lerobot/...`, or pulled from the Hub if needed).

```bash
bash scripts/replay.sh --dataset <dataset_repo_name_or_owner/name> --episode <index>
```

If you pass just the repo name (as in your task config’s `dataset_name`), set
`HF_DATASET_OWNER` in `.env` (same as for recording). Or pass the full id, for
example `rl26-world-models/so101-pick-place-v1`. Replay runs at 30 fps.

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
