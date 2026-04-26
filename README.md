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

## Troubleshooting

- `objc ... AVFFrameReceiver ... implemented in both cv2 and av` warning on macOS:

  ```bash
  pip uninstall -y av
  ```

  Then rerun `bash scripts/teleop.sh`.

- Calibration files location (local, not in git):
  - follower: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<FOLLOWER_ID>.json`
  - leader: `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<LEADER_ID>.json`
