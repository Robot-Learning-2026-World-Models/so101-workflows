# SO101 Workflows: Repo Mental Model and Cloud Inference Notes

This repo is a small operational layer around Hugging Face LeRobot for an
SO-101 leader/follower setup. It does not currently define custom robot classes,
policies, training code, or inference services. Instead, it pins LeRobot and
provides local scripts/configs for:

- teleoperating a physical SO-101 follower arm from a SO-101 leader arm
- recording LeRobot datasets and uploading them to Hugging Face Hub
- replaying recorded dataset actions on the follower
- positioning/calibrating the camera used for observations
- preserving task presets and calibration reference files

## Dependency Model

`pyproject.toml` requires Python `3.12.*` and depends on:

- `lerobot[feetech]>=0.5.1`: the main robotics stack plus Feetech/ST servo support.
- `matplotlib`: used by camera calibration visualization.
- `playsound3`: used by the camera positioning gong.
- `pybooklid`: reads MacBook lid angle for repeatable camera positioning.
- `rerun-sdk`: used by LeRobot display/telemetry paths.

`uv.lock` currently resolves LeRobot to `0.5.1`.

Important detail for future async/cloud work: LeRobot `0.5.1` exposes an
`async` extra that installs `grpcio`/protobuf support, but this repo currently
only asks for the `feetech` extra. The installed package contains
`lerobot.async_inference`, but importing it currently fails until the async extra
or equivalent `grpcio` dependency is added.

Likely future dependency change for LeRobot async inference:

```toml
"lerobot[feetech,async]>=0.5.1",
```

or an explicit `grpcio`/protobuf addition matching LeRobot's metadata.

## Local Configuration

`.env.example` shows the runtime contract expected by scripts:

- `FOLLOWER_PORT`: serial port for the follower arm motor bus.
- `LEADER_PORT`: serial port for the leader arm motor bus.
- `FOLLOWER_ID`: LeRobot robot id, used to find calibration under the HF cache.
- `LEADER_ID`: LeRobot teleoperator id, used to find calibration under the HF cache.
- `HF_DATASET_OWNER`: optional Hub user/org for datasets.
- `HF_DATASET_PRIVATE`: optional upload visibility, defaulting to private in scripts.

`.env` is intentionally ignored by git because it contains machine-specific
ports and possibly organization preferences.

Calibration files used by LeRobot are normally read from:

- follower: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<FOLLOWER_ID>.json`
- leader: `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<LEADER_ID>.json`

The checked-in files under `configs/calibration/` are useful reference/copy
material, but the scripts do not point LeRobot at them directly.

## Hardware/Feature Model

LeRobot's SO follower and leader classes both model the arm as six Feetech
STS3215 motors:

- `shoulder_pan`
- `shoulder_lift`
- `elbow_flex`
- `wrist_flex`
- `wrist_roll`
- `gripper`

The leader is a teleoperator. Its core action is:

```text
leader motor positions -> {"shoulder_pan.pos": ..., ..., "gripper.pos": ...}
```

The follower is the robot. Its observation is:

```text
follower motor positions + configured camera frames
```

Its action is:

```text
target motor positions -> sync_write("Goal_Position", ...)
```

The follower can enforce `max_relative_target` if configured, but the repo
scripts do not currently set it. For autonomous/cloud inference work, adding a
small relative target limit is a useful safety layer.

OpenCV cameras in LeRobot run a background capture thread. The follower's
`get_observation()` reads current motor positions and calls `camera.read_latest()`
for each configured camera. The camera key matters: a key like `front` becomes
the LeRobot dataset/policy feature `observation.images.front`.

## Script Entry Points

### `scripts/teleop.sh`

Purpose: run manual leader-to-follower control.

Flow:

1. Resolve repo root and require `.env`.
2. Parse optional `--display-data`.
3. Load `.env` and require `FOLLOWER_PORT`, `LEADER_PORT`, `FOLLOWER_ID`, `LEADER_ID`.
4. Apply camera defaults: index `0`, `640x480`, `30 fps`.
5. Run `uv run lerobot-teleoperate` with:
   - `--robot.type=so101_follower`
   - follower port/id/camera config
   - `--teleop.type=so101_leader`
   - leader port/id
   - optional Rerun display data

LeRobot's teleop loop does:

```text
robot.get_observation()
teleop.get_action()
default processor pipelines
robot.send_action()
optional Rerun logging
sleep to target fps
```

Use this first whenever changing ports, calibration, camera index, or physical
setup. It is the quickest end-to-end hardware sanity check.

### `scripts/record.sh`

Purpose: record a dataset from leader teleoperation and push it to HF Hub.

Flow:

1. Require `--config configs/tasks/<task>.json`.
2. Parse `--resume` and `--overwrite`; they are mutually exclusive.
3. Load `.env` and require the same arm variables as teleop.
4. Parse the task JSON with a small inline Python snippet.
5. Export known config values into shell variables:
   - `HF_DATASET_OWNER`
   - `DATASET_NAME`
   - `TASK`
   - episode/reset/count settings
   - camera index/width/height/fps
6. Check `hf auth whoami`; the script deliberately does not trigger interactive login.
7. Choose dataset owner: config owner, env owner, or logged-in HF user.
8. For `--overwrite`, remove only the local LeRobot dataset cache path.
9. Run `uv run lerobot-record` with follower, leader, camera, dataset, timing,
   visibility, and `--display_data=true`.
10. For `--resume`, pass `--resume=true` and `--dataset.root=<local cache>`.

Important details:

- The task config's `hf_dataset_owner` overrides `.env` because the JSON is
  parsed after `.env` is sourced.
- `profile_name` is exported but not used by the current shell script.
- `camera.preferred_name`, `camera.backend`, and `camera.min_test_frames` are
  documentation-like today; only `fallback_index`, `width`, `height`, and `fps`
  are consumed by `record.sh`.
- The nested `robot` fields in task configs are also not consumed today.
- `--overwrite` does not delete a Hub dataset; it only clears stale local cache.

LeRobot's record loop does:

```text
robot.get_observation()
robot_observation_processor()
build dataset observation frame
teleop.get_action() OR policy inference
teleop/policy processors
robot_action_processor()
robot.send_action()
dataset.add_frame(observation + action + task)
optional Rerun logging
sleep to dataset fps
save episode
finalize and push_to_hub
```

Even though this repo's wrapper records via teleop, LeRobot's underlying
`lerobot-record` also supports a `--policy.path=...` mode. That can become a
simple local autonomous evaluation path later.

### `scripts/replay.sh`

Purpose: replay actions from a recorded LeRobot dataset episode on the follower.

Flow:

1. Require `.env`, `FOLLOWER_PORT`, and `FOLLOWER_ID`.
2. Require `--dataset <name|owner/name>` and `--episode <n>`.
3. If a short dataset name is passed, prefix it with `HF_DATASET_OWNER`.
4. Run `uv run lerobot-replay` at `30 fps`.

LeRobot's replay loop loads the dataset episode, iterates over the action column,
converts the action vector back into named motor targets, reads current robot
observation for the default action processor, sends the action, and sleeps to the
dataset fps.

Replay is useful for checking dataset action compatibility with the current
calibration and for reproducing failures without the leader arm.

## Task Configs

Task presets live in `configs/tasks/*.json`.

Current presets:

- `so101-first-dataset.json`: `rl26-world-models/so101-pick-place-v1`,
  pick/place, 20 episodes, 20 seconds each.
- `so101-max-speed-dataset.json`: `rl26-world-models/so101-max-speed`,
  fast swing/knock task, 3 episodes, 30 seconds each.
- `so101-task1-mini.json`: `rl26-world-models/so101-task1-mini-v1`,
  roll object to target, 10 episodes, 40 seconds each.
- `so101-task2-mini.json`: `rl26-world-models/so101-task2-mini-v1`,
  obstacle-avoidance rolling task, 10 episodes, 45 seconds each.

All current presets use one OpenCV camera named `front` with index `0`,
`640x480`, and `30 fps`.

Note: `so101-task2-mini.json` currently says `"roll the from..."`; that task
string likely wants a noun inserted before future recording/training.

## Vision Utilities

### `vision/cam_preview.py`

Purpose: position and test the camera.

Flow:

1. Optionally wait until the MacBook lid reaches `--mac-lid-angle-target`.
2. Play `assets/gong.mp3` when the target angle is reached.
3. Open an OpenCV camera by `--camera-idx`.
4. Show live preview with center crop overlays for common model/image sizes.
5. Press:
   - `s` to save a clean frame to `cam_images/`
   - `o` to save an overlay frame
   - `q` to quit

This is for repeatable physical setup, not for LeRobot runtime capture.

### `vision/estimate_instrinsics.py`

Purpose: compute camera intrinsics from checkerboard images.

Flow:

1. Load `.png`, `.jpg`, and `.jpeg` files from `--src`.
2. Detect checkerboard inner corners with OpenCV.
3. Refine corners and run `cv2.calibrateCamera`.
4. Optionally visualize checkerboard poses.
5. Write `camera_intrinsic.json` or `--output` into the source directory.

The output includes both a normal OpenCV `camera_matrix`/`dist_coeff` and a
flattened `intrinsic_matrix` format expected by VidBot-style consumers.

`cam_images/camera_intrinsic.json` is an existing calibration for a `1920x1080`
camera using a `7x10` board and `17 mm` squares. The LeRobot scripts do not
currently consume this file.

## Assets and Ignored Outputs

- `assets/gong.mp3`: audible cue for camera setup.
- `vision/checkerboard_7_10.pdf`: calibration target.
- `cam_images/*.png`: ignored; generated preview/calibration images.
- `.venv/`, `.env`, `outputs/`, caches, and bytecode are ignored.

## Current Missing Pieces

This repo currently does not include:

- policy training configs or wrappers
- policy checkpoints
- autonomous inference scripts
- async cloud inference wrappers
- safety watchdogs beyond LeRobot defaults
- network security for cloud inference
- dataset validation or camera bandwidth measurement tooling

Those are natural next additions.

## Cloud Inference Hypotheses

The physical constraint is non-negotiable: the MacBook connected to the SO-101
must own the serial motor bus and local camera capture. The cloud should produce
actions or action chunks; it should not directly own the USB motor bus.

### Hypothesis 1: Use LeRobot Async Inference Directly

This is the most natural MVP.

Architecture:

```text
MacBook + SO101 follower + camera
  -> RobotClient captures observation
  -> sends timestamped observation to cloud PolicyServer over gRPC
  -> cloud loads policy and predicts an action chunk
  -> MacBook receives action chunk into local queue
  -> MacBook executes queued actions on follower at target fps
```

Why it fits:

- LeRobot already has `lerobot.async_inference.robot_client`.
- LeRobot already has `lerobot.async_inference.policy_server`.
- The design explicitly decouples action execution from action prediction.
- The client keeps an action queue, sends fresh observations before the queue is
  empty, and can aggregate overlapping action chunks.
- Supported policy types in the installed code include `act`, `smolvla`,
  `diffusion`, `tdmpc`, `vqbet`, `pi0`, `pi05`, and `groot`.

Immediate repo work needed:

1. Add the LeRobot async dependency.
2. Add `scripts/async_server.sh` for cloud.
3. Add `scripts/async_client.sh` for the MacBook, mirroring `.env`/camera
   defaults from `teleop.sh`.
4. Add a cloud inference config file, probably under `configs/inference/`.
5. Run loopback first, LAN second, cloud third.

Example cloud server command:

```bash
uv run python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=2
```

Example MacBook client command:

```bash
uv run python -m lerobot.async_inference.robot_client \
  --server_address=<private-cloud-address>:8080 \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${CAMERA_INDEX:-0}, width: ${CAMERA_WIDTH:-640}, height: ${CAMERA_HEIGHT:-480}, fps: ${CAMERA_FPS:-30}}}" \
  --task="<task string>" \
  --policy_type=<act|smolvla|diffusion|pi0|pi05|groot> \
  --pretrained_name_or_path=<hf-user-or-org/model> \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=20 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=True
```

Safety/security caveat: LeRobot's current async code uses `grpc.insecure_channel`
and Python `pickle` for serialized observations/actions. Treat that as trusted
private infrastructure only. Do not expose the server directly to the public
internet. Use a private network/tunnel such as WireGuard, Tailscale, SSH tunnel,
or a patched TLS/authenticated transport before serious cloud testing.

Main risks:

- Raw image observations are large. A single `640x480x3` frame is about `0.9 MB`
  before serialization overhead. At high request rates this can dominate latency.
- Internet jitter can drain the action queue.
- Action chunks predicted from stale observations may become wrong if the scene
  changes quickly.
- Pickle over an unauthenticated network is unsafe.

Best first tuning knobs:

- `actions_per_chunk`: start around `10-20`, increase if the queue drains.
- `chunk_size_threshold`: start around `0.5`, increase for more updates if the
  cloud can keep up.
- camera resolution: keep `640x480` for the first physical test, then reduce or
  compress if bandwidth is the bottleneck.
- fps: reduce below `30` if the queue repeatedly empties.

### Hypothesis 2: Custom Compressed Action-Chunk Service

This is the best path if Hypothesis 1 works conceptually but bandwidth/security
are poor.

Architecture:

```text
MacBook control loop
  -> capture state + image
  -> encode image as JPEG/WebP/H.264 frame or policy-sized tensor
  -> send typed message to cloud
  -> cloud runs LeRobot policy internals
  -> return typed action chunk
  -> local queue/safety/filter executes action
```

Why it might be better than stock async:

- Replace pickle with a typed schema such as protobuf/flatbuffers/msgpack.
- Add TLS/auth from the beginning.
- Compress images before transport.
- Send only the camera views and dimensions expected by the policy.
- Add explicit stale-action handling and end-to-end timestamps.

Costs:

- More code to maintain.
- Must reproduce the LeRobot preprocessing/postprocessing path carefully.
- More ways to introduce subtle feature-name or normalization mismatches.

This path should still reuse LeRobot's policy loading, preprocessors,
postprocessors, and dataset feature conventions. Do not invent a separate action
schema unless there is a strong reason.

### Hypothesis 3: Local System-1 Controller with Cloud System-2 Policy

This is the robust robotics version.

Architecture:

```text
cloud large policy: lower-rate action chunks / waypoints / task corrections
local MacBook: high-rate safety + smoothing + queue + last-good fallback
SO101 follower: receives only locally validated actions
```

The "system 1" part does not need to start as a learned model. A practical first
version is:

- local action queue
- interpolation between chunk actions
- clamp `max_relative_target`
- stale action timeout
- torque-off or hold-position behavior on disconnect
- optional tiny local policy only if cloud reliability is not enough

This pairs well with Hypothesis 1 because LeRobot's async client already behaves
like a simple local action queue. The next layer would be a local safety/action
processor before `robot.send_action()`.

Why it fits the MacBook:

- The MacBook has enough memory to host a fallback policy or at least a local
  queue/interpolator.
- The cloud can stay responsible for expensive VLA/world-model inference.
- The robot never depends on a single remote packet for immediate motor safety.

### Hypothesis 4: Local Inference First, Cloud for Larger Models/Training

This is worth benchmarking even if the desired destination is cloud.

Architecture:

```text
MacBook runs policy locally with MPS/CPU
cloud is used for training, larger teacher models, or batch evaluation
```

Why try it:

- Establishes a latency baseline with no network.
- Helps distinguish model latency from network latency.
- Gives a fallback story for demos.
- LeRobot's synchronous policy path in `lerobot-record --policy.path=...` already
  exists, though this repo does not wrap it yet.

Limitations:

- Large VLA policies may be too slow locally despite fitting in memory.
- MPS support depends on the policy and PyTorch ops used.
- It does not satisfy the preferred "everything in cloud" goal, but it is a
  useful control experiment.

### Hypothesis 5: WebRTC Media/Data Transport

This is a stronger custom-transport option if camera bandwidth is the bottleneck.

Architecture:

```text
MacBook sends low-latency encoded video to cloud over WebRTC
cloud sends action chunks back over WebRTC data channel
local MacBook executes validated actions
```

Why it could help:

- WebRTC has mature low-latency video transport, congestion control, NAT
  traversal, and encryption.
- The cloud can receive compressed video instead of raw frame pickles.

Why it is not the first MVP:

- It bypasses LeRobot's existing async inference stack.
- It requires careful frame timestamping and policy preprocessing.
- Video compression can change visual observations; this needs validation
  against the trained policy's expected input distribution.

### Hypothesis 6: Cloud Owns USB/Motor Bus Remotely

This is the one to avoid.

Streaming USB serial control over the network or letting a cloud VM directly
command the motor bus creates avoidable latency and safety risk. Keep motor
ownership local. Cloud should output proposed actions; the MacBook should remain
the final authority.

## Recommended Experiment Order

1. Confirm baseline hardware:

```bash
bash scripts/teleop.sh --display-data
```

2. Confirm dataset/replay path:

```bash
bash scripts/record.sh --config configs/tasks/so101-task1-mini.json
bash scripts/replay.sh --dataset so101-task1-mini-v1 --episode 0
```

3. Add async dependency and run imports:

```bash
uv run python -c "import grpc; import lerobot.async_inference.robot_client"
```

4. Run LeRobot async loopback on the MacBook before cloud.

5. Run async over LAN or a private tunnel.

6. Run async on a cloud GPU behind a private network, never as a raw public gRPC
   server.

7. Measure:
   - client control-loop fps
   - server inference time
   - observation serialization size/time
   - one-way observation latency
   - action receive latency
   - action queue low-water mark
   - number of empty-queue events
   - visible robot smoothness and task success

8. Only then decide whether stock LeRobot async is enough or whether to build a
   compressed/custom transport.

## Safety Checklist for Autonomous/Cloud Work

- Keep a physical emergency stop or power cutoff available.
- Start with low-speed, low-torque, unobstructed movements.
- Add `max_relative_target` to follower config or wrapper command before fast
  autonomous motion.
- Implement stale-action timeout: if no fresh action chunk arrives within a
  threshold, hold position or disable torque.
- Reject action values outside learned/calibrated joint ranges.
- Log every action actually sent, not just every action predicted.
- Use private authenticated networking before any cloud test.
- Avoid public pickle/gRPC exposure.
- Keep teleop available for intervention when testing policy rollouts.

## Useful Upstream References

- SO-101 setup/calibration docs:
  https://huggingface.co/docs/lerobot/so101
- LeRobot async inference docs:
  https://huggingface.co/docs/lerobot/async
- LeRobotDataset v3 docs:
  https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
- Current main-branch policy deployment/RTC docs, useful for future upgrades but
  not present as a `lerobot-rollout` entrypoint in this repo's locked LeRobot
  `0.5.1` environment:
  https://huggingface.co/docs/lerobot/main/inference
