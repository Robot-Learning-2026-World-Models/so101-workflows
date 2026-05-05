# Fake SO101 Async Latency Testing

This repo includes a fake LeRobot robot type named `fake_so101_follower`. It is
for testing async inference plumbing without connecting the physical SO101 arm.

It mimics the real follower's feature shape:

- state/action keys:
  - `shoulder_pan.pos`
  - `shoulder_lift.pos`
  - `elbow_flex.pos`
  - `wrist_flex.pos`
  - `wrist_roll.pos`
  - `gripper.pos`
- camera key: `front` by default
- camera frame: synthetic `uint8` RGB image, default `640x480`

It does not simulate physics. It only lets you measure:

- async client/server connectivity
- observation serialization and network cost
- policy inference time on the server
- action chunk receive latency
- local action queue behavior
- local control loop frequency

## Dependency

`pyproject.toml` must include the LeRobot async extra:

```toml
"lerobot[feetech,async]>=0.5.1"
```

Then sync once:

```bash
uv sync
```

## Run

Create a disposable random ACT policy that matches the fake robot:

```bash
uv run python scripts/create_fake_act_policy.py
```

Terminal 1, start the policy server:

```bash
bash scripts/async_server.sh
```

Terminal 2, run the fake SO101 client:

```bash
PRETRAINED_NAME_OR_PATH=outputs/fake_act_so101 \
POLICY_TYPE=act \
POLICY_DEVICE=cpu \
DURATION_S=30 \
bash scripts/fake_async_latency_client.sh
```

For a cloud/GPU server, run `scripts/async_server.sh` on the server and point the
client at it:

```bash
SERVER_ADDRESS=<private-server-ip>:8080 \
PRETRAINED_NAME_OR_PATH=outputs/fake_act_so101 \
POLICY_TYPE=act \
POLICY_DEVICE=cuda \
DURATION_S=60 \
bash scripts/fake_async_latency_client.sh
```

Use only a private network or tunnel. LeRobot async uses insecure gRPC and pickle
internally in this version.

## Useful Knobs

The fake client script reads these environment variables:

- `SERVER_ADDRESS`, default `127.0.0.1:8080`
- `POLICY_TYPE`, default `act`
- `PRETRAINED_NAME_OR_PATH`, required
- `POLICY_DEVICE`, default `cpu`
- `CLIENT_DEVICE`, default `cpu`
- `TASK`, default `dummy`
- `DURATION_S`, default `30`
- `FPS`, default `30`
- `ACTIONS_PER_CHUNK`, default `20`
- `CHUNK_SIZE_THRESHOLD`, default `0.5`
- `CAMERA_WIDTH`, default `640`
- `CAMERA_HEIGHT`, default `480`
- `CAMERA_KEY`, default `front`

You can also pass fake robot knobs directly if you run the Python module:

```bash
uv run python -m so101_workflows.fake_async_latency_client \
  --server_address=127.0.0.1:8080 \
  --policy_type=act \
  --pretrained_name_or_path=<your-act-policy> \
  --policy_device=cpu \
  --client_device=cpu \
  --actions_per_chunk=20 \
  --chunk_size_threshold=0.5 \
  --duration_s=30 \
  --robot.type=fake_so101_follower \
  --robot.id=fake_so101 \
  --robot.camera_width=320 \
  --robot.camera_height=240 \
  --robot.observation_latency_ms=5 \
  --robot.action_latency_ms=1
```

## Public ACT Checkpoints

You can point `PRETRAINED_NAME_OR_PATH` at a public LeRobot ACT checkpoint, but
it must expect the same feature names and dimensions as the fake robot. A random
SO101 checkpoint on the Hub may use different camera keys, camera counts, image
shapes, or action conventions. The generated `outputs/fake_act_so101` checkpoint
is safer when you only care about latency plumbing.

## Reading Results

The client prints a summary:

- `avg_control_hz`: the local control loop rate achieved.
- `observations_sent`: how often the client sent observations to the server.
- `actions_executed`: how many queued actions were sent to the fake robot.
- `avg_loop_ms` / `max_loop_ms`: local client loop timing.
- `queue_min` / `queue_max` / `queue_final`: action queue health.

The server logs are where policy inference timing appears. Watch for:

- observation receive one-way latency
- inference time
- serialization time
- total action chunk generation time

If `queue_min` repeatedly reaches `0`, the client is starving for actions. Try:

- increasing `ACTIONS_PER_CHUNK`
- increasing `CHUNK_SIZE_THRESHOLD`
- reducing `FPS`
- reducing `CAMERA_WIDTH` / `CAMERA_HEIGHT`
- moving the policy server closer to the robot client
