import logging
import pickle  # nosec
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import grpc

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.transport import services_pb2
from lerobot.utils.import_utils import register_third_party_plugins

# Importing this module registers `fake_so101_follower` with LeRobot.
import so101_workflows.fake_so101  # noqa: F401


@dataclass
class FakeLatencyClientConfig(RobotClientConfig):
    duration_s: float = 30.0
    verbose: bool = True


def _summary_stats_ms(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None, "avg": None}

    sorted_values = sorted(values)

    def percentile(p: float) -> float:
        index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * p)))
        return round(sorted_values[index], 3)

    return {
        "count": len(values),
        "min": round(sorted_values[0], 3),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": round(sorted_values[-1], 3),
        "avg": round(statistics.fmean(values), 3),
    }


class InstrumentedRobotClient(RobotClient):
    """RobotClient with explicit cloud/action timing metrics.

    LeRobot's built-in verbose "Network latency" is useful, but it is not enough
    to answer when cloud-generated actions actually execute locally.
    """

    def __init__(self, config: RobotClientConfig):
        super().__init__(config)
        self.metrics_lock = threading.Lock()
        self.chunk_receive_latencies_ms: list[float] = []
        self.chunk_receive_times: list[float] = []
        self.action_schedule_lag_ms: list[float] = []
        self.obs_to_action_execute_ms: list[float] = []
        self.first_action_obs_to_execute_ms: list[float] = []
        self.action_origins: dict[int, tuple[float, int]] = {}

    def receive_actions(self, verbose: bool = False):
        """Receive action chunks and record observation-to-chunk timing."""

        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                receive_time = time.time()
                receive_perf = time.perf_counter()

                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                if len(timed_actions) > 0:
                    client_device = self.config.client_device
                    if client_device != "cpu":
                        for timed_action in timed_actions:
                            if timed_action.get_action().device.type != client_device:
                                timed_action.action = timed_action.get_action().to(client_device)

                    self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                    first_action_timestep = timed_actions[0].get_timestep()
                    first_action_timestamp = timed_actions[0].get_timestamp()
                    chunk_latency_ms = (receive_time - first_action_timestamp) * 1000

                    with self.metrics_lock:
                        self.chunk_receive_latencies_ms.append(chunk_latency_ms)
                        self.chunk_receive_times.append(receive_perf)
                        for timed_action in timed_actions:
                            self.action_origins[timed_action.get_timestep()] = (
                                first_action_timestamp,
                                first_action_timestep,
                            )

                    if verbose:
                        with self.latest_action_lock:
                            latest_action = self.latest_action

                        old_size, old_timesteps = self._inspect_action_queue()
                        if not old_timesteps:
                            old_timesteps = [latest_action]

                        incoming_timesteps = [a.get_timestep() for a in timed_actions]

                        self.logger.info(
                            f"Received action chunk for step #{first_action_timestep} | "
                            f"Latest action: #{latest_action} | "
                            f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                            f"Obs-to-chunk-receive latency: {chunk_latency_ms:.2f}ms | "
                            f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                        )

                    start_time = time.perf_counter()
                    self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                    queue_update_time = time.perf_counter() - start_time

                    self.must_go.set()

                    if verbose:
                        new_size, new_timesteps = self._inspect_action_queue()
                        with self.latest_action_lock:
                            latest_action = self.latest_action

                        self.logger.info(
                            f"Latest action: {latest_action} | "
                            f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                            f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                            f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                        )
                        self.logger.debug(
                            f"Queue update complete ({queue_update_time:.6f}s) | "
                            f"Before: {old_size} items | "
                            f"After: {new_size} items | "
                        )

            except grpc.RpcError as e:
                if not self.running:
                    break
                self.logger.error(f"Error receiving actions: {e}")

    def control_loop_action(self, verbose: bool = False) -> dict:
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        action_execute_time = time.time()

        action_timestep = timed_action.get_timestep()
        action_timestamp = timed_action.get_timestamp()
        with self.latest_action_lock:
            self.latest_action = action_timestep

        with self.metrics_lock:
            origin = self.action_origins.pop(action_timestep, None)
            self.action_schedule_lag_ms.append((action_execute_time - action_timestamp) * 1000)
            if origin is not None:
                origin_timestamp, origin_timestep = origin
                obs_to_execute_ms = (action_execute_time - origin_timestamp) * 1000
                self.obs_to_action_execute_ms.append(obs_to_execute_ms)
                if action_timestep == origin_timestep:
                    self.first_action_obs_to_execute_ms.append(obs_to_execute_ms)

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={action_timestamp} | "
                f"Action #{action_timestep} performed | "
                f"Queue size: {current_queue_size}"
            )
            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action


@draccus.wrap()
def main(cfg: FakeLatencyClientConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    register_third_party_plugins()

    client = InstrumentedRobotClient(cfg)
    action_receiver_thread: threading.Thread | None = None
    loop_durations: list[float] = []
    observations_sent = 0
    actions_executed = 0

    if not client.start():
        raise RuntimeError(f"Could not connect to policy server at {cfg.server_address}")

    action_receiver_thread = threading.Thread(
        target=client.receive_actions,
        kwargs={"verbose": cfg.verbose},
        daemon=True,
    )
    action_receiver_thread.start()

    client.start_barrier.wait()
    deadline = time.perf_counter() + cfg.duration_s
    started = time.perf_counter()
    try:
        while client.running and time.perf_counter() < deadline:
            loop_start = time.perf_counter()

            if client.actions_available():
                client.control_loop_action(verbose=cfg.verbose)
                actions_executed += 1

            if client._ready_to_send_observation():
                client.control_loop_observation(cfg.task, verbose=cfg.verbose)
                observations_sent += 1

            elapsed = time.perf_counter() - loop_start
            loop_durations.append(elapsed)
            time.sleep(max(0.0, cfg.environment_dt - elapsed))
    finally:
        client.stop()
        if action_receiver_thread is not None:
            action_receiver_thread.join(timeout=2.0)

    total_s = max(time.perf_counter() - started, 1e-9)
    queue_sizes = client.action_queue_size
    chunk_intervals_ms = [
        1000 * (later - earlier)
        for earlier, later in zip(client.chunk_receive_times, client.chunk_receive_times[1:], strict=False)
    ]
    summary = {
        "duration_s": round(total_s, 3),
        "target_fps": cfg.fps,
        "avg_control_hz": round(len(loop_durations) / total_s, 2),
        "observations_sent": observations_sent,
        "actions_executed": actions_executed,
        "avg_loop_ms": round(1000 * statistics.fmean(loop_durations), 3) if loop_durations else None,
        "max_loop_ms": round(1000 * max(loop_durations), 3) if loop_durations else None,
        "queue_min": min(queue_sizes) if queue_sizes else None,
        "queue_max": max(queue_sizes) if queue_sizes else None,
        "queue_final": queue_sizes[-1] if queue_sizes else None,
        "chunk_receive_latency_ms_all": _summary_stats_ms(client.chunk_receive_latencies_ms),
        "chunk_receive_latency_ms_after_first": _summary_stats_ms(client.chunk_receive_latencies_ms[1:]),
        "chunk_receive_interval_ms_after_first": _summary_stats_ms(chunk_intervals_ms[1:]),
        "chunk_receive_hz_after_first": (
            round(1000 / statistics.fmean(chunk_intervals_ms[1:]), 3)
            if len(chunk_intervals_ms) > 1
            else None
        ),
        "action_schedule_lag_ms": _summary_stats_ms(client.action_schedule_lag_ms),
        "obs_to_action_execute_ms": _summary_stats_ms(client.obs_to_action_execute_ms),
        "first_action_obs_to_execute_ms": _summary_stats_ms(client.first_action_obs_to_execute_ms),
        "first_action_obs_to_execute_ms_after_first": _summary_stats_ms(
            client.first_action_obs_to_execute_ms[1:]
        ),
    }

    print("\nLatency test summary")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if cfg.debug_visualize_queue_size and queue_sizes:
        visualize_action_queue_size(queue_sizes)


if __name__ == "__main__":
    main()
