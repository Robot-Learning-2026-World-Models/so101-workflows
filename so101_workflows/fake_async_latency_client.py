import logging
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins

# Importing this module registers `fake_so101_follower` with LeRobot.
import so101_workflows.fake_so101  # noqa: F401


@dataclass
class FakeLatencyClientConfig(RobotClientConfig):
    duration_s: float = 30.0
    verbose: bool = True


@draccus.wrap()
def main(cfg: FakeLatencyClientConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    register_third_party_plugins()

    client = RobotClient(cfg)
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
    }

    print("\nLatency test summary")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if cfg.debug_visualize_queue_size and queue_sizes:
        visualize_action_queue_size(queue_sizes)


if __name__ == "__main__":
    main()
