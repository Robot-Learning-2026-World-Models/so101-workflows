from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@RobotConfig.register_subclass("fake_so101_follower")
@dataclass(kw_only=True)
class FakeSO101FollowerConfig(RobotConfig):
    """A no-hardware SO101-shaped robot for async inference latency tests."""

    camera_key: str = "front"
    camera_width: int = 640
    camera_height: int = 480
    fps: int = 30
    observation_latency_ms: float = 0.0
    action_latency_ms: float = 0.0
    motion_alpha: float = 0.35
    noise_std: float = 0.0
    log_every: int = 30


class FakeSO101Follower(Robot):
    """LeRobot Robot implementation that mimics SO101 feature names.

    It keeps a tiny joint-state simulation in memory and produces a synthetic RGB
    image. This is meant for network/control-loop testing only, not physics.
    """

    config_class = FakeSO101FollowerConfig
    name = "fake_so101_follower"

    def __init__(self, config: FakeSO101FollowerConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._step = 0
        self._state = {name: 0.0 for name in JOINT_NAMES}
        self._target = dict(self._state)
        self._rng = np.random.default_rng(101)
        self._last_log_t = time.perf_counter()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        motor_features = {f"{joint}.pos": float for joint in JOINT_NAMES}
        camera_features = {
            self.config.camera_key: (
                self.config.camera_height,
                self.config.camera_width,
                3,
            )
        }
        return {**motor_features, **camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in JOINT_NAMES}

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True
        logging.info("%s connected.", self)

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> RobotObservation:
        self._require_connected()
        self._sleep_ms(self.config.observation_latency_ms)

        self._step += 1
        alpha = float(np.clip(self.config.motion_alpha, 0.0, 1.0))
        for joint in JOINT_NAMES:
            self._state[joint] = (1.0 - alpha) * self._state[joint] + alpha * self._target[joint]

        obs: RobotObservation = {f"{joint}.pos": float(self._state[joint]) for joint in JOINT_NAMES}
        if self.config.noise_std > 0:
            for joint in JOINT_NAMES:
                obs[f"{joint}.pos"] += float(self._rng.normal(0.0, self.config.noise_std))

        obs[self.config.camera_key] = self._make_image()
        self._log_periodically("observation")
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        self._require_connected()
        self._sleep_ms(self.config.action_latency_ms)

        sent: RobotAction = {}
        for joint in JOINT_NAMES:
            key = f"{joint}.pos"
            value = float(action.get(key, self._target[joint]))
            self._target[joint] = value
            sent[key] = value
        self._log_periodically("action")
        return sent

    def disconnect(self) -> None:
        self._connected = False
        logging.info("%s disconnected.", self)

    def _make_image(self) -> np.ndarray:
        h = self.config.camera_height
        w = self.config.camera_width
        x = np.linspace(0, 255, w, dtype=np.uint16)[None, :]
        y = np.linspace(0, 255, h, dtype=np.uint16)[:, None]
        image = np.empty((h, w, 3), dtype=np.uint8)
        image[:, :, 0] = ((x + self._step) % 256).astype(np.uint8)
        image[:, :, 1] = ((y + 2 * self._step) % 256).astype(np.uint8)
        joint_signal = int(sum(abs(v) for v in self._state.values()) * 3) % 255
        image[:, :, 2] = joint_signal
        return image

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(f"{self} is not connected.")

    @staticmethod
    def _sleep_ms(ms: float) -> None:
        if ms > 0:
            time.sleep(ms / 1000.0)

    def _log_periodically(self, event: str) -> None:
        if self.config.log_every <= 0 or self._step % self.config.log_every != 0:
            return
        now = time.perf_counter()
        dt = max(now - self._last_log_t, 1e-9)
        logging.info(
            "Fake SO101 %s step=%s approx_rate=%.1f Hz target=%s",
            event,
            self._step,
            self.config.log_every / dt,
            {k: round(v, 2) for k, v in self._target.items()},
        )
        self._last_log_t = now
