#!/usr/bin/env python3
"""
Reassign ID and baud rate for one Feetech motor on an SO-100 / SO-101 arm.

`lerobot-setup-motors` walks every joint in order. Use this when only one
servo was replaced (for example after a burnout): wire that motor alone to
the controller, then run this script for the matching joint name.

Equivalent to LeRobot's per-motor path:
  SOFollower.setup_motors / SOLeader.setup_motors -> FeetechMotorsBus.setup_motor

After success, reconnect the full arm and run calibration if limits/homing
on the new servo do not match your saved JSON (see README calibration paths).
"""

from __future__ import annotations

import argparse
import sys

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

SO_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Program a single replacement Feetech motor to the ID and baud rate "
            "LeRobot expects for that joint on an SO follower or leader."
        )
    )
    p.add_argument(
        "--port",
        required=True,
        help="Serial device (same as lerobot-setup-motors), e.g. /dev/tty.usbmodem…",
    )
    p.add_argument(
        "--role",
        choices=("follower", "leader"),
        required=True,
        help="Whether this is the follower arm or the leader (teleop) arm.",
    )
    p.add_argument(
        "--motor",
        choices=SO_JOINTS,
        required=True,
        help="Joint name for the motor you wired alone to the bus.",
    )
    p.add_argument(
        "--robot-id",
        default=None,
        help="Optional follower id; loads ~/.cache/.../so_follower/<id>.json if present.",
    )
    p.add_argument(
        "--teleop-id",
        default=None,
        help="Optional leader id; loads ~/.cache/.../so_leader/<id>.json if present.",
    )
    p.add_argument(
        "--initial-baudrate",
        type=int,
        default=None,
        help="Current baud rate of the factory motor (skips scanning when set).",
    )
    p.add_argument(
        "--initial-id",
        type=int,
        default=None,
        help="Current servo ID of the lone motor on the bus (skips scanning when set).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.role == "follower":
        device = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    else:
        device = SO101Leader(SO101LeaderConfig(port=args.port, id=args.teleop_id))

    bus = device.bus
    motor = args.motor
    target_id = bus.motors[motor].id

    print(
        f"Connect only the replacement '{motor}' motor to {args.port}, "
        "then press Enter (power cycle the arm if the bus was shared)."
    )
    input()

    try:
        bus.setup_motor(
            motor,
            initial_baudrate=args.initial_baudrate,
            initial_id=args.initial_id,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if bus.is_connected:
            bus.port_handler.closePort()
        return 1

    if bus.is_connected:
        bus.port_handler.closePort()

    print(
        f"Done: '{motor}' should now be ID {target_id} at the bus default baud rate. "
        "Rewire the full chain, then verify with lerobot-find-port or your usual teleop."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
