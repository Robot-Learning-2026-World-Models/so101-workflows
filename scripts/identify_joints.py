#!/usr/bin/env python3
"""
Walk SO-100 / SO-101 joints one at a time so you can see which physical servo is which.

Feetech STS3215 does not expose a simple "blink the status LED now" register in the
control table that LeRobot uses. The onboard LED is mainly tied to alarm conditions
(EEPROM ``LED_Alarm_Condition``), not a harmless RAM toggle. This script instead:

1. For each joint in order: disables torque on every other motor, then on the
   selected motor rapidly toggles torque OFF/ON a few times (some boards show a
   visible change on the LED or you hear/feel a click).
2. Optionally adds a very small raw goal-position nudge (encoder ticks) so the
   selected joint moves slightly while the others stay limp.

Usage::

    uv run python scripts/identify_joints.py --port /dev/tty.usbmodemXXXX

    # 5-DOF arm (no gripper in software), same order as ``so_follower`` without gripper::
    uv run python scripts/identify_joints.py --port /dev/tty.usbmodemXXXX --no-gripper

Power the arm, keep it in a safe pose, and stay clear of pinch points. Press Enter
to advance to the next joint (or use ``--auto`` for timed steps).
"""

from __future__ import annotations

import argparse
import sys
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


def _so_motors(include_gripper: bool) -> dict[str, Motor]:
    norm_body = MotorNormMode.DEGREES
    motors: dict[str, Motor] = {
        "shoulder_pan": Motor(1, "sts3215", norm_body),
        "shoulder_lift": Motor(2, "sts3215", norm_body),
        "elbow_flex": Motor(3, "sts3215", norm_body),
        "wrist_flex": Motor(4, "sts3215", norm_body),
        "wrist_roll": Motor(5, "sts3215", norm_body),
    }
    if include_gripper:
        motors["gripper"] = Motor(6, "sts3215", MotorNormMode.RANGE_0_100)
    return motors


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="Serial device for this arm (follower or leader port).")
    p.add_argument(
        "--no-gripper",
        action="store_true",
        help="Omit ID 6 / gripper (matches a 5-motor ``motors={{...}}`` map).",
    )
    p.add_argument(
        "--torque-pulses",
        type=int,
        default=6,
        help="Torque OFF/ON cycles on the selected motor only (default: 6).",
    )
    p.add_argument(
        "--pulse-period",
        type=float,
        default=0.07,
        help="Seconds between torque edges during a pulse (default: 0.07).",
    )
    p.add_argument(
        "--nudge-ticks",
        type=int,
        default=24,
        help="Raw encoder ticks to nudge Goal_Position after pulsing; 0 disables (default: 24).",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="Advance to the next joint automatically after --step-s (no Enter).",
    )
    p.add_argument(
        "--step-s",
        type=float,
        default=3.0,
        help="Seconds per joint when --auto is set (default: 3).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    motors = _so_motors(include_gripper=not args.no_gripper)
    joint_order = list(motors.keys())

    bus = FeetechMotorsBus(port=args.port, motors=motors, calibration=None)

    print(
        "\nConnecting and checking every configured motor ID on the bus.\n"
        "Keep hands clear; torque will be toggled on ONE joint at a time.\n"
    )

    try:
        bus.connect(handshake=True)
        for m in bus.motors:
            bus.write("Operating_Mode", m, OperatingMode.POSITION.value)

        for idx, name in enumerate(joint_order):
            mid = bus.motors[name].id
            print("\n" + "=" * 60)
            print(f"  SELECTED JOINT #{idx + 1}/{len(joint_order)}: {name!r}  (bus ID {mid})")
            print("=" * 60)

            for other in bus.motors:
                bus.disable_torque(other)

            bus.enable_torque(name)

            for _ in range(max(0, args.torque_pulses)):
                bus.disable_torque(name)
                time.sleep(args.pulse_period)
                bus.enable_torque(name)
                time.sleep(args.pulse_period)

            if args.nudge_ticks > 0:
                pos = int(bus.read("Present_Position", name, normalize=False))
                delta = args.nudge_ticks
                bus.write("Goal_Position", name, pos + delta, normalize=False)
                time.sleep(0.25)
                bus.write("Goal_Position", name, pos, normalize=False)
                time.sleep(0.2)

            bus.disable_torque(name)

            if args.auto:
                if idx < len(joint_order) - 1:
                    print(f"\n(auto-advance in {args.step_s:.1f} s…)\n")
                    time.sleep(args.step_s)
            else:
                if idx < len(joint_order) - 1:
                    input("\nPress Enter for the next joint… ")

        print("\nDone. Re-enable torque in your teleop app or power-cycle if needed.\n")
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        if bus.is_connected:
            try:
                bus.disable_torque()
            except Exception:
                pass
            bus.port_handler.closePort()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
