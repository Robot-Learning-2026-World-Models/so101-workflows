"""
Camera preview and calibration capture.

  --mode opencv (default): raw preview; ENTER/s save, q / ESC quit (no overlays).
  --mode mac: resolution overlays + lid readout + optional -a gong; o saves with overlays burned in.

Preview keys (opencv): ENTER / s -> save, q / ESC -> quit
Preview keys (mac):    ENTER / s -> save clean, o -> save with overlays, q / ESC -> quit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

# ---- Config ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAVE_DIR = REPO_ROOT / "outputs/captured_images"  # <-- edit me
DEFAULT_PREFIX = "checkerboard"  # <-- edit me
# ---------------------------------------------------------------------------

angle_tol = 5e-2

resolution_overlays = [
    (256, 256),
    (512, 512),
    (640, 640),
    (960, 540),
    (1280, 720),
]

OVERLAY_COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (0, 165, 255),
    (0, 255, 0),
    (255, 0, 0),
]


def draw_mac_rects_and_lid(frame) -> None:
    """Resolution guides + lid angle (mac mode only)."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    for i, (rw, rh) in enumerate(resolution_overlays):
        if h < rh or w < rw:
            continue
        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        x1 = cx - rw // 2
        y1 = cy - rh // 2
        x2 = cx + rw // 2
        y2 = cy + rh // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{rw}x{rh}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            frame,
            label,
            (x2 - tw - 4, y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    try:
        import pybooklid

        angle = pybooklid.read_lid_angle()
        angle_label = f"lid: {angle:.3f}"
    except Exception:
        angle_label = "lid: N/A"
    cv2.putText(
        frame,
        angle_label,
        (16, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )


def draw_mac_preview(frame, idx: int) -> None:
    """Live HUD: rectangles, lid, and saved index."""
    draw_mac_rects_and_lid(frame)
    cv2.putText(
        frame,
        f"saved: {idx}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main(args: argparse.Namespace) -> None:
    if args.mac_lid_angle_target is not None and args.mode != "mac":
        raise SystemExit("--mac-lid-angle-target requires --mode mac")

    if args.mode == "mac" and args.mac_lid_angle_target is not None:
        import pybooklid
        from playsound3 import playsound

        tgt = args.mac_lid_angle_target
        while True:
            current_angle = pybooklid.read_lid_angle()
            if abs(current_angle - tgt) < angle_tol:
                print(f"Target angle of {tgt} reached!")
                playsound(str(REPO_ROOT / "assets" / "gong.mp3"))
                break
            print(f"{current_angle=}")
            time.sleep(0.02)
        print(f"Current angle: {pybooklid.read_lid_angle()}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    existing = list(save_dir.glob(f"{prefix}_*.png"))
    idx = len(existing)

    cap = cv2.VideoCapture(args.camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_idx}")

    win = "cam_preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"Camera index: {args.camera_idx}")
    print(f"Saving to: {save_dir.resolve()}")
    if args.mode == "mac":
        print(
            "ENTER / s = save clean, o = save with overlays, q / ESC = quit."
        )
    else:
        print("ENTER / s = save clean, q / ESC = quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            display = frame.copy()
            if args.mode == "mac":
                draw_mac_preview(display, idx)
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (13, 10, ord("s")):
                path = save_dir / f"{prefix}_{idx:02d}.png"
                cv2.imwrite(str(path), frame)
                print(f"saved {path.name}")
                idx += 1
            elif args.mode == "mac" and key == ord("o"):
                overlay_frame = frame.copy()
                draw_mac_rects_and_lid(overlay_frame)
                path = save_dir / f"{prefix}_{idx:02d}.png"
                cv2.imwrite(str(path), overlay_frame)
                print(f"saved (with overlays) {path.name}")
                idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="cam_preview",
        description="Preview camera; opencv = minimal UI, mac = crop/lid overlays.",
    )
    parser.add_argument(
        "--mode",
        choices=("opencv", "mac"),
        default="opencv",
        help="opencv: raw preview, no overlays. mac: crop overlays + lid (pybooklid).",
    )
    parser.add_argument(
        "-a",
        "--mac-lid-angle-target",
        type=int,
        default=None,
        help="Mac mode only: poll lid until this angle (then play gong).",
    )
    parser.add_argument(
        "-c",
        "--camera-idx",
        type=int,
        default=0,
        help="OpenCV camera index (same as cv2.VideoCapture index). Default: 0.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"Directory for PNGs (default: {DEFAULT_SAVE_DIR})",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help=f"Filename prefix: <prefix>_NN.png (default: {DEFAULT_PREFIX})",
    )
    args = parser.parse_args()
    main(args)
