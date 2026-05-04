import cv2
import pybooklid
import argparse
import os
import time
from pathlib import Path
from playsound3 import playsound

angle_tol = 5e-2

resolution_overlays = [
    (256, 256),
    (512, 512),
    (640, 640),
    (960, 540),
    (1280, 720)
]

OVERLAY_COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (0, 165, 255),
    (0, 255, 0), 
    (255, 0, 0), 
]


def draw_overlays(frame):
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
        cv2.putText(frame, label, (x2 - tw - 4, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    try:
        angle = pybooklid.read_lid_angle()
        angle_label = f"lid: {angle:.3f}"
    except Exception:
        angle_label = "lid: N/A"
    cv2.putText(frame, angle_label, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3, cv2.LINE_AA)


def main(args):
    if (tgt := args.mac_lid_angle_target) is not None:
        while True:
            current_angle = pybooklid.read_lid_angle()
            if abs(current_angle - tgt) < angle_tol:
                print(f"Target angle of {tgt} reached!")
                playsound("assets/gong.mp3")
                break
            print(f"{current_angle=}")
            time.sleep(0.02)
        print(f"Current angle: {pybooklid.read_lid_angle()}")

    save_dir = Path("cam_images")
    save_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(args.camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_idx}")

    print("Press 's' to save clean frame, 'o' to save frame with overlays, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        display = frame.copy()
        draw_overlays(display)
        cv2.imshow("cam_preview", display)
        key = cv2.waitKey(1) & 0xFF
        ts = int(time.time())
        if key == ord('s'):
            path = save_dir / f"{ts}_clean.png"
            cv2.imwrite(str(path), frame)
            print(f"Saved clean frame: {path}")
        elif key == ord('o'):
            overlay_frame = frame.copy()
            draw_overlays(overlay_frame)
            path = save_dir / f"{ts}_overlay.png"
            cv2.imwrite(str(path), overlay_frame)
            print(f"Saved overlay frame: {path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="World's best camera setup script (with gongs)", description="Use this script to position the camera, grab calibration images, and see different potential crops")
    parser.add_argument("-a", "--mac-lid-angle-target", type=int, default=None, required=False)
    parser.add_argument("-c", "--camera-idx", type=int, default=0, required=False)
    args = parser.parse_args()
    main(args)
