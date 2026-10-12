from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mss
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one template image from screen")
    parser.add_argument("--name", required=True, help="Template filename without extension")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index for mss")
    parser.add_argument("--x", type=int, help="Left x inside captured monitor")
    parser.add_argument("--y", type=int, help="Top y inside captured monitor")
    parser.add_argument("--w", type=int, default=80, help="Width of crop")
    parser.add_argument("--h", type=int, default=80, help="Height of crop")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    templates_dir = project_dir / "templates"
    debug_dir = project_dir / "debug_frames"
    templates_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    with mss.mss() as sct:
        monitor_index = max(1, min(args.monitor, len(sct.monitors) - 1))
        monitor = sct.monitors[monitor_index]
        shot = sct.grab(monitor)

    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
    fh, fw = frame.shape[:2]

    w = max(10, min(args.w, fw))
    h = max(10, min(args.h, fh))

    if args.x is None or args.y is None:
        x = max(0, (fw // 2) - (w // 2))
        y = max(0, (fh // 2) - (h // 2))
    else:
        x = max(0, min(args.x, fw - w))
        y = max(0, min(args.y, fh - h))

    crop = frame[y : y + h, x : x + w]

    out_template = templates_dir / f"{args.name}.png"
    cv2.imwrite(str(out_template), crop)

    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
    preview_path = debug_dir / "capture_preview.jpg"
    cv2.imwrite(str(preview_path), preview)

    print(f"Saved template: {out_template}")
    print(f"Saved preview: {preview_path}")


if __name__ == "__main__":
    main()
