from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mp_sam2.modeling import load_image_predictor
from mp_sam2.prompts import PromptConfig, segment_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic MP-SAM2 image inference.")
    parser.add_argument("--input", type=Path, required=True, help="An image or directory of images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold-offset", type=int, default=0)
    parser.add_argument("--box-expand-ratio", type=float, default=0.10)
    parser.add_argument("--positive-points", type=int, default=4)
    parser.add_argument("--negative-points", type=int, default=2)
    parser.add_argument("--minimum-negative-distance", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    inputs = [args.input] if args.input.is_file() else [p for p in sorted(args.input.iterdir()) if p.suffix.lower() in extensions]
    predictor = load_image_predictor(args.checkpoint, device=args.device)
    config = PromptConfig(
        threshold_offset=args.threshold_offset,
        box_expand_ratio=args.box_expand_ratio,
        num_positive=args.positive_points,
        num_negative=args.negative_points,
        min_negative_distance=args.minimum_negative_distance,
    )
    rows = []
    for path in inputs:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Skipping unreadable image: {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask, metadata = segment_image(predictor, rgb, config)
        cv2.imwrite(str(args.output / f"{path.stem}_mask.png"), mask * 255)
        overlay = bgr.copy()
        color = np.zeros_like(bgr)
        color[..., 2] = mask * 255
        active = mask.astype(bool)
        overlay[active] = (0.55 * bgr[active] + 0.45 * color[active]).astype(np.uint8)
        cv2.imwrite(str(args.output / f"{path.stem}_overlay.png"), overlay)
        rows.append({"image": path.name, **metadata})
    (args.output / "inference.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Processed {len(rows)} image(s); outputs: {args.output}")


if __name__ == "__main__":
    main()

