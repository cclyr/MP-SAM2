from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mp_sam2.modeling import load_image_predictor
from mp_sam2.prompts import load_prompt_config, segment_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic MP-SAM2 image inference.")
    parser.add_argument("--input", type=Path, required=True, help="An image or directory of images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "prompt.yaml")
    parser.add_argument("--model-config", default="sam2_hiera_l.yaml")
    parser.add_argument("--threshold-offset", type=int)
    parser.add_argument("--box-expand-ratio", type=float)
    parser.add_argument("--positive-points", type=int)
    parser.add_argument("--negative-points", type=int)
    parser.add_argument("--minimum-negative-distance", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if args.input.is_file():
        inputs = [(args.input, Path(args.input.name))]
    elif args.input.is_dir():
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Use a separate output directory for directory inference")
        output_is_nested = args.output.resolve().is_relative_to(args.input.resolve())
        inputs = [
            (path, path.relative_to(args.input))
            for path in sorted(args.input.rglob("*"))
            if path.is_file() and path.suffix.lower() in extensions
            and not (output_is_nested and path.resolve().is_relative_to(args.output.resolve()))
        ]
    else:
        raise FileNotFoundError(args.input)
    if not inputs:
        raise RuntimeError(f"No input images found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    predictor = load_image_predictor(args.checkpoint, model_config=args.model_config, device=args.device)
    config = load_prompt_config(
        args.config,
        threshold_offset=args.threshold_offset,
        box_expand_ratio=args.box_expand_ratio,
        num_positive=args.positive_points,
        num_negative=args.negative_points,
        min_negative_distance=args.minimum_negative_distance,
    )
    rows = []
    for path, relative in inputs:
        destination = args.output / relative.parent
        destination.mkdir(parents=True, exist_ok=True)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Skipping unreadable image: {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask, metadata = segment_image(predictor, rgb, config)
        cv2.imwrite(str(destination / f"{path.stem}_mask.png"), mask * 255)
        overlay = bgr.copy()
        color = np.zeros_like(bgr)
        color[..., 2] = mask * 255
        active = mask.astype(bool)
        overlay[active] = (0.55 * bgr[active] + 0.45 * color[active]).astype(np.uint8)
        cv2.imwrite(str(destination / f"{path.stem}_overlay.png"), overlay)
        rows.append({"image": relative.as_posix(), **metadata})
    (args.output / "inference.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Processed {len(rows)} image(s); outputs: {args.output}")


if __name__ == "__main__":
    main()

