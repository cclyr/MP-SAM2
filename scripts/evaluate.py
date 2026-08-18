from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from mp_sam2.metrics import segmentation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate binary PNG predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--mask-suffix", default="_enhanced")
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for prediction_path in sorted(args.predictions.glob("*.png")):
        stem = prediction_path.stem.removesuffix("_mask")
        candidates = [args.masks / f"{stem}{args.mask_suffix}.png", args.masks / f"{stem}.png"]
        mask_path = next((path for path in candidates if path.exists()), None)
        if mask_path is None:
            continue
        prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if target.shape != prediction.shape:
            target = cv2.resize(target, (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_NEAREST)
        rows.append({"image": stem, **segmentation_metrics(prediction > 0, target > 0)})
    if not rows:
        raise RuntimeError("No prediction/ground-truth pairs were found.")
    keys = ("accuracy", "precision", "recall", "f1", "iou", "dice")
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["image_count"] = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

