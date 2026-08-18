from __future__ import annotations

from typing import Dict

import numpy as np


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute image-level binary segmentation metrics."""
    pred = np.asarray(prediction).astype(bool)
    gt = np.asarray(target).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} != {gt.shape}")

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "accuracy": float((tp + tn) / (tp + fp + fn + tn + eps)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall + eps)),
        "iou": float(tp / (tp + fp + fn + eps)),
        "dice": float(2 * tp / (2 * tp + fp + fn + eps)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

