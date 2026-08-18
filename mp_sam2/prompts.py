from __future__ import annotations

import dataclasses
import random
from contextlib import nullcontext
from typing import Any

import cv2
import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class PromptConfig:
    threshold_offset: int = 0
    box_expand_ratio: float = 0.10
    num_positive: int = 4
    num_negative: int = 2
    min_negative_distance: float = 6.0
    min_gap_ratio: float = 0.03
    hybrid_rerank: bool = True
    score_weight: float = 0.5
    dynamic_mask_only: bool = True
    prediction_threshold: float = 0.5


def threshold_prior(image_rgb: np.ndarray, offset: int = 0) -> tuple[np.ndarray, int]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    applied = int(np.clip(round(float(otsu)) + offset, 0, 255))
    _, mask = cv2.threshold(gray, applied, 1, cv2.THRESH_BINARY)
    if mask.mean() > 0.5:
        mask = 1 - mask
    return mask.astype(np.uint8), applied


def _quality(mask: np.ndarray) -> bool:
    foreground = int(mask.sum())
    if not foreground:
        return False
    count, _ = cv2.connectedComponents(mask.astype(np.uint8))
    ys, xs = np.where(mask > 0)
    bbox_area = max((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1), 1)
    ratio = foreground / mask.size
    fill = foreground / bbox_area
    return bool(0.0005 <= ratio <= 0.25 and count - 1 <= 5 and fill >= 0.25)


def _spaced_maxima(score: np.ndarray, support: np.ndarray, count: int, gap: int) -> list[tuple[int, int]]:
    height, width = support.shape
    output: list[tuple[int, int]] = []
    for flat in np.argsort(score.ravel())[::-1]:
        y, x = divmod(int(flat), width)
        if support[y, x] == 0 or score[y, x] <= 0:
            continue
        if all(abs(x - px) + abs(y - py) >= gap for px, py in output):
            output.append((x, y))
        if len(output) >= count:
            break
    return output


def construct_prompts(prior: np.ndarray, config: PromptConfig) -> dict[str, Any]:
    height, width = prior.shape
    mask_input = cv2.resize(prior.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)[None]
    box = None
    coords = None
    labels = None
    ys, xs = np.where(prior > 0)
    use_structured = ys.size > 0 and not (config.dynamic_mask_only and _quality(prior))
    if use_structured:
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        ex = int((x1 - x0 + 1) * config.box_expand_ratio)
        ey = int((y1 - y0 + 1) * config.box_expand_ratio)
        box = np.asarray([max(0, x0 - ex), max(0, y0 - ey), min(width - 1, x1 + ex), min(height - 1, y1 + ey)], dtype=np.float32)
        distance = cv2.distanceTransform(prior, cv2.DIST_L2, 5)
        gap = max(2, int(config.min_gap_ratio * max(height, width)))
        positive = _spaced_maxima(distance, prior, config.num_positive, gap)
        background = (prior == 0).astype(np.uint8)
        background_distance = cv2.distanceTransform(background, cv2.DIST_L2, 5)
        negative_yx = np.column_stack(np.where(background_distance >= config.min_negative_distance))
        if not len(negative_yx):
            negative_yx = np.column_stack(np.where(background > 0))
        negative: list[tuple[int, int]] = []
        if len(negative_yx):
            step = max(1, len(negative_yx) // max(config.num_negative, 1))
            negative = [(int(x), int(y)) for y, x in negative_yx[::step][: config.num_negative]]
        points = positive + negative
        if points:
            coords = np.asarray(points, dtype=np.float32)
            labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32)
    return {"mask_input": mask_input, "box": box, "point_coords": coords, "point_labels": labels}


def segment_image(predictor, image_rgb: np.ndarray, config: PromptConfig | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    config = config or PromptConfig()
    prior, threshold = threshold_prior(image_rgb, config.threshold_offset)
    prompts = construct_prompts(prior, config)
    with torch.no_grad():
        predictor.set_image(image_rgb)
    device = next(predictor.model.parameters()).device
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    only_mask = prompts["box"] is None and prompts["point_coords"] is None
    with context, torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=prompts["point_coords"],
            point_labels=prompts["point_labels"],
            box=prompts["box"],
            mask_input=prompts["mask_input"],
            multimask_output=not only_mask,
        )
    candidates = masks if masks.ndim == 3 else masks[None]
    scores = np.asarray(scores).reshape(-1)
    best = int(np.argmax(scores))
    if config.hybrid_rerank and prior.any() and len(candidates) > 1:
        prompt_iou = []
        for candidate in candidates:
            binary = candidate > config.prediction_threshold
            union = np.logical_or(binary, prior).sum()
            prompt_iou.append(np.logical_and(binary, prior).sum() / union if union else 0.0)
        combined = config.score_weight * scores + (1 - config.score_weight) * np.asarray(prompt_iou)
        best = int(np.argmax(combined))
    prediction = (candidates[best] > config.prediction_threshold).astype(np.uint8)
    return prediction, {
        "threshold": threshold,
        "score": float(scores[best]),
        "used_box": prompts["box"] is not None,
        "used_points": prompts["point_coords"] is not None,
        "point_count": 0 if prompts["point_coords"] is None else len(prompts["point_coords"]),
    }

