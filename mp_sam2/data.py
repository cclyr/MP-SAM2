from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Sample = Tuple[Path, Path]


def collect_samples(
    image_root: str | Path,
    mask_root: str | Path,
    categories: Iterable[str] | None = None,
) -> List[Sample]:
    """Collect image/mask pairs from ``root/category/name.png`` folders."""
    image_root = Path(image_root)
    mask_root = Path(mask_root)
    names = list(categories) if categories else sorted(p.name for p in image_root.iterdir() if p.is_dir())
    samples: List[Sample] = []
    for category in names:
        for image_path in sorted((image_root / category).glob("*.png")):
            mask_path = mask_root / category / f"{image_path.stem}_enhanced.png"
            if mask_path.exists():
                samples.append((image_path, mask_path))
    return samples


def nested_subset(samples: Sequence[Sample], budget: int, seed: int) -> List[Sample]:
    """Create the balanced, deterministic subset used by the few-shot experiments."""
    by_category: dict[str, List[Sample]] = {}
    for sample in samples:
        by_category.setdefault(sample[0].parent.name, []).append(sample)
    rng = random.Random(seed)
    for category_samples in by_category.values():
        category_samples.sort(key=lambda item: (item[0].parent.name, item[0].name))
        rng.shuffle(category_samples)
    categories = sorted(by_category)
    order: List[Sample] = []
    offsets = {name: 0 for name in categories}
    start = seed % max(len(categories), 1)
    while len(order) < len(samples):
        progress = False
        for delta in range(len(categories)):
            name = categories[(start + delta) % len(categories)]
            index = offsets[name]
            if index < len(by_category[name]):
                order.append(by_category[name][index])
                offsets[name] += 1
                progress = True
        if not progress:
            break
    if budget <= 0 or budget > len(order):
        raise ValueError(f"budget must be in [1, {len(order)}], got {budget}")
    return order[:budget]


class CrackDataset:
    """Random mini-batch loader matching the training protocol in the paper."""

    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int = 1024,
        points_per_sample: int = 64,
        negative_ratio: float = 0.10,
    ) -> None:
        if not samples:
            raise ValueError("At least one image/mask pair is required.")
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.points_per_sample = int(points_per_sample)
        self.negative_ratio = float(negative_ratio)
        self._cache: dict[Path, np.ndarray] = {}

    def _read(self, path: Path, flag: int) -> np.ndarray:
        value = self._cache.get(path)
        if value is None:
            value = cv2.imread(str(path), flag)
            if value is None:
                raise FileNotFoundError(path)
            self._cache[path] = value
        return value.copy()

    def _sample_one(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image_path, mask_path = self.samples[np.random.randint(len(self.samples))]
        image = cv2.cvtColor(self._read(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = self._read(mask_path, cv2.IMREAD_GRAYSCALE)
        scale = min(self.image_size / image.shape[1], self.image_size / image.shape[0])
        size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        mask = (cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        positive = np.argwhere(mask > 0)
        negative = np.argwhere(mask == 0)
        if not len(positive):
            return self._sample_one()
        count_negative = min(int(round(self.points_per_sample * self.negative_ratio)), len(negative))
        count_positive = max(1, self.points_per_sample - count_negative)
        positive_indices = np.random.choice(len(positive), count_positive, replace=len(positive) < count_positive)
        points = [positive[positive_indices][:, [1, 0]]]
        labels = [np.ones(count_positive, dtype=np.int32)]
        if count_negative:
            negative_indices = np.random.choice(len(negative), count_negative, replace=len(negative) < count_negative)
            points.append(negative[negative_indices][:, [1, 0]])
            labels.append(np.zeros(count_negative, dtype=np.int32))
        return image, mask, np.concatenate(points).astype(np.float32), np.concatenate(labels)

    def read_batch(self, batch_size: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        batch = [self._sample_one() for _ in range(batch_size)]
        images, masks, points, labels = zip(*batch)
        return list(images), np.stack(masks), np.stack(points), np.stack(labels)

