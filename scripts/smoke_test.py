from __future__ import annotations

import numpy as np

from mp_sam2.metrics import segmentation_metrics
from mp_sam2.prompts import PromptConfig, construct_prompts


def main() -> None:
    prior = np.zeros((256, 256), dtype=np.uint8)
    prior[80:176, 126:130] = 1
    prompts = construct_prompts(prior, PromptConfig(dynamic_mask_only=False))
    assert prompts["mask_input"].shape == (1, 256, 256)
    assert prompts["box"].shape == (4,)
    assert prompts["point_coords"].shape[1] == 2
    metrics = segmentation_metrics(prior, prior)
    assert abs(metrics["iou"] - 1.0) < 1e-8
    assert abs(metrics["dice"] - 1.0) < 1e-8
    print("MP-SAM2 prompt and metric smoke test passed.")


if __name__ == "__main__":
    main()

