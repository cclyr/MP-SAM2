from __future__ import annotations

from pathlib import Path

import torch

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _state_dict(path: str | Path, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    payload = torch.load(str(path), map_location=device, weights_only=False)
    return payload["model"] if isinstance(payload, dict) and "model" in payload else payload


def load_image_predictor(
    checkpoint: str | Path,
    model_config: str = "sam2_hiera_l.yaml",
    device: str = "cuda",
) -> SAM2ImagePredictor:
    model = build_sam2(model_config, ckpt_path=None, device=device)
    missing, unexpected = model.load_state_dict(_state_dict(checkpoint, device), strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing[:10]}")
    model.eval()
    return SAM2ImagePredictor(model)


def load_video_predictor(
    base_checkpoint: str | Path,
    finetuned_checkpoint: str | Path | None,
    model_config: str = "sam2_hiera_l.yaml",
    device: str = "cuda",
):
    predictor = build_sam2_video_predictor(
        model_config,
        ckpt_path=None,
        device=device,
        apply_postprocessing=False,
    )
    current = predictor.state_dict()
    base = _state_dict(base_checkpoint, device)
    compatible = {
        key: value
        for key, value in base.items()
        if key in current and current[key].shape == value.shape
    }
    predictor.load_state_dict(compatible, strict=False)
    if finetuned_checkpoint is not None:
        missing, unexpected = predictor.load_state_dict(
            _state_dict(finetuned_checkpoint, device), strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Fine-tuned checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
    predictor.eval()
    return predictor
