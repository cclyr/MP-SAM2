from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data import CrackDataset, Sample


class HybridLoss(nn.Module):
    """BCE + 2*Dice + Focal + 1.5*IoU, as used in MP-SAM2."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        probability = torch.sigmoid(logits)
        flat_p = probability.flatten()
        flat_y = target.flatten()
        smooth = 1e-6
        intersection = (flat_p * flat_y).sum()
        dice = 1 - (2 * intersection + smooth) / (flat_p.sum() + flat_y.sum() + smooth)
        union = flat_p.sum() + flat_y.sum() - intersection
        iou = 1 - (intersection + smooth) / (union + smooth)
        bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        focal = ((1 - torch.exp(-bce_map)) ** 2 * bce_map).mean()
        bce = bce_map.mean()
        total = bce + 2 * dice + focal + 1.5 * iou
        return total, {
            "total": float(total.detach()),
            "bce": float(bce.detach()),
            "dice": float(dice.detach()),
            "focal": float(focal.detach()),
            "iou": float(iou.detach()),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_stage(model: nn.Module, stage: str) -> int:
    if stage == "decoder_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "image_encoder" not in name
    elif stage == "partial_encoder":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                "image_encoder" not in name
                or any(token in name for token in ("blocks.30", "blocks.31", "neck"))
            )
    elif stage == "full_finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def split_steps(sample_count: int, batch_size: int, epochs: int) -> list[int]:
    total = max(3, math.ceil(sample_count / batch_size) * epochs)
    first = max(1, round(total * 0.30))
    second = max(1, round(total * 0.40))
    return [first, second, total - first - second]


def _load_base(checkpoint: Path, model_config: str, device: str) -> SAM2ImagePredictor:
    model = build_sam2(model_config, ckpt_path=None, device=device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Base checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    return SAM2ImagePredictor(model)


def _forward_embeddings(predictor: SAM2ImagePredictor, images: list[np.ndarray], train_encoder: bool) -> None:
    if train_encoder and hasattr(predictor.set_image_batch, "__wrapped__"):
        # SAM2's public predictor marks image embedding as no-grad for inference.
        # Calling the wrapped function enables the intended progressive encoder update.
        predictor.set_image_batch.__wrapped__(predictor, images)
    else:
        predictor.set_image_batch(images)


def train(
    samples: Sequence[Sample],
    base_checkpoint: str | Path,
    output_checkpoint: str | Path,
    *,
    model_config: str = "sam2_hiera_l.yaml",
    seed: int = 42,
    epochs: int = 80,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    stage_steps: Sequence[int] | None = None,
    device: str = "cuda",
) -> dict:
    """Run the three-stage progressive fine-tuning protocol."""
    set_seed(seed)
    base_checkpoint = Path(base_checkpoint)
    output_checkpoint = Path(output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    predictor = _load_base(base_checkpoint, model_config, device)
    dataset = CrackDataset(samples, image_size=1024, points_per_sample=64, negative_ratio=0.10)
    optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    criterion = HybridLoss()
    stages = ("decoder_only", "partial_encoder", "full_finetune")
    steps = list(stage_steps) if stage_steps else split_steps(len(samples), batch_size, epochs)
    history = []
    started = time.time()

    for stage, count in zip(stages, steps):
        trainable = configure_stage(predictor.model, stage)
        predictor.model.train()
        last = {}
        stage_started = time.time()
        for _ in trange(int(count), desc=stage, unit="iter"):
            images, masks, point_coords, point_labels = dataset.read_batch(batch_size)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                _forward_embeddings(predictor, images, train_encoder=stage != "decoder_only")
                _, coords, labels, _ = predictor._prep_prompts(
                    point_coords, point_labels, box=None, mask_logits=None, normalize_coords=True
                )
                sparse, dense = predictor.model.sam_prompt_encoder(
                    points=(coords, labels), boxes=None, masks=None
                )
                high_resolution = predictor._features["high_res_feats"]
                low_resolution, _, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"],
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=high_resolution,
                )
                prediction = predictor._transforms.postprocess_masks(low_resolution, predictor._orig_hw[-1])[:, 0]
                target = torch.as_tensor(masks, dtype=torch.float32, device=device)
                loss, last = criterion(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        history.append({
            "stage": stage,
            "steps": int(count),
            "trainable_parameters": int(trainable),
            "duration_seconds": time.time() - stage_started,
            "last_loss": last,
        })

    metadata = {
        "architecture": "SAM2 Hiera Large",
        "seed": seed,
        "training_images": len(samples),
        "epochs_equivalent": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "stage_steps": dict(zip(stages, steps)),
        "loss_weights": {"bce": 1.0, "dice": 2.0, "focal": 1.0, "iou": 1.5},
        "history": history,
        "duration_seconds": time.time() - started,
    }
    torch.save({"model": predictor.model.state_dict(), "metadata": metadata}, output_checkpoint)
    output_checkpoint.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
