from __future__ import annotations

import argparse
from pathlib import Path

from mp_sam2.config import load_config
from mp_sam2.data import collect_samples, nested_subset
from mp_sam2.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MP-SAM2 on paired crack images and masks.")
    parser.add_argument("--config", type=Path, help="Defaults to configs/full.yaml or configs/fewshot.yaml according to --budget.")
    parser.add_argument("--images", type=Path, required=True, help="Image root: images/category/name.png")
    parser.add_argument("--masks", type=Path, required=True, help="Mask root: masks/category/name_enhanced.png")
    parser.add_argument("--categories", nargs="+", default=["01", "02"])
    parser.add_argument("--budget", default="full", help="A positive few-shot image count, or full")
    parser.add_argument("--seed", type=int, default=42, help="Optional control of training randomness.")
    parser.add_argument("--epochs", type=int, help="Override epochs_equivalent in the few-shot configuration.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--model-config", help="Override the bundled SAM2 architecture configuration.")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--full-stage-steps",
        nargs=3,
        type=int,
        metavar=("STAGE1", "STAGE2", "STAGE3"),
        help="Use 3000 4000 3000 for the original full-data protocol.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_data = args.budget.lower() == "full"
    config_path = args.config or Path(__file__).resolve().parents[1] / "configs" / ("full.yaml" if full_data else "fewshot.yaml")
    settings = load_config(config_path, {
        "model", "input_size", "batch_size", "epochs_equivalent", "optimizer",
        "learning_rate", "weight_decay", "stage_ratio", "stage_steps",
        "partial_encoder_blocks", "partial_encoder_neck", "loss_weights",
        "training_points", "negative_point_ratio",
    })
    if settings.get("optimizer", "AdamW") != "AdamW":
        raise ValueError("The released trainer supports optimizer: AdamW")
    if not full_data and (args.full_stage_steps is not None or "stage_steps" in settings):
        raise ValueError("For a few-shot budget, use configs/fewshot.yaml with stage_ratio")
    if args.epochs is not None and full_data and (args.full_stage_steps is not None or "stage_steps" in settings):
        raise ValueError("The full-data schedule uses stage_steps; --epochs applies to the few-shot schedule")
    samples = collect_samples(args.images, args.masks, args.categories)
    if args.budget.lower() != "full":
        samples = nested_subset(samples, int(args.budget), args.seed)
    stage_steps = args.full_stage_steps if args.full_stage_steps is not None else settings.get("stage_steps")
    metadata = train(
        samples,
        args.base_checkpoint,
        args.output,
        seed=args.seed,
        model_config=args.model_config or settings.get("model", "sam2_hiera_l.yaml"),
        epochs=args.epochs if args.epochs is not None else settings.get("epochs_equivalent", 80),
        batch_size=args.batch_size if args.batch_size is not None else settings.get("batch_size", 2),
        learning_rate=args.learning_rate if args.learning_rate is not None else settings.get("learning_rate", 1e-5),
        weight_decay=args.weight_decay if args.weight_decay is not None else settings.get("weight_decay", 0.01),
        stage_steps=stage_steps,
        stage_ratio=settings.get("stage_ratio", (0.30, 0.40, 0.30)),
        input_size=settings.get("input_size", 1024),
        training_points=settings.get("training_points", 64),
        negative_point_ratio=settings.get("negative_point_ratio", 0.10),
        partial_encoder_blocks=settings.get("partial_encoder_blocks", (30, 31)),
        partial_encoder_neck=settings.get("partial_encoder_neck", True),
        loss_weights=settings.get("loss_weights"),
        device=args.device,
    )
    print(f"Saved checkpoint to {args.output}")
    print(metadata)


if __name__ == "__main__":
    main()

