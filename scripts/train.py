from __future__ import annotations

import argparse
from pathlib import Path

from mp_sam2.data import collect_samples, nested_subset
from mp_sam2.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MP-SAM2 on paired crack images and masks.")
    parser.add_argument("--images", type=Path, required=True, help="Image root: images/category/name.png")
    parser.add_argument("--masks", type=Path, required=True, help="Mask root: masks/category/name_enhanced.png")
    parser.add_argument("--categories", nargs="+", default=["01", "02"])
    parser.add_argument("--budget", default="full", help="5, 10, 50, or full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
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
    samples = collect_samples(args.images, args.masks, args.categories)
    if args.budget.lower() != "full":
        samples = nested_subset(samples, int(args.budget), args.seed)
    stage_steps = args.full_stage_steps if args.budget.lower() == "full" else None
    metadata = train(
        samples,
        args.base_checkpoint,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stage_steps=stage_steps,
        device=args.device,
    )
    print(f"Saved checkpoint to {args.output}")
    print(metadata)


if __name__ == "__main__":
    main()

