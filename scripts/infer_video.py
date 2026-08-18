from __future__ import annotations

import argparse
import json
from pathlib import Path

from mp_sam2.video import segment_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MP-SAM2 memory-based video inference.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, help="Optional short-run limit for validation.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = segment_video(
        args.video,
        args.output,
        args.base_checkpoint,
        args.checkpoint,
        contrast=args.contrast,
        max_frames=args.max_frames,
        device=args.device,
    )
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
