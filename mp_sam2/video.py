from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from .modeling import load_image_predictor, load_video_predictor
from .prompts import PromptConfig, segment_image


def _contrast(image: np.ndarray, factor: float) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=float(factor), beta=0)


def segment_video(
    video_path: str | Path,
    output_path: str | Path,
    base_checkpoint: str | Path,
    finetuned_checkpoint: str | Path,
    *,
    contrast: float = 1.0,
    max_frames: int | None = None,
    device: str = "cuda",
    prompt_config: PromptConfig | None = None,
    model_config: str = "sam2_hiera_l.yaml",
) -> dict:
    """Automatic first-frame MP-SAM2 initialization followed by SAM2 memory propagation."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mp_sam2_video_") as temporary:
        frame_root = Path(temporary)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frames: list[Path] = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = _contrast(frame, contrast)
            path = frame_root / f"{index:05d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            frames.append(path)
            index += 1
            if max_frames is not None and index >= max_frames:
                break
        capture.release()
        if not frames:
            raise RuntimeError("No video frames were decoded.")

        first_bgr = cv2.imread(str(frames[0]))
        first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
        image_predictor = load_image_predictor(finetuned_checkpoint, model_config=model_config, device=device)
        initial_mask, initial_metadata = segment_image(image_predictor, first_rgb, prompt_config)
        del image_predictor
        torch.cuda.empty_cache()

        video_predictor = load_video_predictor(
            base_checkpoint, finetuned_checkpoint, model_config=model_config, device=device
        )
        import sam2.modeling.sam.transformer as transformer

        transformer.USE_FLASH_ATTN = False
        transformer.MATH_KERNEL_ON = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        predictions: list[np.ndarray | None] = [None] * len(frames)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            state = video_predictor.init_state(
                video_path=str(frame_root),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )
            video_predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=initial_mask.astype(bool))
            for frame_index, _, logits in video_predictor.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=len(frames)
            ):
                predictions[frame_index] = (logits[0, 0].detach().float().cpu().numpy() > 0).astype(np.uint8)
            video_predictor.reset_state(state)
        if any(mask is None for mask in predictions):
            raise RuntimeError("SAM2 memory propagation did not return every frame.")

        height, width = first_bgr.shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video: {output_path}")
        mask_dir = output_path.parent / f"{output_path.stem}_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for index, (frame_path, mask) in enumerate(zip(frames, predictions)):
            frame = cv2.imread(str(frame_path))
            assert mask is not None
            overlay = frame.copy()
            color = np.zeros_like(frame)
            color[..., 2] = mask * 255
            active = mask.astype(bool)
            overlay[active] = (0.55 * frame[active] + 0.45 * color[active]).astype(np.uint8)
            writer.write(overlay)
            cv2.imwrite(str(mask_dir / f"{index:05d}.png"), mask * 255)
        writer.release()
        return {
            "frames": len(frames),
            "fps": float(fps),
            "contrast": float(contrast),
            "initial_prompt": initial_metadata,
            "output": str(output_path),
            "mask_directory": str(mask_dir),
        }
