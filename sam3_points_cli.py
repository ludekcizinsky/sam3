#!/usr/bin/env python
"""SAM 3 video segmentation with SAM2-style point prompts.

Reads a JSON file with absolute (pixel) points/labels, converts to relative
coordinates, runs SAM3 tracking, and writes per-object mask PNGs.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 with point prompts from JSON.")
    parser.add_argument("--frames-dir", required=True, help="Directory with input frames.")
    parser.add_argument("--prompts-json", required=True, help="JSON file with prompts.")
    parser.add_argument("--output-dir", required=True, help="Directory to save masks.")
    return parser.parse_args()


def _load_first_frame_size(frames_dir: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths = sorted(frames_dir.glob(ext))
        if paths:
            with Image.open(paths[0]) as img:
                width, height = img.size
            return width, height
    raise RuntimeError(f"No frames found in {frames_dir}")


def _load_frames(frames_dir: Path):
    frames = []
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_paths.extend(frames_dir.glob(ext))
    if not img_paths:
        raise RuntimeError(f"No image frames found in {frames_dir}")
    try:
        img_paths.sort(key=lambda p: int(p.stem))
    except Exception:
        img_paths.sort()
    for p in img_paths:
        frames.append(np.array(Image.open(p).convert("RGB")))
    return frames


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    if frame.max() <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _apply_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    frame_uint8 = _to_uint8(frame)
    mask_bool = mask.astype(bool)
    masked_frame = np.zeros_like(frame_uint8)
    masked_frame[mask_bool] = frame_uint8[mask_bool]
    return masked_frame


def save_masks(outputs_per_frame, frames, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masked_dir = output_dir / "masked_images"
    masks_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)
    union_masks_dir = masks_dir / "union"
    union_masked_dir = masked_dir / "union"
    union_masks_dir.mkdir(parents=True, exist_ok=True)
    union_masked_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, masks_for_frame in outputs_per_frame.items():
        if not masks_for_frame:
            continue

        frame = frames[frame_idx]
        frame_h, frame_w = frame.shape[:2]
        combined_mask = None
        for obj_id, mask in masks_for_frame.items():
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.asarray(mask)

            if mask_np.shape != (frame_h, frame_w):
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8),
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST,
                )

            obj_dir = masks_dir / f"{int(obj_id):02d}"
            obj_dir.mkdir(parents=True, exist_ok=True)

            binary = (mask_np > 0).astype(np.uint8)
            Image.fromarray(binary * 255).save(obj_dir / f"{frame_idx:04d}.png")

            obj_masked_dir = masked_dir / f"{int(obj_id):02d}"
            obj_masked_dir.mkdir(parents=True, exist_ok=True)
            masked_frame = _apply_mask(frame, binary)
            Image.fromarray(masked_frame).save(obj_masked_dir / f"{frame_idx:04d}.png")

            if combined_mask is None:
                combined_mask = np.zeros_like(binary, dtype=np.uint8)
            combined_mask |= binary

        if combined_mask is not None:
            Image.fromarray(combined_mask * 255).save(
                union_masks_dir / f"{frame_idx:04d}.png"
            )
            union_masked = _apply_mask(frame, combined_mask)
            Image.fromarray(union_masked).save(
                union_masked_dir / f"{frame_idx:04d}.png"
            )


def _collect_outputs(predictor, inference_state, prompt_frame):
    outputs_per_frame = {}

    for frame_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
        inference_state=inference_state,
        start_frame_idx=prompt_frame,
        max_frame_num_to_track=None,
        reverse=False,
        propagate_preflight=True,
    ):
        frame_out = {}
        if video_res_masks is not None:
            for i, obj_id in enumerate(obj_ids):
                frame_out[int(obj_id)] = video_res_masks[i, 0].detach().cpu().numpy()
        outputs_per_frame[frame_idx] = frame_out

    for frame_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
        inference_state=inference_state,
        start_frame_idx=prompt_frame,
        max_frame_num_to_track=None,
        reverse=True,
        propagate_preflight=False,
    ):
        if frame_idx in outputs_per_frame:
            continue
        frame_out = {}
        if video_res_masks is not None:
            for i, obj_id in enumerate(obj_ids):
                frame_out[int(obj_id)] = video_res_masks[i, 0].detach().cpu().numpy()
        outputs_per_frame[frame_idx] = frame_out

    return outputs_per_frame


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    prompts_path = Path(args.prompts_json)
    output_dir = Path(args.output_dir)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    prompt_frame = int(prompts.get("prompt_frame", 0))
    objects = prompts.get("objects", [])

    if "frame_size" in prompts:
        width, height = prompts["frame_size"]
    else:
        width, height = _load_first_frame_size(frames_dir)

    frames = _load_frames(frames_dir)

    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    inference_state = predictor.init_state(
        video_path=str(frames_dir),
        async_loading_frames=False,
        offload_video_to_cpu=True,
    )

    for obj in objects:
        obj_id = int(obj["id"])
        points = obj.get("points", [])
        labels = obj.get("labels", [])
        box = obj.get("box", None)
        if not points and box is None:
            continue

        rel_points = None
        rel_labels = None
        if points:
            rel_points = [[p[0] / width, p[1] / height] for p in points]
            rel_labels = labels

        rel_box = None
        if box is not None:
            rel_box = np.array(
                [
                    box[0] / width,
                    box[1] / height,
                    box[2] / width,
                    box[3] / height,
                ],
                dtype=np.float32,
            )

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=prompt_frame,
            obj_id=obj_id,
            points=rel_points,
            labels=rel_labels,
            box=rel_box,
            clear_old_points=True,
            rel_coordinates=True,
        )

    outputs_per_frame = _collect_outputs(predictor, inference_state, prompt_frame)
    save_masks(outputs_per_frame, frames, output_dir)
    print(f"Saved masks to {output_dir}")


if __name__ == "__main__":
    main()
