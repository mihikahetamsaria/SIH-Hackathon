"""
Standalone HMR2.0 inference on a video, without detectron2/PHALP.
Uses YOLOv8 (ultralytics) for person detection instead of detectron2's ViTDet,
then feeds boxes into HMR2.0's own ViTDetDataset + model.

Run from inside the 4D-Humans repo root, with the '4D-humans' conda env active:
    python run_hmr2_on_video.py --video /path/to/your_video.mp4 --out_dir output/

Requires: torch, torchvision, ultralytics, opencv-python, and the hmr2 package
installed via `pip install -e . --no-deps` from the 4D-Humans repo.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from pathlib import Path
from hmr2.configs import get_config
from hmr2.models import HMR2, check_smpl_exists, DEFAULT_CHECKPOINT
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import cam_crop_to_full


def load_hmr2_no_renderer(checkpoint_path=DEFAULT_CHECKPOINT):
    """
    Reimplementation of hmr2.models.load_hmr2(), with init_renderer=False added.
    This skips construction of SkeletonRenderer/MeshRenderer entirely, which on
    macOS crash at construction time (pyrender's OffscreenRenderer only supports
    the 'egl' or 'osmesa' platforms, neither of which exist on macOS).
    We don't need rendering for this script - only the SMPL parameter outputs.
    """
    model_cfg_path = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, \
            f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    check_smpl_exists()
    model = HMR2.load_from_checkpoint(
        checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False
    )
    return model, model_cfg


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_frames(video_path, out_dir, every_n=1):
    """Extract frames from video to out_dir. Returns sorted list of frame paths."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_paths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            fname = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(fname, frame)
            saved_paths.append(fname)
        frame_idx += 1
    cap.release()
    print(f"Extracted {len(saved_paths)} frames to {out_dir}")
    return saved_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out_dir", default="output", help="Where to save frames + results")
    parser.add_argument("--every_n", type=int, default=1, help="Process every Nth frame (use >1 to subsample)")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="YOLO person detection confidence threshold")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    frames_dir = os.path.join(args.out_dir, "frames")
    results_dir = os.path.join(args.out_dir, "smpl_params")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Extract frames from the video
    frame_paths = extract_frames(args.video, frames_dir, every_n=args.every_n)

    # 2. Load person detector (replaces detectron2/ViTDet)
    print("Loading YOLO detector...")
    det_model = YOLO("yolov8n.pt")  # downloads automatically on first run

    # 3. Load HMR2.0 model + config
    print("Loading HMR2.0 model...")
    model, model_cfg = load_hmr2_no_renderer(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    # 4. Loop over frames: detect people, run HMR2.0, save params
    for frame_path in frame_paths:
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]
        img_cv2 = cv2.imread(frame_path)
        if img_cv2 is None:
            print(f"Warning: could not read {frame_path}, skipping")
            continue

        # Detect people with YOLO (class 0 = person in COCO)
        det_results = det_model(img_cv2, classes=[0], conf=args.conf_thresh, verbose=False)
        boxes = det_results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            print(f"{frame_name}: no person detected, skipping")
            continue

        # Feed boxes into HMR2.0's own dataset class
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        all_outputs = []
        for batch in dataloader:
            batch = {
                k: (v.to(device, dtype=torch.float32) if v.dtype == torch.float64 else v.to(device))
                   if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            with torch.no_grad():
                out = model(batch)

            # Extract SMPL params (pose, shape, camera) as numpy, move to CPU
            pred_smpl_params = {
                k: v.detach().cpu().numpy() for k, v in out["pred_smpl_params"].items()
            }
            pred_cam = out["pred_cam"].detach().cpu().numpy()

            # Convert crop-space camera params to full-frame camera translation.
            # Needed later to project the 3D mesh back onto the original (uncropped) frame.
            # Positional args here match the original 4D-Humans demo.py's call pattern exactly.
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size_t = batch["img_size"].float()
            scaled_focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size_t.max()
            )
            pred_cam_t_full = cam_crop_to_full(
                out["pred_cam"], box_center, box_size, img_size_t, scaled_focal_length
            ).detach().cpu().numpy()

            all_outputs.append({
                "global_orient": pred_smpl_params.get("global_orient"),
                "body_pose": pred_smpl_params.get("body_pose"),
                "betas": pred_smpl_params.get("betas"),
                "pred_cam": pred_cam,
                "pred_cam_t_full": pred_cam_t_full,
                "focal_length": scaled_focal_length.detach().cpu().numpy(),
                "img_size": img_size_t.detach().cpu().numpy(),
                "boxes": boxes,
            })

        # Save per-frame results (one npz per frame, containing all detected people)
        out_path = os.path.join(results_dir, f"{frame_name}.npz")
        np.savez(
            out_path,
            global_orient=np.concatenate([o["global_orient"] for o in all_outputs], axis=0),
            body_pose=np.concatenate([o["body_pose"] for o in all_outputs], axis=0),
            betas=np.concatenate([o["betas"] for o in all_outputs], axis=0),
            pred_cam=np.concatenate([o["pred_cam"] for o in all_outputs], axis=0),
            pred_cam_t_full=np.concatenate([o["pred_cam_t_full"] for o in all_outputs], axis=0),
            focal_length=all_outputs[0]["focal_length"],
            img_size=all_outputs[0]["img_size"],
            boxes=boxes,
        )
        print(f"{frame_name}: {len(boxes)} person(s) -> saved {out_path}")

    print(f"\nDone. Per-frame SMPL parameters saved in: {results_dir}")


if __name__ == "__main__":
    main()