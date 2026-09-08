"""
Astronaut experiment pipeline: HMR2 (4D-Humans) -> Gemini -> playback UI
==========================================================================

ONE script, run once, does everything in order:

  Phase 0  Extract frames from the video, burn in timestamps.
  Phase 1  Run YOLOv8 + HMR2 on every frame -> real 3D SMPL joints, decoded
           by NAME (elbows, knees, wrists, hands...) via the standard SMPL
           kinematic tree, reduced to a compact per-frame telemetry summary:
           joint angles, hand separation/reach, tilt, velocity, and a 2D
           pixel-space skeleton for drawing.
  Phase 2  Feed BOTH the frame images AND the pose telemetry into Gemini,
           chunk by chunk, so Gemini's judgments are grounded in real 3D
           geometry instead of guessed from pixels alone.
  Phase 3  Launch the Tk playback UI: video on the left with the HMR2
           skeleton drawn live on top of it (cyan normally, red when Gemini
           flags an error at that timestamp), telemetry log on the right.

WHERE TO RUN THIS
------------------
Run it from inside the '4D-Humans' repo root, with the '4D-humans' conda
env active. You also need the Gemini SDK in that SAME env:

    conda activate 4D-humans
    pip install google-genai pydantic --no-deps
    export GEMINI_API_KEY="your-key-here"      # (or GOOGLE_API_KEY)

USAGE
-----
    python astronaut_pipeline.py --video 1707_020_AR_EN.mp4 --out_dir output

Everything is cached: re-running the same command skips HMR2 inference
for frames already processed, skips Gemini calls if the final timeline
is already cached, and just reopens the UI.

NOTE ON JOINT DECODING
-----------------------
Earlier versions of this script pulled the 24 canonical SMPL body joints
via `model.smpl(...).joints[:, :24, :]`, trusting that HMR2's bundled SMPL
wrapper returns the standard 24 canonical joints first, in kinematic-tree
order. On this HMR2 build that assumption is wrong: the wrapper's `.joints`
tensor concatenates extra (OpenPose/H36M-style) keypoints from a second
regressor alongside the canonical 24, and a naive `[:, :24]` slice can grab
the wrong set of 3D points -- same shape, plausible-looking numbers, wrong
anatomical identities. Since SMPL_BONES / SMPL_JOINT_NAMES below assume
standard kinematic-tree indices, that mismatch silently wired bones between
unrelated points (visible as a scrambled, radiating skeleton overlay).

Fix: derive the 24 canonical joints ourselves from mesh VERTICES via the
model's own J_regressor matrix -- the actual matrix that *defines* what
"pelvis", "left_elbow", etc. mean in SMPL -- rather than trusting a
wrapper-bundled joints tensor. Vertices have no such ordering ambiguity
(this is why a separate vertex-based mesh overlay for this same data
rendered correctly), so regressing joints from verts guarantees correct
identity/order by construction. See decode_smpl_joints() below.
"""

import argparse
import json
import os
import pickle
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk

# --- macOS fix: prevent hmr2 from ever importing the real pyrender package ---
# hmr2/utils/renderer.py does `import pyrender` at module level, and pyrender's
# own __init__.py unconditionally probes for an EGL/OSMesa library at import
# time. Neither exists on macOS, so the import crashes before our code even
# runs -- even though we never construct a renderer (init_renderer=False).
# We never render anything in this script, so a permissive stub module that
# hands back a dummy class for any attribute is enough to satisfy every
# `import pyrender` / `from pyrender import X` in the hmr2 codebase.
import sys
import types as _types

if "pyrender" not in sys.modules:
    class _AutoStubModule(_types.ModuleType):
        def __getattr__(self, name):
            # Let dunder lookups (__file__, __spec__, __path__, ...) fail
            # normally -- Python's own `inspect` module walks every entry in
            # sys.modules during stack introspection (triggered later by an
            # unrelated torch/torchvision import) and calls
            # getattr(module, '__file__', None) on each one. If we returned a
            # fake class instead of raising here, that breaks far away in
            # code that assumes __file__ is a string or None.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return type(name, (), {})

    sys.modules["pyrender"] = _AutoStubModule("pyrender")

from ultralytics import YOLO

from hmr2.configs import get_config
from hmr2.models import HMR2, check_smpl_exists, DEFAULT_CHECKPOINT
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import cam_crop_to_full

import random
import re

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field


DEFAULT_PROTOCOL = """
1. Align the primary fluid coupling perfectly straight.
2. Secure using the specialized torque driver.
3. The astronaut must maintain three points of contact on the handrails.
"""

# ---------------------------------------------------------------------------
# Standard SMPL body-joint kinematic tree (fixed by the SMPL model definition
# itself -- NOT dependent on which HMR2 checkpoint you're using, so these
# names/indices/parents are safe to hardcode). This is the joint set that
# model.smpl.J_regressor is defined against, so as long as we regress joints
# from vertices via J_regressor (see decode_smpl_joints), these names/indices
# are guaranteed to line up correctly.
# ---------------------------------------------------------------------------
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]
J = {name: i for i, name in enumerate(SMPL_JOINT_NAMES)}

SMPL_BONES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23),
]


# ---------------------------------------------------------------------------
# Output schema for Gemini
# ---------------------------------------------------------------------------
class FrameAnalysis(BaseModel):
    timestamp_sec: float = Field(description="The exact timestamp in seconds read from the image frame.")
    is_error: bool = Field(description="True if protocol violated or poor microgravity stabilization.")
    hardware_interaction: str = Field(description="Specific hardware handled (e.g., fluid valves, clamps).")
    action_description: str = Field(description="Kinematic description of the current action.")
    correction: str = Field(description="If is_error is True, specify the mechanical correction required.")


class ExperimentTimeline(BaseModel):
    timeline: list[FrameAnalysis]


# ---------------------------------------------------------------------------
# Phase 0: frame extraction (with burned-in timestamps)
# ---------------------------------------------------------------------------
def extract_frames(video_path, out_dir, target_fps):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(original_fps / target_fps))

    frames_meta = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / original_fps
            cv2.putText(frame, f"T+{timestamp:.2f}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            fname = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(fname, frame)
            frames_meta.append({"path": fname, "timestamp": timestamp, "frame_idx": frame_idx})
        frame_idx += 1
    cap.release()
    print(f"[Phase 0] Extracted {len(frames_meta)} frames -> {out_dir}")
    return frames_meta, original_fps


# ---------------------------------------------------------------------------
# Phase 1: HMR2 inference -> named-joint pose telemetry + 2D skeleton
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_hmr2_no_renderer(checkpoint_path=DEFAULT_CHECKPOINT):
    model_cfg_path = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    check_smpl_exists()
    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False)
    return model, model_cfg


def decode_smpl_joints(model, pred_smpl_params):
    """Derive the 24 canonical SMPL joints from mesh VERTICES via the model's
    own J_regressor, rather than trusting model.smpl(...).joints directly.

    HMR2's internal SMPL wrapper concatenates extra (OpenPose/H36M-style)
    keypoints alongside the 24 canonical body joints in its `.joints` output,
    and a naive `[:, :24, :]` slice can silently grab the wrong set of 3D
    points -- same shape, plausible values, wrong anatomical identities. This
    scrambles bone connectivity downstream even though nothing raises an
    error. J_regressor is the actual matrix that *defines* the standard 24
    SMPL joints (pelvis, elbows, knees, ...) from the 6890 mesh vertices, so
    regressing joints ourselves guarantees the SMPL_JOINT_NAMES ordering
    above is correct by construction -- exactly the same vertices used by
    the (separately verified working) mesh overlay renderer.

    Returns (verts, joints24):
        verts    -- (N_people, 6890, 3) numpy, model/canonical space
        joints24 -- (N_people, 24, 3)   numpy, model/canonical space,
                    same coordinate space as verts (no pelvis recentering).
    """
    smpl_params = {k: v.float() for k, v in pred_smpl_params.items()}
    smpl_out = model.smpl(**smpl_params, pose2rot=False)
    verts = smpl_out.vertices  # (N, 6890, 3), torch, still on device

    J_regressor = model.smpl.J_regressor.to(verts.device).float()  # (24, 6890)
    joints24 = torch.einsum('jv,nvc->njc', J_regressor, verts)     # (N, 24, 3)

    return verts.detach().cpu().numpy(), joints24.detach().cpu().numpy()


def _angle_at(joints, a, b, c):
    """Angle in degrees at joint b, between vectors b->a and b->c.
    180 deg = fully straight limb, small angle = sharply bent."""
    v1, v2 = joints[a] - joints[b], joints[c] - joints[b]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def compute_limb_metrics(joints24):
    return {
        "left_elbow_deg": _angle_at(joints24, J["left_shoulder"], J["left_elbow"], J["left_wrist"]),
        "right_elbow_deg": _angle_at(joints24, J["right_shoulder"], J["right_elbow"], J["right_wrist"]),
        "left_knee_deg": _angle_at(joints24, J["left_hip"], J["left_knee"], J["left_ankle"]),
        "right_knee_deg": _angle_at(joints24, J["right_hip"], J["right_knee"], J["right_ankle"]),
        "hands_separation": float(np.linalg.norm(joints24[J["left_hand"]] - joints24[J["right_hand"]])),
        "left_hand_reach_y": float(joints24[J["left_shoulder"]][1] - joints24[J["left_hand"]][1]),
        "right_hand_reach_y": float(joints24[J["right_shoulder"]][1] - joints24[J["right_hand"]][1]),
    }


def tilt_angle_deg(global_orient):
    R = global_orient.reshape(3, 3)
    body_up = R[:, 1]
    cos_angle = np.clip(np.dot(body_up, np.array([0.0, 1.0, 0.0])), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def project_joints_to_2d(joints24, cam_t, focal_length, img_w, img_h):
    """Perspective-project canonical-space 3D joints into original-frame
    pixel coordinates.

    joints24 comes from decode_smpl_joints(), which regresses joints from
    mesh vertices via J_regressor -- so joints24 lives in the SAME coordinate
    space as the mesh vertices used by the (known-correct) mesh overlay
    renderer. We therefore add cam_t directly, the same way that renderer
    does (verts + cam_t), with NO extra pelvis-recentering step -- adding an
    extra recentering here would double-offset the points relative to the
    proven-correct convention.

    NOTE: if the overlay looks flipped/offset on your footage, your build's
    batch['img_size'] may be ordered [H, W] instead of [W, H] -- swap
    img_w/img_h below.
    """
    cam_pts = joints24 + cam_t
    x = cam_pts[:, 0] / cam_pts[:, 2] * focal_length + img_w / 2.0
    y = cam_pts[:, 1] / cam_pts[:, 2] * focal_length + img_h / 2.0
    return np.stack([x, y], axis=-1)


def run_hmr2_pose_extraction(frames_meta, out_dir, conf_thresh, batch_size, force=False):
    smpl_dir = os.path.join(out_dir, "smpl_params")
    os.makedirs(smpl_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "pose_summary.json")

    if os.path.exists(summary_path) and not force:
        print(f"[Phase 1] Found cached {summary_path}, skipping HMR2 inference.")
        with open(summary_path) as f:
            return json.load(f)

    device = get_device()
    print(f"[Phase 1] Using device: {device}")

    print("[Phase 1] Loading YOLO detector...")
    det_model = YOLO("yolov8n.pt")

    print("[Phase 1] Loading HMR2 model...")
    model, model_cfg = load_hmr2_no_renderer(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    pose_summary = []
    prev_primary_joints = None
    prev_timestamp = None

    for meta in frames_meta:
        frame_path, timestamp = meta["path"], meta["timestamp"]
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]
        cache_path = os.path.join(smpl_dir, f"{frame_name}.npz")

        img_cv2 = cv2.imread(frame_path)
        if img_cv2 is None:
            print(f"  {frame_name}: could not read frame, skipping")
            continue
        img_h, img_w = img_cv2.shape[:2]

        det_results = det_model(img_cv2, classes=[0], conf=conf_thresh, verbose=False)
        boxes = det_results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            pose_summary.append({"timestamp_sec": timestamp, "num_people": 0, "primary": None})
            print(f"  {frame_name}: no person detected")
            continue

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_joints, all_orients, all_box_sizes, all_cam_t, all_focal = [], [], [], [], []
        for batch in dataloader:
            batch = {
                k: (v.to(device, dtype=torch.float32) if v.dtype == torch.float64 else v.to(device))
                if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            with torch.no_grad():
                out = model(batch)

            pred_smpl_params = out["pred_smpl_params"]
            # verts_batch is unused here (only needed for a full mesh overlay);
            # joints24_batch is the guaranteed-correctly-ordered canonical joints.
            verts_batch, joints24_batch = decode_smpl_joints(model, pred_smpl_params)  # (B,6890,3), (B,24,3)
            global_orient = pred_smpl_params["global_orient"].detach().cpu().numpy()  # (B, 1, 3, 3)

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size_t = batch["img_size"].float()
            scaled_focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size_t.max()
            )
            cam_t_full = cam_crop_to_full(
                out["pred_cam"], box_center, box_size, img_size_t, scaled_focal_length
            ).detach().cpu().numpy()

            all_joints.append(joints24_batch)
            all_orients.append(global_orient)
            all_box_sizes.append(box_size.detach().cpu().numpy())
            all_cam_t.append(cam_t_full)
            all_focal.append(np.full((joints24_batch.shape[0],), float(scaled_focal_length)))

        joints_all = np.concatenate(all_joints, axis=0)
        orients_all = np.concatenate(all_orients, axis=0)
        box_sizes = np.concatenate(all_box_sizes, axis=0)
        cam_t_all = np.concatenate(all_cam_t, axis=0)
        focal_all = np.concatenate(all_focal, axis=0)

        np.savez(cache_path, joints=joints_all, global_orient=orients_all, boxes=boxes, cam_t=cam_t_all)

        # primary person = largest detected box (single-astronaut scene assumption)
        primary_idx = int(np.argmax(box_sizes))
        primary_joints = joints_all[primary_idx]
        primary_orient = orients_all[primary_idx]

        velocity = None
        if prev_primary_joints is not None and prev_timestamp is not None:
            dt = timestamp - prev_timestamp
            if dt > 0:
                velocity = float(np.linalg.norm(primary_joints - prev_primary_joints, axis=-1).mean() / dt)

        skeleton_2d = project_joints_to_2d(
            primary_joints, cam_t_all[primary_idx], focal_all[primary_idx], img_w, img_h
        )

        primary_summary = {
            "velocity_magnitude": velocity,
            "body_tilt_deg": tilt_angle_deg(primary_orient),
            "bbox": boxes[primary_idx].tolist(),
            "skeleton_2d": skeleton_2d.tolist(),
            **compute_limb_metrics(primary_joints),
        }

        pose_summary.append({
            "timestamp_sec": timestamp,
            "num_people": int(len(boxes)),
            "primary": primary_summary,
        })

        prev_primary_joints = primary_joints
        prev_timestamp = timestamp
        print(f"  {frame_name}: {len(boxes)} person(s), tilt={primary_summary['body_tilt_deg']:.1f} deg, "
              f"L-elbow={primary_summary['left_elbow_deg']:.0f} deg, "
              f"R-elbow={primary_summary['right_elbow_deg']:.0f} deg")

    with open(summary_path, "w") as f:
        json.dump(pose_summary, f, indent=2)
    print(f"[Phase 1] Wrote pose telemetry -> {summary_path}")
    return pose_summary


def nearest_pose_entry(pose_summary, timestamp, tol=1.0):
    if not pose_summary:
        return None
    best = min(pose_summary, key=lambda e: abs(e["timestamp_sec"] - timestamp))
    return best if abs(best["timestamp_sec"] - timestamp) <= tol else None


# ---------------------------------------------------------------------------
# Phase 2: Gemini analysis, grounded in the pose telemetry
# ---------------------------------------------------------------------------
def format_pose_line(entry):
    if not entry or not entry.get("primary"):
        return f"T+{entry['timestamp_sec']:.2f}s: no person detected by pose model" if entry else None
    p = entry["primary"]
    required = ["left_elbow_deg", "right_elbow_deg", "left_knee_deg", "right_knee_deg",
                "hands_separation", "left_hand_reach_y", "right_hand_reach_y", "body_tilt_deg"]
    if not all(k in p for k in required):
        return f"T+{entry['timestamp_sec']:.2f}s: incomplete pose telemetry (stale cache? re-run with --force_pose)"
    v = f"{p['velocity_magnitude']:.4f}" if p["velocity_magnitude"] is not None else "n/a"
    return (
        f"T+{entry['timestamp_sec']:.2f}s: velocity={v}, tilt={p['body_tilt_deg']:.1f}deg, "
        f"L-elbow={p['left_elbow_deg']:.0f}deg, R-elbow={p['right_elbow_deg']:.0f}deg, "
        f"L-knee={p['left_knee_deg']:.0f}deg, R-knee={p['right_knee_deg']:.0f}deg, "
        f"hands_apart={p['hands_separation']:.3f}, "
        f"L-hand-reach={p['left_hand_reach_y']:.3f}, R-hand-reach={p['right_hand_reach_y']:.3f}"
    )


def _extract_retry_delay_seconds(exc, default=20.0):
    """Pull the server-suggested backoff out of a google.genai APIError.
    429 RESOURCE_EXHAUSTED responses include a RetryInfo entry like
    {'@type': '...google.rpc.RetryInfo', 'retryDelay': '38s'} inside the
    error details -- honor that instead of guessing, since the server knows
    its own quota reset window. Falls back to `default` if it can't find one."""
    try:
        details = getattr(exc, "details", None) or {}
        # google-genai APIError often exposes the raw error body via .details
        # or via str(exc); check both since the SDK's shape has varied.
        candidates = []
        if isinstance(details, dict):
            candidates.append(details)
        candidates.append({"_raw": str(exc)})

        for c in candidates:
            text = json.dumps(c) if isinstance(c, dict) else str(c)
            match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", text)
            if match:
                return float(match.group(1))
    except Exception:
        pass
    return default


def _call_gemini_with_retry(client, gemini_model, contents, config, max_retries=5):
    """Call generate_content, retrying on 429/RESOURCE_EXHAUSTED using the
    server's own suggested retryDelay (from RetryInfo) plus jitter, with
    exponential growth as a fallback if the server doesn't specify one.
    Re-raises on non-retryable errors or once max_retries is exhausted."""
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=gemini_model, contents=contents, config=config)
        except genai_errors.APIError as e:
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            is_rate_limit = status == 429 or "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
            if not is_rate_limit or attempt == max_retries:
                raise
            delay = _extract_retry_delay_seconds(e, default=min(60, 5 * (2 ** attempt)))
            delay += random.uniform(0, 2)  # jitter, avoid thundering herd on retries
            print(f"    Rate limited (attempt {attempt + 1}/{max_retries}), "
                  f"waiting {delay:.1f}s before retry...")
            time.sleep(delay)


def analyze_video_with_gemini(frames_meta, pose_summary, protocol, gemini_model, chunk_size=100,
                               request_pacing_sec=2.0):
    print(f"\n[Phase 2] Sending {len(frames_meta)} frames to Gemini in chunks of {chunk_size}...")
    client = genai.Client()
    master_timeline = []

    for i in range(0, len(frames_meta), chunk_size):
        chunk = frames_meta[i:i + chunk_size]
        start_t, end_t = chunk[0]["timestamp"], chunk[-1]["timestamp"]
        print(f"  Chunk {i // chunk_size + 1}: T+{start_t:.1f}s to T+{end_t:.1f}s")

        # Small fixed pause between chunk requests so we don't fire a burst of
        # huge multi-image, high-res requests back-to-back and trip a
        # requests-per-minute / tokens-per-minute quota in the first place.
        if i > 0 and request_pacing_sec > 0:
            time.sleep(request_pacing_sec)

        chunk_parts, pose_lines = [], []
        for item in chunk:
            with open(item["path"], "rb") as f:
                img_bytes = f.read()
            chunk_parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

            entry = nearest_pose_entry(pose_summary, item["timestamp"])
            line = format_pose_line(entry)
            if line:
                pose_lines.append(line)

        prompt_text = f"""
Analyze this sequential batch of frames from a microgravity experiment chronologically.
These frames span from approximately T+{start_t:.1f}s to T+{end_t:.1f}s.
The exact timestamp is burned into the top-left of every frame.

Independent 3D body-pose telemetry, extracted via an SMPL/HMR2 model (NOT from
looking at the images) -- use this to ground your visual judgment:
- velocity: joint displacement per second. Spikes = fast/jerky motion (poor stabilization).
- tilt: degrees the spine is off vertical.
- elbow/knee angles: 180deg = fully straight limb, small angle = sharply bent.
  An elbow near 90deg usually means gripping/turning something; two nearly-straight
  arms mean the astronaut isn't actively engaging hardware.
- hands_apart: 3D distance between the two hands. Small = both hands on the same object.
- hand-reach: shoulder height minus hand height. Positive = reaching upward, negative = downward.
{chr(10).join(pose_lines)}

Expected Protocol:
{protocol}
"""

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExperimentTimeline,
            temperature=0.0,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        )

        try:
            response = _call_gemini_with_retry(
                client, gemini_model, [prompt_text] + chunk_parts, config
            )
            chunk_analysis: ExperimentTimeline = response.parsed
            if chunk_analysis is None:
                print(f"  Error processing chunk starting at T+{start_t:.1f}s: "
                      f"response.parsed was None (model output likely failed schema validation)")
                continue
            master_timeline.extend(chunk_analysis.timeline)
        except Exception as e:
            import traceback
            print(f"  Error processing chunk starting at T+{start_t:.1f}s: {e}")
            traceback.print_exc()

    sorted_timeline = sorted(master_timeline, key=lambda x: x.timestamp_sec)
    print(f"[Phase 2] Master timeline assembled with {len(sorted_timeline)} entries.")
    return sorted_timeline


# ---------------------------------------------------------------------------
# Phase 3: playback UI with a live HMR skeleton overlay
# ---------------------------------------------------------------------------
class TelemetryApp:
    SKELETON_OK_COLOR = (255, 255, 0)     # BGR cyan
    SKELETON_ERROR_COLOR = (0, 0, 255)    # BGR red

    def __init__(self, root, video_path, timeline, pose_summary):
        self.root = root
        self.root.title("Microgravity Experiment Telemetry Monitor")
        self.root.geometry("1150x680")
        self.root.configure(bg="#1e1e1e")

        self.video_path = video_path
        self.timeline = list(timeline)
        self.pose_summary = pose_summary
        self.current_is_error = False

        self.video_frame = tk.Frame(root, bg="black")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(self.video_frame, text="Cyan = HMR2 3D skeleton (live) \u00b7 turns red when Gemini flags an error",
                 fg="#8be9fd", bg="black", font=("Arial", 10)).pack(anchor="w")
        self.canvas = tk.Canvas(self.video_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.log_frame = tk.Frame(root, bg="#252526")
        self.log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(self.log_frame, text="AI + Pose Telemetry Live Stream", fg="#ffffff", bg="#252526",
                 font=("Arial", 14, "bold")).pack(anchor="w", padx=5, pady=5)
        self.text_box = tk.Text(self.log_frame, bg="#1e1e1e", fg="#00ff00", insertbackground="white",
                                 font=("Courier", 11))
        self.text_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = 0
        self.is_playing = True
        self.thread = threading.Thread(target=self.playback_loop, daemon=True)
        self.thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def log_message(self, message):
        self.text_box.insert(tk.END, message + "\n")
        self.text_box.see(tk.END)

    def draw_skeleton(self, frame, entry):
        if not entry or not entry.get("primary"):
            return
        p = entry["primary"]
        pts = p.get("skeleton_2d")
        if not pts:
            return
        color = self.SKELETON_ERROR_COLOR if self.current_is_error else self.SKELETON_OK_COLOR

        bbox = p.get("bbox")
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        for a, b in SMPL_BONES:
            pa, pb = pts[a], pts[b]
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    def playback_loop(self):
        while self.is_playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.log_message("\n--- Playback Concluded ---")
                break

            current_t = self.frame_count / self.fps

            while self.timeline and current_t >= self.timeline[0].timestamp_sec:
                telem = self.timeline.pop(0)
                self.current_is_error = telem.is_error
                status = "\u274c ERROR" if telem.is_error else "\u2705 NOMINAL"
                self.log_message(f"[T+{telem.timestamp_sec:.2f}s] {status}")
                self.log_message(f"Hardware: {telem.hardware_interaction}")
                self.log_message(f"Action: {telem.action_description}")
                if telem.is_error:
                    self.log_message(f"Correction: {telem.correction}")

                pose_entry = nearest_pose_entry(self.pose_summary, telem.timestamp_sec)
                line = format_pose_line(pose_entry)
                if line:
                    self.log_message(f"Pose: {line.split(': ', 1)[1]}")
                self.log_message("-" * 40)

            pose_entry = nearest_pose_entry(self.pose_summary, current_t, tol=2.0 / self.fps if self.fps else 1.0)
            self.draw_skeleton(frame, pose_entry)
            cv2.putText(frame, f"T+{current_t:.2f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)

            canvas_width = self.canvas.winfo_width() or 600
            canvas_height = self.canvas.winfo_height() or 500
            if canvas_width > 10 and canvas_height > 10:
                image = image.resize((canvas_width, canvas_height), Image.Resampling.BILINEAR)

            photo = ImageTk.PhotoImage(image=image)
            self.canvas.image = photo
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

            self.frame_count += 1
            time.sleep(1.0 / self.fps)

    def on_close(self):
        self.is_playing = False
        self.cap.release()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out_dir", default="output", help="Where frames/telemetry/cache are saved")
    parser.add_argument("--protocol_file", default=None, help="Optional path to a text file with the protocol")
    parser.add_argument("--target_fps", type=float, default=3.0)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--request_pacing_sec", type=float, default=2.0,
                         help="Fixed pause between Gemini chunk requests, to avoid bursting past rate limits")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gemini_model", default="gemini-2.5-flash",
                         help="Must match a model name available to your API key")
    parser.add_argument("--force_pose", action="store_true", help="Re-run HMR2 even if pose_summary.json exists")
    parser.add_argument("--force_gemini", action="store_true", help="Re-run Gemini even if timeline cache exists")
    parser.add_argument("--skip_ui", action="store_true", help="Skip the playback window, just compute + cache")
    args = parser.parse_args()

    protocol = DEFAULT_PROTOCOL
    if args.protocol_file:
        with open(args.protocol_file) as f:
            protocol = f.read()

    frames_dir = os.path.join(args.out_dir, "frames")
    timeline_cache = os.path.join(args.out_dir, "timeline_cache.pkl")

    frames_meta, _ = extract_frames(args.video, frames_dir, args.target_fps)

    pose_summary = run_hmr2_pose_extraction(
        frames_meta, args.out_dir, args.conf_thresh, args.batch_size, force=args.force_pose
    )

    if os.path.exists(timeline_cache) and not args.force_gemini:
        print(f"[Phase 2] Found cached {timeline_cache}, skipping Gemini calls.")
        with open(timeline_cache, "rb") as f:
            timeline = pickle.load(f)
    else:
        timeline = analyze_video_with_gemini(frames_meta, pose_summary, protocol, args.gemini_model,
                                              args.chunk_size, args.request_pacing_sec)
        with open(timeline_cache, "wb") as f:
            pickle.dump(timeline, f)
        print(f"[Phase 2] Timeline cached -> {timeline_cache}")

    if not timeline:
        print("No timeline data produced -- stopping before UI.")
        return

    if not args.skip_ui:
        root = tk.Tk()
        TelemetryApp(root, args.video, timeline, pose_summary)
        root.mainloop()


if __name__ == "__main__":
    main()