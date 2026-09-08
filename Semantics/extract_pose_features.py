"""
Convert per-frame SMPL parameters (rotation matrices) into interpretable
joint-angle features suitable as input to an activity-recognition classifier.

Approach: rather than decomposing SMPL's rotation matrices directly (which
requires knowing SMPL's internal per-joint local axis conventions), we run
the SMPL forward kinematics via `smplx` to get actual 3D joint positions,
then compute angles as the angle between anatomically relevant bone vectors
(e.g. elbow angle = angle between upper-arm and forearm vectors at the elbow).

Important property: angles computed this way (angle between two bone vectors
meeting at a joint) are ORIENTATION-INVARIANT - rotating the whole body in
space doesn't change the angle at the elbow. So most of these features don't
need the rack-relative calibration to be meaningful. The exceptions are
explicitly marked below (torso lean, wrist height) - those reference an
external "up" direction and are placeholders using camera-up until a real
rack-relative transform is calibrated.

Run from inside the 4D-Humans repo root, with the '4D-humans' conda env active:
    python extract_pose_features.py --params_dir output/smpl_params \
        --out_csv output/pose_features.csv
"""

import argparse
import glob
import os
import numpy as np
import torch
import smplx
import pandas as pd


# Standard SMPL joint indices (0 = pelvis/root, via global_orient).
# body_pose array index i corresponds to joint (i + 1) in this list.
# This is the standard SMPL kinematic tree convention (same in smplx).
# If in doubt, verify with: print(model.parents) after loading the model.
SMPL_JOINTS = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3,
    "left_knee": 4, "right_knee": 5, "spine2": 6,
    "left_ankle": 7, "right_ankle": 8, "spine3": 9,
    "left_foot": 10, "right_foot": 11, "neck": 12,
    "left_collar": 13, "right_collar": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    "left_hand": 22, "right_hand": 23,
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_smpl_model(device):
    cache_dir = os.path.expanduser("~/.cache/4DHumans/data")
    model = smplx.create(
        model_path=cache_dir, model_type="smpl", gender="neutral",
        num_betas=10, batch_size=1,
    ).to(device)
    model.eval()
    return model


def angle_between(v1, v2):
    """Angle in degrees between two 3D vectors."""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_joint_features(joints, body_scale):
    """
    joints: (24, 3) array of 3D joint positions (SMPL order, root-relative or not - doesn't matter for angles).
    body_scale: scalar used to normalize distance-based features across
                different subjects / camera distances (e.g. shoulder width).
    Returns a dict of interpretable features for this single frame.
    """
    J = SMPL_JOINTS
    feats = {}

    # --- Orientation-invariant joint angles (bone-vector angles) ---
    # Elbow flexion: angle at the elbow between upper arm and forearm.
    for side in ["left", "right"]:
        shoulder = joints[J[f"{side}_shoulder"]]
        elbow = joints[J[f"{side}_elbow"]]
        wrist = joints[J[f"{side}_wrist"]]
        upper_arm = shoulder - elbow
        forearm = wrist - elbow
        feats[f"elbow_angle_{side}"] = angle_between(upper_arm, forearm)

        # Shoulder abduction: angle between torso axis (spine) and upper arm.
        spine_vec = joints[J["neck"]] - joints[J["pelvis"]]
        upper_arm_dir = elbow - shoulder
        feats[f"shoulder_abduction_{side}"] = angle_between(spine_vec, upper_arm_dir)

        # Knee flexion: angle at the knee between thigh and shin.
        hip = joints[J[f"{side}_hip"]]
        knee = joints[J[f"{side}_knee"]]
        ankle = joints[J[f"{side}_ankle"]]
        thigh = hip - knee
        shin = ankle - knee
        feats[f"knee_angle_{side}"] = angle_between(thigh, shin)

    # Torso twist: angle between the hip line and shoulder line, projected
    # onto the plane perpendicular to the spine. Orientation-invariant since
    # it's a relative angle between two body-fixed lines, not an absolute direction.
    hip_line = joints[J["right_hip"]] - joints[J["left_hip"]]
    shoulder_line = joints[J["right_shoulder"]] - joints[J["left_shoulder"]]
    feats["torso_twist"] = angle_between(hip_line, shoulder_line)

    # --- Features that reference an external "up" direction ---
    # PLACEHOLDER: using world/camera-frame vertical (0,1,0) as "up" here.
    # Replace this with the rack-relative "up" vector once the rack
    # calibration transform exists - until then, these two features are
    # NOT reliable for an astronaut in an arbitrary orientation and should
    # be treated as provisional.
    camera_up = np.array([0.0, -1.0, 0.0])  # SMPL/camera convention: -Y is often "up"; verify against your data
    spine_vec = joints[J["neck"]] - joints[J["pelvis"]]
    feats["torso_lean_vs_camera_up_PROVISIONAL"] = angle_between(spine_vec, camera_up)

    for side in ["left", "right"]:
        wrist = joints[J[f"{side}_wrist"]]
        shoulder = joints[J[f"{side}_shoulder"]]
        # Signed distance along camera_up axis, normalized by body_scale.
        rel = wrist - shoulder
        feats[f"wrist_height_rel_shoulder_{side}_PROVISIONAL"] = float(np.dot(rel, camera_up) / body_scale)

    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_dir", required=True, help="Directory of per-frame .npz SMPL params")
    parser.add_argument("--out_csv", default="output/pose_features.csv")
    args = parser.parse_args()

    device = get_device()
    model = load_smpl_model(device)

    npz_paths = sorted(glob.glob(os.path.join(args.params_dir, "*.npz")))
    if not npz_paths:
        print(f"No .npz files found in {args.params_dir}")
        return
    print(f"Found {len(npz_paths)} parameter files")

    rows = []
    for npz_path in npz_paths:
        frame_name = os.path.splitext(os.path.basename(npz_path))[0]
        data = np.load(npz_path)
        n_people = data["global_orient"].shape[0]

        for person_idx in range(n_people):
            global_orient = torch.from_numpy(data["global_orient"][person_idx:person_idx + 1]).float().to(device)
            body_pose = torch.from_numpy(data["body_pose"][person_idx:person_idx + 1]).float().to(device)
            betas = torch.from_numpy(data["betas"][person_idx:person_idx + 1]).float().to(device)

            with torch.no_grad():
                smpl_out = model(global_orient=global_orient, body_pose=body_pose, betas=betas, pose2rot=False)
            joints = smpl_out.joints[0, :24].cpu().numpy()  # first 24 = standard SMPL joints

            # Body scale reference for normalizing distance-based features:
            # shoulder-to-shoulder width (fairly stable across poses, unlike e.g. height).
            shoulder_width = np.linalg.norm(
                joints[SMPL_JOINTS["left_shoulder"]] - joints[SMPL_JOINTS["right_shoulder"]]
            )
            body_scale = max(shoulder_width, 1e-4)

            feats = compute_joint_features(joints, body_scale)
            feats["frame_id"] = frame_name
            feats["person_idx"] = person_idx
            feats["detected"] = True
            rows.append(feats)

        print(f"{frame_name}: extracted features for {n_people} person(s)")

    df = pd.DataFrame(rows)
    # Put frame_id / person_idx / detected first for readability
    cols = ["frame_id", "person_idx", "detected"] + [c for c in df.columns if c not in ("frame_id", "person_idx", "detected")]
    df = df[cols]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nDone. Feature table saved to: {args.out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()