"""
Render the reconstructed SMPL mesh on top of the original video frames,
without using pyrender (whose OffscreenRenderer doesn't support macOS at all).

Approach:
  1. Load the SMPL body model via `smplx` (pure PyTorch, no OpenGL).
  2. For each frame, reconstruct 3D mesh vertices from the saved SMPL
     parameters (global_orient, body_pose, betas).
  3. Add the saved full-frame camera translation to get camera-space 3D points.
  4. Project to 2D with a simple pinhole camera model.
  5. Rasterize the mesh as filled, flat-shaded triangles directly with OpenCV,
     using a painter's algorithm (draw far-to-near) for a basic solid look.

Run from inside the 4D-Humans repo root, with the '4D-humans' conda env active,
AFTER re-running run_hmr2_on_video.py with the updated version that saves
pred_cam_t_full / focal_length / img_size:

    python render_mesh_overlay.py --frames_dir output/frames \
        --params_dir output/smpl_params --out_video output/mesh_overlay.mp4
"""

import argparse
import os
import glob
import numpy as np
import cv2
import torch
import smplx


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_smpl_model(device):
    # SMPL_NEUTRAL.pkl was placed at ~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl
    # earlier in setup. smplx expects a folder structure: <model_path>/smpl/SMPL_NEUTRAL.pkl
    cache_dir = os.path.expanduser("~/.cache/4DHumans/data")
    model = smplx.create(
        model_path=cache_dir,
        model_type="smpl",
        gender="neutral",
        num_betas=10,
        batch_size=1,
    ).to(device)
    model.eval()
    return model


def project_points(points_3d, focal_length, img_w, img_h):
    """Simple pinhole projection, principal point at image center."""
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z = np.clip(z, 1e-4, None)  # avoid divide-by-zero / behind-camera artifacts
    x_2d = focal_length * x / z + img_w / 2.0
    y_2d = focal_length * y / z + img_h / 2.0
    return np.stack([x_2d, y_2d], axis=-1)


def render_mesh_on_frame(frame, verts_2d, verts_depth, faces, color=(180, 130, 60), alpha=0.75):
    """
    Draw the mesh as filled, flat-shaded triangles using a painter's algorithm:
    sort faces back-to-front by average depth, fill each with a shade based
    on a fixed light direction (approximated via face 'compactness' as a cheap
    stand-in for a normal-based shading term, since we're keeping this simple
    and dependency-free).
    """
    overlay = frame.copy()

    face_depths = verts_depth[faces].mean(axis=1)
    draw_order = np.argsort(-face_depths)  # far first (largest depth first)

    depth_min, depth_max = verts_depth.min(), verts_depth.max()
    depth_range = max(depth_max - depth_min, 1e-6)

    for face_idx in draw_order:
        tri = faces[face_idx]
        pts_2d = verts_2d[tri]
        if not np.all(np.isfinite(pts_2d)):
            continue
        pts_2d_int = pts_2d.astype(np.int32)

        # Cheap "shading": closer to camera (smaller depth) drawn lighter,
        # farther drawn darker - gives a basic sense of depth without true normals.
        mean_depth = face_depths[face_idx]
        shade = 1.0 - 0.5 * (mean_depth - depth_min) / depth_range
        shaded_color = tuple(int(c * shade) for c in color)

        cv2.fillConvexPoly(overlay, pts_2d_int, shaded_color, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return blended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True, help="Directory of extracted video frames")
    parser.add_argument("--params_dir", required=True, help="Directory of per-frame .npz SMPL params")
    parser.add_argument("--out_video", default="output/mesh_overlay.mp4", help="Output overlay video path")
    parser.add_argument("--fps", type=float, default=6.0, help="Output video FPS (match your --every_n subsampling)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = load_smpl_model(device)
    faces = model.faces  # (13776, 3) triangle vertex indices

    npz_paths = sorted(glob.glob(os.path.join(args.params_dir, "*.npz")))
    if not npz_paths:
        print(f"No .npz files found in {args.params_dir}")
        return
    print(f"Found {len(npz_paths)} parameter files")

    writer = None
    for npz_path in npz_paths:
        frame_name = os.path.splitext(os.path.basename(npz_path))[0]
        frame_path = os.path.join(args.frames_dir, f"{frame_name}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: could not find matching frame {frame_path}, skipping")
            continue

        data = np.load(npz_path)
        n_people = data["global_orient"].shape[0]
        img_h, img_w = frame.shape[:2]

        rendered = frame.copy()
        for person_idx in range(n_people):
            global_orient = torch.from_numpy(data["global_orient"][person_idx:person_idx + 1]).float().to(device)
            body_pose = torch.from_numpy(data["body_pose"][person_idx:person_idx + 1]).float().to(device)
            betas = torch.from_numpy(data["betas"][person_idx:person_idx + 1]).float().to(device)

            with torch.no_grad():
                smpl_out = model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas,
                    pose2rot=False,  # we saved rotation matrices, not axis-angle
                )
            verts = smpl_out.vertices[0].cpu().numpy()  # (6890, 3), model-space

            cam_t_full = data["pred_cam_t_full"][person_idx]  # (3,)
            focal_length = float(np.array(data["focal_length"]).reshape(-1)[0])

            verts_cam = verts + cam_t_full  # move into camera-space
            verts_2d = project_points(verts_cam, focal_length, img_w, img_h)
            verts_depth = verts_cam[:, 2]

            rendered = render_mesh_on_frame(rendered, verts_2d, verts_depth, faces)

        if writer is None:
            os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out_video, fourcc, args.fps, (img_w, img_h))

        writer.write(rendered)
        print(f"{frame_name}: rendered {n_people} person(s)")

    if writer is not None:
        writer.release()
        print(f"\nDone. Overlay video saved to: {args.out_video}")
    else:
        print("No frames were rendered.")


if __name__ == "__main__":
    main()