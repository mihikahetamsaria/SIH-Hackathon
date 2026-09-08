"""
pose_and_box.py

Runs MediaPipe Pose on an input video, and for every frame:
  1. Detects the person's pose (33 body landmarks)
  2. Draws the pose skeleton on the frame
  3. Draws a bounding box around the detected person
      (computed from the landmark coordinates, padded a bit)

Also writes out a per-frame JSON log of landmark coordinates + visibility
scores, since you'll likely want that for feature extraction later
(this part is optional -- pass --no-json to skip it).

Usage:
    python pose_and_box.py --input path/to/video.mp4 --output path/to/output.mp4

Requirements:
    pip install mediapipe opencv-python
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def compute_bbox(landmarks, frame_width, frame_height, padding_ratio=0.08):
    """
    Given MediaPipe pose landmarks (normalized 0-1 coords), compute a
    pixel-space bounding box around the person, with a bit of padding
    so the box doesn't hug the skeleton too tightly.

    Landmarks with very low visibility are ignored when computing the
    box, so a single bad/occluded point doesn't blow up the box size.
    """
    visible_points = [
        (lm.x, lm.y) for lm in landmarks if lm.visibility > 0.3
    ]
    if not visible_points:
        return None

    xs = [p[0] for p in visible_points]
    ys = [p[1] for p in visible_points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # pad the box a little (in normalized coords) so it's not skin-tight
    pad_x = (x_max - x_min) * padding_ratio + 0.02
    pad_y = (y_max - y_min) * padding_ratio + 0.02

    x_min = max(0.0, x_min - pad_x)
    x_max = min(1.0, x_max + pad_x)
    y_min = max(0.0, y_min - pad_y)
    y_max = min(1.0, y_max + pad_y)

    # convert to pixel coordinates
    px_x_min = int(x_min * frame_width)
    px_x_max = int(x_max * frame_width)
    px_y_min = int(y_min * frame_height)
    px_y_max = int(y_max * frame_height)

    return px_x_min, px_y_min, px_x_max, px_y_max


def process_video(input_path, output_path, json_path=None,
                   min_detection_confidence=0.5, min_tracking_confidence=0.5,
                   model_complexity=1):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    all_frame_data = []  # only filled in if json_path is given

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe expects RGB, OpenCV gives BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            frame_record = {
                "frame_index": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 4),
                "person_detected": False,
                "bbox": None,
                "landmarks": None,
            }

            if results.pose_landmarks:
                frame_record["person_detected"] = True

                # draw the skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # draw the bounding box
                bbox = compute_bbox(
                    results.pose_landmarks.landmark, frame_width, frame_height
                )
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(
                        frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, "person", (x_min, max(0, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
                    frame_record["bbox"] = {
                        "x_min": x_min, "y_min": y_min,
                        "x_max": x_max, "y_max": y_max,
                    }

                if json_path:
                    frame_record["landmarks"] = [
                        {
                            "id": i,
                            "x": round(lm.x, 5),
                            "y": round(lm.y, 5),
                            "z": round(lm.z, 5),
                            "visibility": round(lm.visibility, 4),
                        }
                        for i, lm in enumerate(results.pose_landmarks.landmark)
                    ]
            else:
                # no person detected this frame -- leave bbox/landmarks as None
                # so downstream code can explicitly handle "no detection"
                cv2.putText(
                    frame, "no person detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            out.write(frame)
            if json_path:
                all_frame_data.append(frame_record)

            frame_idx += 1
            if total_frames:
                print(f"\rProcessing frame {frame_idx}/{total_frames}", end="", flush=True)

    print()  # newline after progress bar
    cap.release()
    out.release()

    if json_path:
        with open(json_path, "w") as f:
            json.dump(
                {
                    "source_video": str(input_path),
                    "fps": fps,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "frames": all_frame_data,
                },
                f,
                indent=2,
            )
        print(f"Landmark data written to: {json_path}")

    print(f"Annotated video written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Pose on a video: draw skeleton + bounding box."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input video")
    parser.add_argument("--output", "-o", required=True, help="Path to save annotated output video")
    parser.add_argument(
        "--no-json", action="store_true",
        help="Skip writing the per-frame landmark JSON log (written by default, next to the output video)"
    )
    parser.add_argument(
        "--model-complexity", type=int, default=1, choices=[0, 1, 2],
        help="MediaPipe pose model complexity: 0=lite/fastest, 1=full (default), 2=heavy/most accurate"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    json_path = None
    if not args.no_json:
        json_path = output_path.with_suffix(".json")

    try:
        process_video(
            args.input,
            args.output,
            json_path=json_path,
            model_complexity=args.model_complexity,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()