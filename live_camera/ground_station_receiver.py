"""
Ground-station receiver for the on-board UDP mirror (GUI/stream_sender.py).

Run on the "ground" machine, then enable streaming in the app's STREAM dialog
with this machine's IP and the same port:

    python ground_station_receiver.py --port 5599
    python ground_station_receiver.py --port 5599 --save ground_out     # also keep video + metadata
    python ground_station_receiver.py --port 5599 --headless            # no window, just print/save

Packet format (one UDP datagram per frame):
    [2 bytes big-endian: JSON length][JSON metadata][JPEG bytes]

Needs only opencv-python and numpy. Press q or Esc in the window to quit.
"""

import argparse
import json
import socket
import struct
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

STATUS_COLORS = {  # BGR
    "done": (80, 175, 76),
    "in_progress": (0, 165, 255),
    "wrong": (60, 60, 230),
    "pending": (150, 150, 150),
}
NO_SIGNAL_AFTER = 3.0  # seconds


def parse_packet(data):
    if len(data) < 3:
        raise ValueError("packet too short")
    (meta_len,) = struct.unpack(">H", data[:2])
    if 2 + meta_len >= len(data):
        raise ValueError("bad metadata length")
    meta = json.loads(data[2:2 + meta_len].decode("utf-8"))
    frame = cv2.imdecode(np.frombuffer(data[2 + meta_len:], np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("JPEG decode failed")
    return meta, frame


def annotate(frame, meta, scale):
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    status = str(meta.get("status", "pending"))
    color = STATUS_COLORS.get(status, STATUS_COLORS["pending"])
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)
    lines = [f"STEP: {meta.get('step', '-')}   STATUS: {status.upper()}",
             f"ELAPSED: {meta.get('elapsed', '--:--')}   SENT: {meta.get('ts', '-')}"]
    alert = meta.get("alert")
    if alert:
        lines.insert(0, f"ALERT: {alert}")
    y = 22
    for i, text in enumerate(lines):
        c = (60, 60, 230) if (alert and i == 0) else (255, 255, 255)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        y += 20
    return frame


def main():
    ap = argparse.ArgumentParser(description="Ground station receiver for the experiment video mirror")
    ap.add_argument("--host", default="0.0.0.0", help="interface to listen on (default: all)")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--save", metavar="DIR", help="save received video + metadata (jsonl) here")
    ap.add_argument("--headless", action="store_true", help="no window; print events only")
    ap.add_argument("--scale", type=float, default=2.0, help="window upscale factor (default 2)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after N seconds (0 = run until quit)")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.5)
    print(f"Listening on {args.host}:{args.port}  (Ctrl+C to stop)")

    writer = meta_file = None
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        video_path = out / f"ground_{stamp}.mp4"
        meta_file = open(out / f"ground_{stamp}.jsonl", "w", encoding="utf-8")

    last_key, last_rx, frames = None, 0.0, 0
    started = time.time()
    signal_lost = True
    try:
        while True:
            if args.max_seconds and time.time() - started > args.max_seconds:
                break
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                if not signal_lost and time.time() - last_rx > NO_SIGNAL_AFTER:
                    signal_lost = True
                    print("NO SIGNAL")
                if not args.headless:
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue
            try:
                meta, frame = parse_packet(data)
            except Exception as e:
                print("Bad packet ignored:", e)
                continue
            last_rx, frames = time.time(), frames + 1
            if signal_lost:
                print(f"Signal from {addr[0]}:{addr[1]}")
                signal_lost = False
            key = (meta.get("step"), meta.get("status"), meta.get("alert"))
            if key != last_key:  # print only when something changes
                print(f"[{meta.get('ts', '-')}] step={meta.get('step')} status={meta.get('status')}"
                      + (f" ALERT={meta.get('alert')}" if meta.get("alert") else ""))
                last_key = key
            if meta_file:
                meta_file.write(json.dumps({"rx_time": datetime.now().isoformat(timespec="milliseconds"), **meta}) + "\n")
                meta_file.flush()
            shown = annotate(frame, meta, args.scale)
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0,
                                             (shown.shape[1], shown.shape[0]))
                writer.write(shown)
            if not args.headless:
                cv2.imshow("Ground Station - experiment feed", shown)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        if writer:
            writer.release()
        if meta_file:
            meta_file.close()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"Stopped. {frames} frames received.")


if __name__ == "__main__":
    main()
