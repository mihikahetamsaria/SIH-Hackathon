"""
Optional, low-bandwidth mirror of the live feed to a ground-control IP.

Design constraints this follows directly from the problem statement:
    - "Space stations operate on restricted data bandwidth to Earth" ->
      frames are downscaled + JPEG-compressed and sent at a low fixed
      rate (default 2 fps), not the full local feed.
    - "Software should be able to run locally without internet
      connection" -> this is a pure mirror. It never gates local
      recording or logging: if it's disabled, unconfigured, or the
      network is unreachable, maybe_send() is a silent no-op. It never
      raises out of this class and never blocks the caller.
    - UDP, not TCP, on purpose: on a lossy/high-latency space-to-ground
      link, dropping one stale frame is fine; stalling the video loop to
      retry a failed TCP send is not.

Wire format (one UDP datagram per frame):
    [2 bytes: big-endian length of JSON metadata] [JSON bytes] [JPEG bytes]

The metadata JSON carries the "relevant data" the challenge asks for
alongside the picture itself: current step, status, elapsed time, and a
timestamp - so ground control gets context, not just pixels.
"""

import json
import socket
import struct
import time

import cv2

MAX_UDP_PAYLOAD = 60000  # stay comfortably under the ~65507B datagram ceiling


class UDPFrameStreamer:
    def __init__(self, max_dim: int = 320, jpeg_quality: int = 50, send_interval: float = 0.5):
        self.enabled = False
        self.ip = None
        self.port = None
        self._sock = None
        self.max_dim = max_dim
        self.jpeg_quality = jpeg_quality
        self.send_interval = send_interval  # seconds between frames (0.5s = 2fps)
        self._last_sent = 0.0

    def configure(self, ip: str, port: int):
        """Set (or change) the destination. Safe to call while disabled."""
        self.ip = ip
        self.port = port
        try:
            if self._sock is None:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError:
            self._sock = None

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled) and self._sock is not None and bool(self.ip) and bool(self.port)

    def maybe_send(self, frame, metadata: dict):
        """Call this once per video-loop tick. No-ops unless enabled,
        configured, and the send interval has elapsed. Never raises."""
        if not self.enabled or not self._sock or not self.ip or not self.port:
            return
        now = time.time()
        if now - self._last_sent < self.send_interval:
            return

        try:
            h, w = frame.shape[:2]
            scale = self.max_dim / float(max(h, w))
            small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else frame

            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                return

            meta_bytes = json.dumps(metadata, default=str).encode("utf-8")
            packet = struct.pack(">H", len(meta_bytes)) + meta_bytes + buf.tobytes()
            if len(packet) > MAX_UDP_PAYLOAD:
                return  # drop rather than fragment across datagrams

            self._sock.sendto(packet, (self.ip, self.port))
            self._last_sent = now
        except Exception:
            # Link down, host unreachable, socket hiccup, bad frame, etc.
            # This path must never propagate - it would take the whole
            # video loop down over a feature that's explicitly optional.
            pass

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
