"""
Continuous full-session recorder ("store the video locally").

- write() is called from the UI thread once per camera tick. It only copies the
  frame and puts it on a queue, so it never blocks the video loop.
- A worker thread burns in a timestamp and writes to disk.
- Output is real-time: if the UI loop runs slower than `fps`, the last frame is
  repeated so the video's duration matches wall-clock time (max 5 repeats per
  tick, so a long stall is not silently papered over).
- If the disk or codec fails, the app keeps running and `last_error` is set.
"""

import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2


class SessionRecorder:
    def __init__(self, fps=20.0, queue_size=90, on_error=None):
        self.fps = float(fps)
        self._queue_size = queue_size
        self._on_error = on_error
        self.last_error = None
        self.path = None
        self._q = None
        self._thread = None
        self._writer = None
        self._size = None
        self._t0 = None
        self._frames_due = 0
        self._started_wall = None
        self.frames_written = 0
        self.frames_dropped = 0

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, path):
        """Begin a new recording. Returns True if the worker started."""
        self.stop()
        self.path = Path(path)
        self.last_error = None
        self._writer = None
        self._size = None
        self._t0 = None
        self._frames_due = 0
        self.frames_written = 0
        self.frames_dropped = 0
        self._started_wall = datetime.now()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._fail(f"cannot create recordings folder: {e}")
            return False
        self._q = queue.Queue(maxsize=self._queue_size)
        self._thread = threading.Thread(target=self._worker, args=(self._q,),
                                        name="session-recorder", daemon=True)
        self._thread.start()
        return True

    def write(self, frame):
        """Non-blocking. Safe to call every tick; no-op when not recording."""
        if not self.active:
            return
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        due = int((now - self._t0) * self.fps) + 1 - self._frames_due
        if due <= 0:
            return
        due = min(due, 5)
        self._frames_due += due
        try:
            # copy: the worker draws the timestamp, and the UI still uses `frame`
            self._q.put_nowait((frame.copy(), due, now))
        except queue.Full:
            self.frames_dropped += due

    def stop(self):
        """Flush and close the file. Returns the path (or None if nothing was recorded)."""
        thread, q = self._thread, self._q
        self._thread = None
        self._q = None
        if thread is not None and q is not None:
            q.put(None)
            thread.join(timeout=5.0)
        return self.path if self.frames_written else None

    # ---------------------------------------------------------------- worker
    def _worker(self, q):
        while True:
            item = q.get()
            if item is None:
                break
            frame, repeat, ts = item
            try:
                if self._writer is None and not self._open(frame):
                    break
                self._burn_timestamp(frame, ts)
                if (frame.shape[1], frame.shape[0]) != self._size:
                    frame = cv2.resize(frame, self._size)
                for _ in range(repeat):
                    self._writer.write(frame)
                    self.frames_written += 1
            except Exception as e:  # never take the app down over a recording problem
                self._fail(f"write failed: {e}")
                break
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None

    def _open(self, frame):
        h, w = frame.shape[:2]
        self._size = (w, h)
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, self._size)
        if not writer.isOpened():
            self._fail("could not open video writer (mp4v codec unavailable?)")
            return False
        self._writer = writer
        return True

    def _burn_timestamp(self, frame, ts):
        h = frame.shape[0]
        wall = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
        elapsed = int(ts - self._started_wall.timestamp())
        text = f"{wall}   T+{elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        cv2.rectangle(frame, (0, h - th - 14), (tw + 14, h), (0, 0, 0), -1)
        cv2.putText(frame, text, (7, h - 7), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _fail(self, message):
        self.last_error = message
        print("SessionRecorder:", message)
        if self._on_error:
            try:
                self._on_error(message)
            except Exception:
                pass
