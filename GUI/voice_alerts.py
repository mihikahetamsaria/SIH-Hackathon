"""
voice_alerts.py
Non-blocking spoken alerts for the HAR On-board Experiment Validation GUI.

The Qt GUI thread never calls pyttsx3.runAndWait().
A dedicated worker thread owns the TTS engine and processes queued messages.

Behavior:
    - Import failure -> TTS unavailable, GUI still works
    - Engine initialization failure -> TTS unavailable, GUI still works
    - speak() is always non-blocking
    - shutdown() is safe to call multiple times
"""

from __future__ import annotations

import queue
import threading
from typing import Optional


try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    PYTTSX3_AVAILABLE = False


class VoiceAlertManager:
    """
    Background TTS manager.

    Usage:
        self.voice_alerts = VoiceAlertManager()

        self.voice_alerts.speak(
            "Step 3 was skipped. Please go back and align the sample tray."
        )

        self.voice_alerts.shutdown()
    """

    def __init__(self, rate: int = 175):
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._rate = int(rate)

        # Actual runtime state.
        self._ready = threading.Event()
        self._initialization_failed = threading.Event()
        self._shutdown_requested = False

        if not PYTTSX3_AVAILABLE:
            return

        self._thread = threading.Thread(
            target=self._worker,
            name="VoiceAlertWorker",
            daemon=True,
        )
        self._thread.start()

        # Wait briefly for the worker to attempt engine initialization.
        #
        # This does NOT block the GUI for the duration of speech. It only
        # waits for startup to determine whether a TTS backend is usable.
        self._ready.wait(timeout=3.0)

    @property
    def available(self) -> bool:
        """
        True only if pyttsx3 successfully initialized its engine.
        """
        return (
            PYTTSX3_AVAILABLE
            and self._thread is not None
            and self._thread.is_alive()
            and self._ready.is_set()
            and not self._initialization_failed.is_set()
            and not self._shutdown_requested
        )

    def _worker(self):
        """
        Worker thread owns the pyttsx3 engine.
        """

        engine = None

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)

            # Engine is genuinely ready now.
            self._ready.set()

        except Exception:
            # Do not let a missing/broken OS speech backend kill the GUI.
            self._initialization_failed.set()
            self._ready.set()

            # Drain the queue until shutdown so callers can safely continue
            # calling speak() without accumulating blocked worker state.
            while True:
                item = self._queue.get()

                if item is None:
                    return

        # Normal TTS processing loop.
        while True:
            text = self._queue.get()

            if text is None:
                break

            if not text.strip():
                continue

            try:
                engine.say(text)
                engine.runAndWait()

            except Exception:
                # A failed utterance should not terminate the worker.
                continue

        # Best-effort engine cleanup.
        try:
            if engine is not None:
                engine.stop()
        except Exception:
            pass

    def speak(self, text: str):
        """
        Non-blocking.

        The text is queued immediately and the GUI thread returns.
        """

        if not text:
            return

        if not isinstance(text, str):
            text = str(text)

        if not self.available:
            return

        self._queue.put(text)

    def shutdown(self):
        """
        Safely stop the worker thread.

        Safe to call more than once.
        """

        if self._shutdown_requested:
            return

        self._shutdown_requested = True

        if self._thread is None:
            return

        # Wake the worker.
        self._queue.put(None)

        # Do not wait indefinitely during application shutdown.
        self._thread.join(timeout=3.0)

        self._thread = None