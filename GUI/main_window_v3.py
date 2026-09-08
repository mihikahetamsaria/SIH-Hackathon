"""
Parth's track: GUI & Logging - v3 layout.

This restyles the app to match the "AI GUIDED ASSISTANT" reference mockup,
built on top of the same data sources as v1/v2:
    - data/experiment_definition.json  (Section 5a)
    - data/step_record_log.json        (Section 5b, polled every second)

Nothing about the schema or the polling model changed - this file is a
new *view* on the same data. Mihika's module (or placeholder_simulator.py)
doesn't need to change to work with this version.

Layout:
    - Top:    nav bar (Projects / Logs / Analytics / Settings)
    - Left:   Guided Task Procedure list (StepRow widgets from widgets_v3.py)
    - Center: video feed with a floating AI-message speech bubble, plus a
              live elapsed-time badge burned into the bottom-right corner
    - Right:  Preview Area (reference clip player + quick-demo button)
    - Bottom: Immediate Guidance Dashboard (current action, circular
              progress ring, step-completion dots, Repeat Step)

On a skipped/out_of_order step, this also opens v2's MistakeReviewDialog
(your recorded attempt -> "Show correct way" -> the reference clip) - the
v3 redesign didn't replace that feature, it just wasn't wired up yet.

Run with: python main_v3.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsDropShadowEffect,
)

from log_reader import load_step_record_log, load_experiment_definition
from theme import THEME, build_stylesheet, hex_to_bgr, get_status_colors
import session_logger
from log_files_dialog import LogFilesDialog
from task_table_dialog import TaskTableDialog
from theme_dialog import ThemeDialog
from mistake_review_dialog import MistakeReviewDialog
from widgets_v3 import CircularProgress, StatusDot, StepRow, make_icon
from voice_alerts import VoiceAlertManager
from stream_sender import UDPFrameStreamer
from stream_dialog import StreamDialog

DATA_DIR = Path(__file__).parent / "data"
STEP_LOG_PATH = DATA_DIR / "step_record_log.json"
EXPERIMENT_DEF_PATH = DATA_DIR / "experiment_definition.json"
REFERENCE_CLIPS_DIR = DATA_DIR / "reference_clips"
RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

LOG_POLL_MS = 1000
VIDEO_POLL_MS = 33
RECORDING_FPS = 20.0


class VideoOverlayContainer(QWidget):
    """Holds the video QLabel plus a floating speech-bubble label on top of
    it (bottom-right, like the mockup's "I see the RED BOX..." bubble)."""

    def __init__(self, video_label, bubble_label, parent=None):
        super().__init__(parent)
        self.video_label = video_label
        self.bubble_label = bubble_label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(video_label)

        bubble_label.setParent(self)
        self._reposition()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition()

    def _reposition(self):
        margin = 18
        bubble = self.bubble_label
        bubble_w = min(360, self.width() - 2 * margin)
        bubble.setFixedWidth(bubble_w)
        bubble.adjustSize()
        bubble.move(self.width() - bubble.width() - margin, margin)


class MainWindowV3(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Guided Assistant - HAR On-board Experiment Validation")
        self.resize(1500, 880)

        self.theme = dict(THEME)
        self.status_colors = get_status_colors(self.theme)

        self.experiment_def = load_experiment_definition(EXPERIMENT_DEF_PATH) or {}
        self.steps_def = self.experiment_def.get("steps", [])
        self.step_lookup = {s["step_id"]: s for s in self.steps_def}

        self._last_status_by_step = {}
        self._step_rows = {}
        self._status_dots = {}
        self._step_start_times = {}
        self._overlay_text = "Waiting for experiment to start..."
        self._overlay_status = "pending"
        self._current_timer_text = "--:--"
        self._current_step_name = "-"

        self.active_writers = {}
        self.step_recordings = {}
        self.session_started = datetime.now()
        self.session_archive_path = session_logger.new_session_path(self.session_started)
        self._theme_dialog = None
        self._stream_dialog = None

        # Spoken alerts run on a background thread (see voice_alerts.py) so
        # a 2-4s sentence never freezes the video feed.
        self.voice_alerts = VoiceAlertManager()

        # Optional low-bandwidth UDP mirror to ground control. Disabled by
        # default and a no-op until configured + enabled via the STREAM
        # dialog - the app is fully offline-capable either way.
        self.stream = UDPFrameStreamer()

        self._build_header()
        self._build_body()

        self.capture = cv2.VideoCapture(0)
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._update_video_frame)
        self.video_timer.start(VIDEO_POLL_MS)

        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self._poll_step_log)
        self.log_timer.start(LOG_POLL_MS)
        self._poll_step_log()

    def _add_shadow(self, widget, blur_radius=18, alpha=70, y_offset=5):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur_radius)
        shadow.setColor(QColor(0, 0, 0, alpha))
        shadow.setOffset(0, y_offset)
        widget.setGraphicsEffect(shadow)

    # ---------------------------------------------------------------- header

    def _build_header(self):
        header = QWidget()
        header.setObjectName("headerBar")
        header.setFixedHeight(56)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)

        for label, handler in (
            ("PROJECTS", None),
            ("LOGS", lambda: LogFilesDialog(self).exec()),
            ("ANALYTICS", lambda: TaskTableDialog(self).exec()),
            ("STREAM", self._open_stream_dialog),
            ("SETTINGS", self._open_theme_dialog),
        ):
            btn = QPushButton(label)
            btn.setObjectName("navLink")
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
            if handler:
                btn.clicked.connect(handler)
            layout.addWidget(btn)

        layout.addStretch()

        self.header = header

    def _open_theme_dialog(self):
        dialog = ThemeDialog(self.theme, self._on_theme_changed, parent=self)
        dialog.show()
        self._theme_dialog = dialog

    def _open_stream_dialog(self):
        dialog = StreamDialog(self.stream, parent=self)
        dialog.exec()

    def _on_theme_changed(self, theme: dict):
        self.theme = theme
        self.status_colors = get_status_colors(self.theme)
        QApplication.instance().setStyleSheet(build_stylesheet(self.theme) + self._extra_stylesheet())
        self.video_label.setStyleSheet(f"background-color: #000; border-radius: 10px;")
        self._refresh_all_step_rows()

    # ------------------------------------------------------------------ body

    def _build_body(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.header)

        body_row = QHBoxLayout()
        body_row.setContentsMargins(16, 16, 16, 8)
        body_row.setSpacing(16)
        body_row.addWidget(self._build_step_panel(), stretch=3)
        body_row.addWidget(self._build_video_panel(), stretch=5)
        body_row.addWidget(self._build_preview_panel(), stretch=2)
        root.addLayout(body_row, stretch=1)

        root.addWidget(self._build_dashboard())

        self.setCentralWidget(central)
        QApplication.instance().setStyleSheet(build_stylesheet(self.theme) + self._extra_stylesheet())

    def _extra_stylesheet(self) -> str:
        t = self.theme
        return f"""
        #headerBar {{ background-color: {t['panel_bg']}; border-bottom: 1px solid {t['border']}; }}
        QPushButton#navLink {{
            background: transparent; border: none; font-weight: 600; padding: 6px 10px;
            color: {t['text']};
        }}
        QPushButton#navLink:hover {{ color: {t['accent']}; }}
        #panelCard {{ background-color: {t['panel_bg']}; border-radius: 12px; }}
        #dashboardBar {{ background-color: {t['panel_bg']}; border-top: 1px solid {t['border']}; }}
        """

    def _build_step_panel(self):
        card = QWidget()
        card.setObjectName("panelCard")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(14, 14, 14, 14)

        title = QLabel("GUIDED TASK PROCEDURE")
        title.setObjectName("sectionTitle")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(10)

        for step in self.steps_def:
            row = StepRow(step, self.theme)
            self._step_rows[step["step_id"]] = row
            inner_layout.addWidget(row)
        inner_layout.addStretch()

        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)
        self._add_shadow(card)
        return card

    def _build_video_panel(self):
        card = QWidget()
        card.setObjectName("panelCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)

        self.video_label = QLabel("No camera feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(600, 480)
        self.video_label.setStyleSheet("background-color: #000; border-radius: 10px;")

        self.ai_bubble = QLabel("")
        self.ai_bubble.setObjectName("bubbleLabel")
        self.ai_bubble.setWordWrap(True)
        self.ai_bubble.hide()

        container = VideoOverlayContainer(self.video_label, self.ai_bubble)
        layout.addWidget(container)
        self._add_shadow(card)
        self.video_container = container
        return card

    def _build_preview_panel(self):
        card = QWidget()
        card.setObjectName("panelCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)

        title = QLabel("PREVIEW AREA")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.preview_video = QVideoWidget()
        self.preview_video.setMinimumHeight(200)
        layout.addWidget(self.preview_video, stretch=1)

        self.preview_player = QMediaPlayer(self)
        self.preview_audio = QAudioOutput(self)
        self.preview_player.setAudioOutput(self.preview_audio)
        self.preview_player.setVideoOutput(self.preview_video)

        self.preview_label = QLabel("Select a step to preview its reference clip")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        controls = QHBoxLayout()
        play_btn = QPushButton("PLAY")
        pause_btn = QPushButton("PAUSE")
        play_btn.clicked.connect(self.preview_player.play)
        pause_btn.clicked.connect(self.preview_player.pause)
        controls.addWidget(play_btn)
        controls.addWidget(pause_btn)
        layout.addLayout(controls)

        demo_btn = QPushButton("Watch a Quick Demo")
        demo_btn.setObjectName("demoBtn")
        demo_btn.clicked.connect(self._play_current_reference_clip)
        layout.addWidget(demo_btn)

        self._add_shadow(card)
        return card

    def _build_dashboard(self):
        bar = QWidget()
        bar.setObjectName("dashboardBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 14, 24, 14)
        layout.setSpacing(24)

        left = QVBoxLayout()
        self.current_action_label = QLabel("Current Action: -")
        self.current_action_label.setStyleSheet("font-size: 20px; font-weight: 800;")
        left.addWidget(self.current_action_label)
        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        left.addWidget(self.validation_label)
        layout.addLayout(left, stretch=3)

        mid = QVBoxLayout()
        self.completion_label = QLabel("Step Completion: 0%")
        self.completion_label.setStyleSheet("font-weight: 700;")
        mid.addWidget(self.completion_label)
        dots_row = QHBoxLayout()
        for step in self.steps_def:
            dot = StatusDot()
            self._status_dots[step["step_id"]] = dot
            dots_row.addWidget(dot)
        mid.addLayout(dots_row)
        self.task_progress_label = QLabel("Task Progress: - of -")
        self.task_progress_label.setStyleSheet("font-size: 12px; opacity: 0.8;")
        mid.addWidget(self.task_progress_label)
        layout.addLayout(mid, stretch=3)

        btn_col = QVBoxLayout()
        self.repeat_btn = QPushButton("REPEAT STEP")
        self.repeat_btn.clicked.connect(self._on_repeat_step)
        btn_col.addWidget(self.repeat_btn)
        btn_col.addStretch()
        layout.addLayout(btn_col, stretch=2)

        self.progress_ring = CircularProgress()
        self.progress_ring.set_colors(self.theme["border"], self.theme["success"], self.theme["text"])
        layout.addWidget(self.progress_ring, stretch=1)

        self._add_shadow(bar, blur_radius=14, alpha=50, y_offset=-2)
        return bar

    # --------------------------------------------------------------- video

    def _update_video_frame(self):
        if not self.capture or not self.capture.isOpened():
            return
        ok, frame = self.capture.read()
        if not ok:
            return

        for writer in self.active_writers.values():
            writer.write(frame)

        self._draw_overlay(frame)

        # Best-effort mirror to ground control - downscaled/compressed,
        # rate-limited, and a total no-op when disabled/unreachable. Never
        # touches the local recording/logging path above.
        self.stream.maybe_send(
            frame,
            {
                "step": self._current_step_name,
                "status": self._overlay_status,
                "elapsed": self._current_timer_text,
                "ts": datetime.now().isoformat(timespec="seconds"),
            },
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _draw_overlay(self, frame):
        color = hex_to_bgr(self.status_colors.get(self._overlay_status, self.theme["pending"]))
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)
        self._draw_timer_badge(frame, color)

    def _draw_timer_badge(self, frame, color):
        text = self._current_timer_text
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.9, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        pad = 12
        x2, y2 = w - 16, h - 16
        x1, y1 = x2 - tw - 2 * pad, y2 - th - 2 * pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, text, (x1 + pad, y2 - pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    # ----------------------------------------------------------- log poll

    def _poll_step_log(self):
        data = load_step_record_log(STEP_LOG_PATH)
        if data is None:
            return

        steps = data.get("steps", [])
        current_id = data.get("current_expected_step_id")
        current_step = self.step_lookup.get(current_id, {})
        total = len(steps) if steps else 1
        done_count = sum(1 for s in steps if s["status"] == "done")
        step_index = next((i for i, s in enumerate(steps) if s["step_id"] == current_id), len(steps) - 1)

        self._current_step_name = current_step.get("name", "-")
        self.current_action_label.setText(f"Current Action: {self._current_step_name}")
        pct = int(done_count / total * 100)
        self.completion_label.setText(f"Step Completion: {pct}%")
        self.task_progress_label.setText(f"Task Progress: Step {step_index + 1} of {total}")
        self.progress_ring.set_value(pct)

        active_record = next((s for s in steps if s["status"] == "in_progress"), None)
        if active_record is None:
            active_record = next((s for s in steps if s["step_id"] == current_id), None)
        self._current_timer_text = self._timer_text_for(active_record) if active_record else "--:--"

        self._update_step_rows(steps, current_id)
        self._update_status_dots(steps, current_id)
        self._handle_status_transitions(steps)
        self._persist_archive(steps)

    def _update_step_rows(self, steps, current_id):
        for s in steps:
            row = self._step_rows.get(s["step_id"])
            if not row:
                continue
            is_current = s["step_id"] == current_id or s["status"] == "in_progress"
            timer_text = self._timer_text_for(s)
            row.set_state(s["status"], is_current, timer_text, self.status_colors)
            if is_current:
                hint = self.step_lookup.get(s["step_id"], {}).get(
                    "corrective_message", "Follow the on-screen guidance for this step."
                )
                row.set_hint(hint)

    def _timer_text_for(self, step_record: dict) -> str:
        ts_start = step_record.get("ts_start")
        ts_end = step_record.get("ts_end")
        if step_record["status"] == "in_progress" and ts_start:
            elapsed = int(time.time() - self._parse_ts(ts_start))
            return f"0:{elapsed:02d}" if elapsed < 60 else f"{elapsed // 60}:{elapsed % 60:02d}"
        if ts_start and ts_end:
            dur = int(self._parse_ts(ts_end) - self._parse_ts(ts_start))
            return f"0:{dur:02d}" if dur < 60 else f"{dur // 60}:{dur % 60:02d}"
        return "--:--"

    @staticmethod
    def _parse_ts(ts_str: str) -> float:
        try:
            return datetime.fromisoformat(ts_str).timestamp()
        except (ValueError, TypeError):
            return time.time()

    def _update_status_dots(self, steps, current_id):
        for s in steps:
            dot = self._status_dots.get(s["step_id"])
            if not dot:
                continue
            color = self.status_colors.get(s["status"], self.theme["pending"])
            dot.set_status(s["status"], color, self.theme["border"])

    def _refresh_all_step_rows(self):
        # Re-applies current colors/status after a theme change.
        data = load_step_record_log(STEP_LOG_PATH)
        if data:
            self._update_step_rows(data.get("steps", []), data.get("current_expected_step_id"))
            self._update_status_dots(data.get("steps", []), data.get("current_expected_step_id"))

    def _persist_archive(self, steps):
        enriched = []
        for s in steps:
            video_path = self.step_recordings.get(s["step_id"])
            enriched.append({**s, "video_path": str(video_path) if video_path else None})
        session_logger.save_session(self.session_archive_path, self.session_started, enriched)

    def _handle_status_transitions(self, steps):
        for s in steps:
            step_id = s["step_id"]
            status = s["status"]
            prev = self._last_status_by_step.get(step_id)
            if status != prev:
                self._on_status_changed(s, prev)
            self._last_status_by_step[step_id] = status

    def _on_status_changed(self, step: dict, prev_status):
        step_id = step["step_id"]
        status = step["status"]

        if status == "in_progress":
            self._start_recording(step_id)
        elif prev_status == "in_progress" and status != "in_progress":
            self._stop_recording(step_id)

        if status == "in_progress":
            self._overlay_status = "in_progress"
            self._show_ai_bubble(f"I see the {step.get('name', 'step')}. Go ahead.", status)
            self.validation_label.setText(f"Validation: In progress. {step['name']}.")
        elif status == "done":
            self._overlay_status = "done"
            QApplication.beep()
            self._show_ai_bubble(f"Nicely done: {step['name']}.", status)
            self.validation_label.setText(f"Validation: Step complete. {step['name']}.")
        elif status in ("skipped", "out_of_order"):
            message = step.get("corrective_action") or self.step_lookup.get(
                step_id, {}
            ).get("corrective_message", "Please check the current step.")
            self._overlay_status = status
            self._show_ai_bubble(message, status)
            self.validation_label.setText(f"Validation: {status.replace('_', ' ').title()}. {message}")
            self._trigger_voice_alert(step, status, message)
            self._auto_show_reference(step_id)
            self._open_mistake_review(step, message)

    def _trigger_voice_alert(self, step: dict, status: str, message: str):
        """Beep-beep, then (half a second later, off the GUI thread) speak
        what went wrong. The beeps are an immediate, language-independent
        cue; the spoken sentence carries the actual corrective content the
        challenge spec asks for."""
        step_name = step.get("name", f"step {step.get('step_id', '')}")
        status_phrase = "skipped" if status == "skipped" else "performed out of order"

        QApplication.beep()
        QTimer.singleShot(180, QApplication.beep)
        QTimer.singleShot(
            500,
            lambda: self.voice_alerts.speak(f"{step_name} was {status_phrase}. {message}"),
        )

    def _open_mistake_review(self, step: dict, message: str):
        step_id = step["step_id"]
        your_clip = self.step_recordings.get(step_id)
        reference_clip = REFERENCE_CLIPS_DIR / f"step_{step_id}.mp4"
        dialog = MistakeReviewDialog(
            step_name=step.get("name", f"Step {step_id}"),
            message=message,
            your_clip_path=your_clip,
            reference_clip_path=reference_clip if reference_clip.exists() else None,
            parent=self,
        )
        dialog.show()
        self._mistake_dialog = dialog  # keep alive while open

    def _show_ai_bubble(self, text: str, status: str):
        color = self.status_colors.get(status, self.theme["accent"])
        self.ai_bubble.setText(text)
        self.ai_bubble.setStyleSheet(
            f"background-color: rgba(20,20,20,220); color: white; border: 1px solid {color}; "
            f"border-radius: 12px; padding: 12px; font-size: 13px;"
        )
        self.ai_bubble.show()
        self.ai_bubble.adjustSize()
        QTimer.singleShot(4000, self.ai_bubble.hide)

    # ------------------------------------------------------------ recording

    def _start_recording(self, step_id):
        if not self.capture or not self.capture.isOpened():
            return
        w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        path = RECORDINGS_DIR / f"step_{step_id}_{int(time.time())}.mp4"
        writer = cv2.VideoWriter(str(path), fourcc, RECORDING_FPS, (w, h))
        if not writer.isOpened():
            return
        self.active_writers[step_id] = writer
        self.step_recordings[step_id] = path

    def _stop_recording(self, step_id):
        writer = self.active_writers.pop(step_id, None)
        if writer:
            writer.release()

    # -------------------------------------------------------- preview panel

    def _current_step_id(self):
        data = load_step_record_log(STEP_LOG_PATH)
        return data.get("current_expected_step_id") if data else None

    def _auto_show_reference(self, step_id: int):
        self._load_reference_clip(step_id)

    def _play_current_reference_clip(self):
        step_id = self._current_step_id()
        if step_id is not None:
            self._load_reference_clip(step_id)
            self.preview_player.play()

    def _load_reference_clip(self, step_id: int):
        step = self.step_lookup.get(step_id, {})
        clip_path = REFERENCE_CLIPS_DIR / f"step_{step_id}.mp4"
        if clip_path.exists():
            self.preview_label.setText(f"Correct way: {step.get('name', '')}")
            self.preview_player.setSource(QUrl.fromLocalFile(str(clip_path)))
        else:
            self.preview_label.setText(
                f"No clip yet for: {step.get('name', '')}\n"
                f"(add data/reference_clips/step_{step_id}.mp4)"
            )
            self.preview_player.stop()

    def _on_repeat_step(self):
        step_id = self._current_step_id()
        if step_id is not None:
            self._load_reference_clip(step_id)
            self.preview_player.setPosition(0)
            self.preview_player.play()

    # ---------------------------------------------------------- lifecycle

    def closeEvent(self, event):
        if self.capture:
            self.capture.release()
        for writer in self.active_writers.values():
            writer.release()
        self.voice_alerts.shutdown()
        self.stream.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindowV3()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()