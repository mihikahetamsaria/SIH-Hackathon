"""
Shown automatically when a step is marked 'skipped' or 'out_of_order'
(Section 6 / new feature: mistake review).

Flow: play the clip recorded during the astronaut's own attempt at this
step (with the corrective message on screen) -> "Show correct way" button
-> play the reference clip for the correct technique. Falls back to a
text-only message if either clip is missing, rather than crashing.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton


class MistakeReviewDialog(QDialog):
    def __init__(
        self,
        step_name: str,
        message: str,
        your_clip_path: Optional[Path],
        reference_clip_path: Optional[Path],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Mistake review - {step_name}")
        self.resize(720, 520)
        self.setModal(False)

        self._step_name = step_name
        self._message = message
        self.your_clip_path = your_clip_path
        self.reference_clip_path = reference_clip_path

        layout = QVBoxLayout(self)

        self.headline = QLabel()
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout.addWidget(self.headline)

        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget, stretch=1)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)

        btn_row = QHBoxLayout()
        self.replay_btn = QPushButton("Replay")
        self.replay_btn.clicked.connect(self._replay)
        self.next_btn = QPushButton("Show correct way \u2192")
        self.next_btn.clicked.connect(self._show_reference)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.replay_btn)
        btn_row.addWidget(self.next_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._load_your_clip()

    def _load_your_clip(self):
        have_yours = bool(self.your_clip_path and self.your_clip_path.exists())
        have_reference = bool(
            self.reference_clip_path and self.reference_clip_path.exists()
        )

        if have_yours:
            self.headline.setText(
                f"What went wrong on '{self._step_name}':\n{self._message}"
            )
            self.player.setSource(QUrl.fromLocalFile(str(self.your_clip_path)))
            self.player.play()
            self.next_btn.setEnabled(have_reference)
        else:
            # No recording of the mistake - skip straight to the reference clip.
            self.headline.setText(
                f"What went wrong on '{self._step_name}':\n{self._message}\n"
                "(No recording of your attempt was available.)"
            )
            self.replay_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self._show_reference()

    def _show_reference(self):
        self.next_btn.setEnabled(False)
        if self.reference_clip_path and self.reference_clip_path.exists():
            self.headline.setText(f"Correct way to do '{self._step_name}':")
            self.player.setSource(QUrl.fromLocalFile(str(self.reference_clip_path)))
            self.player.play()
            self.replay_btn.setEnabled(True)
        else:
            self.headline.setText(
                f"Correct way to do '{self._step_name}':\n"
                f"(No reference clip found - add one at "
                f"data/reference_clips/step_<id>.mp4)"
            )
            self.replay_btn.setEnabled(False)

    def _replay(self):
        self.player.setPosition(0)
        self.player.play()