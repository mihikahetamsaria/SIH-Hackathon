"""
"Log Files" toolbar dialog: pick a past session on the left, see its
written step-by-step record on the right, and play back the recording
for any step that has one (double-click a row, or select + Play).
"""

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QLabel,
)

import session_logger


class LogFilesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log Files")
        self.resize(860, 540)

        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(QLabel("Sessions"))
        self.session_list = QListWidget()
        self.session_list.setFixedWidth(200)
        for path in session_logger.list_sessions():
            item = QListWidgetItem(path.stem.replace("session_", ""))
            item.setData(Qt.UserRole, path)
            self.session_list.addItem(item)
        self.session_list.itemClicked.connect(self._load_session)
        left.addWidget(self.session_list, stretch=1)
        layout.addLayout(left)

        right = QVBoxLayout()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Step", "Status", "Start", "End", "Note"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.cellDoubleClicked.connect(lambda row, _col: self._play_row(row))
        right.addWidget(self.table, stretch=1)

        self.play_btn = QPushButton("Play recording for selected step")
        self.play_btn.clicked.connect(lambda: self._play_row(self.table.currentRow()))
        right.addWidget(self.play_btn)

        self.preview_video = QVideoWidget()
        self.preview_video.setFixedHeight(220)
        right.addWidget(self.preview_video)

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.preview_video)

        layout.addLayout(right, stretch=1)

        self._current_steps = []
        if self.session_list.count():
            self.session_list.setCurrentRow(0)
            self._load_session(self.session_list.item(0))
        else:
            right.addWidget(QLabel("No sessions logged yet - run the app once first."))

    def _load_session(self, item: QListWidgetItem):
        data = session_logger.load_session(item.data(Qt.UserRole))
        self._current_steps = data.get("steps", [])
        self.table.setRowCount(len(self._current_steps))
        for row, s in enumerate(self._current_steps):
            self.table.setItem(row, 0, QTableWidgetItem(s.get("name", "")))
            self.table.setItem(row, 1, QTableWidgetItem(s.get("status", "")))
            self.table.setItem(row, 2, QTableWidgetItem(s.get("ts_start") or "-"))
            self.table.setItem(row, 3, QTableWidgetItem(s.get("ts_end") or "-"))
            note = s.get("corrective_action") or ("has video" if s.get("video_path") else "")
            self.table.setItem(row, 4, QTableWidgetItem(note or ""))

    def _play_row(self, row: int):
        if row < 0 or row >= len(self._current_steps):
            return
        video_path = self._current_steps[row].get("video_path")
        if video_path and Path(video_path).exists():
            self.player.setSource(QUrl.fromLocalFile(str(video_path)))
            self.player.play()
        else:
            self.player.stop()
