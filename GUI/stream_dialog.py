"""
Small settings dialog for the ground-control mirror stream. Follows the
same "click a nav button, get a dialog" pattern as LogFilesDialog /
TaskTableDialog / ThemeDialog.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QPushButton,
)


class StreamDialog(QDialog):
    def __init__(self, streamer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ground Control Stream")
        self.setMinimumWidth(360)
        self.streamer = streamer

        layout = QVBoxLayout(self)

        info = QLabel(
            "Optional low-bandwidth mirror of the live feed and current "
            "step status, sent over UDP. Local recording and the step log "
            "work exactly the same whether this is on or off - the app "
            "still runs fully offline."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("Ground station IP:"))
        self.ip_edit = QLineEdit(streamer.ip or "")
        self.ip_edit.setPlaceholderText("e.g. 192.168.1.50")
        ip_row.addWidget(self.ip_edit)
        layout.addLayout(ip_row)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        self.port_edit = QLineEdit(str(streamer.port) if streamer.port else "5599")
        port_row.addWidget(self.port_edit)
        layout.addLayout(port_row)

        self.enable_box = QCheckBox("Enable streaming")
        self.enable_box.setChecked(streamer.enabled)
        layout.addWidget(self.enable_box)

        note = QLabel(
            "If the network is unreachable, frames are silently dropped - "
            "this never blocks or crashes the app."
        )
        note.setWordWrap(True)
        note.setStyleSheet("opacity: 0.7; font-size: 11px;")
        layout.addWidget(note)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def _save(self):
        ip = self.ip_edit.text().strip()
        try:
            port = int(self.port_edit.text().strip())
        except ValueError:
            port = 5599

        if ip:
            self.streamer.configure(ip, port)
        self.streamer.set_enabled(self.enable_box.isChecked() and bool(ip))
        self.accept()
