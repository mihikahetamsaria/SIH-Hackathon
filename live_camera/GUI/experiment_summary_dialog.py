from datetime import datetime
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea, QWidget

class ExperimentSummaryDialog(QDialog):
    view_log = Signal()
    run_again = Signal()
    load_new = Signal()

    def __init__(self, experiment_name, steps, event_recordings, start_time=None, end_time=None, parent=None):
        super().__init__(parent)
        self.experiment_name = experiment_name or "Experiment"
        self.steps = steps or []
        self.event_recordings = event_recordings or []
        self.start_time = self._parse_timestamp(start_time) if start_time is not None else None
        self.end_time = self._parse_timestamp(end_time) if end_time is not None else datetime.now().timestamp()
        self.setWindowTitle("Experiment Summary")
        self.setModal(False)
        self.setMinimumSize(720, 620)
        self._build_ui()

    def _build_ui(self):
        theme = getattr(self.parent(), "theme", {}) if self.parent() else {}
        bg = theme.get("bg", theme.get("background", "#0b1118"))
        panel = theme.get("panel_bg", "#111923")
        text = theme.get("text", "#ffffff")
        muted = theme.get("muted_text", theme.get("muted", "#9aa7b5"))
        border = theme.get("border", "#263442")
        accent = theme.get("accent", "#55aaff")
        success = theme.get("success", "#43d17a")
        self.setStyleSheet(f"QDialog{{background:{bg};color:{text};}}QLabel{{color:{text};}}QFrame#card{{background:{panel};border:1px solid {border};border-radius:12px;}}QLabel#title{{font-size:24px;font-weight:800;}}QLabel#subtitle{{color:{muted};font-size:12px;}}QLabel#section{{font-size:13px;font-weight:800;}}QLabel#step{{font-size:13px;}}QLabel#duration{{color:{muted};font-size:12px;}}QPushButton{{background:{panel};color:{text};border:1px solid {border};border-radius:8px;padding:9px 16px;font-weight:700;}}QPushButton:hover{{border-color:{accent};}}QPushButton#primary{{background:{success};color:#07110b;border:none;}}")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(14)
        title = QLabel("EXPERIMENT SUMMARY")
        title.setObjectName("title")
        root.addWidget(title)
        completed = datetime.fromtimestamp(self.end_time).strftime("%d %b %Y, %H:%M:%S")
        subtitle = QLabel(f"{self.experiment_name}  •  Completed {completed}")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)
        status_card = QFrame()
        status_card.setObjectName("card")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(16, 14, 16, 14)
        status_layout.addWidget(QLabel("STATUS"))
        status = QLabel("✓ Experiment Completed")
        status.setStyleSheet(f"color:{success};font-size:16px;font-weight:800;")
        status_layout.addWidget(status)
        root.addWidget(status_card)
        step_card = QFrame()
        step_card.setObjectName("card")
        step_layout = QVBoxLayout(step_card)
        step_layout.setContentsMargins(16, 14, 16, 14)
        step_layout.addWidget(QLabel("STEP SUMMARY"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMaximumHeight(210)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 4, 0, 4)
        inner_layout.setSpacing(7)
        for step in self.steps:
            row = QHBoxLayout()
            step_name = step.get("name", f"Step {step.get('step_id', '')}")
            name = QLabel(f"✓  {step_name}")
            name.setObjectName("step")
            duration = QLabel(self._duration(step))
            duration.setObjectName("duration")
            duration.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(name, 1)
            row.addWidget(duration)
            inner_layout.addLayout(row)
        inner_layout.addStretch()
        scroll.setWidget(inner)
        step_layout.addWidget(scroll)
        root.addWidget(step_card)
        correct = sum(1 for x in self.event_recordings if str(x.get("status", "")).upper() == "CORRECT")
        wrong = sum(1 for x in self.event_recordings if str(x.get("status", "")).upper() in {"WRONG", "OUT_OF_ORDER", "UNEXPECTED", "SKIPPED"})
        total_duration = self._total_duration()
        perf_card = QFrame()
        perf_card.setObjectName("card")
        perf_layout = QVBoxLayout(perf_card)
        perf_layout.setContentsMargins(16, 14, 16, 14)
        perf_layout.addWidget(QLabel("PERFORMANCE"))
        completed_count = sum(1 for s in self.steps if s.get("status") == "done")
        values = (
            ("Steps completed", f"{completed_count} / {len(self.steps)}"),
            ("Correct attempts", str(correct)),
            ("Wrong / out-of-order", str(wrong)),
            ("Event recordings", str(len(self.event_recordings))),
            ("Total experiment time", self._format_seconds(total_duration)),
        )
        for label, value in values:
            row = QHBoxLayout()
            a = QLabel(label)
            a.setObjectName("step")
            b = QLabel(value)
            b.setObjectName("duration")
            b.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(a, 1)
            row.addWidget(b)
            perf_layout.addLayout(row)
        root.addWidget(perf_card)
        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        log_btn = QPushButton("VIEW LOG")
        log_btn.clicked.connect(self.view_log.emit)
        again_btn = QPushButton("RUN AGAIN")
        again_btn.setObjectName("primary")
        again_btn.clicked.connect(self.run_again.emit)
        new_btn = QPushButton("LOAD NEW EXPERIMENT")
        new_btn.clicked.connect(self.load_new.emit)
        buttons.addWidget(log_btn)
        buttons.addStretch()
        buttons.addWidget(again_btn)
        buttons.addWidget(new_btn)
        root.addLayout(buttons)

    def _duration(self, step):
        start = step.get("ts_start")
        end = step.get("ts_end")
        if start is None or end is None:
            return "--:--"
        return self._format_seconds(max(0, self._parse_timestamp(end) - self._parse_timestamp(start)))

    def _total_duration(self):
        run_start = self.start_time
        run_end = self.end_time
        if run_start is None:
            starts = [self._parse_timestamp(s["ts_start"]) for s in self.steps if s.get("ts_start") is not None]
            run_start = min(starts) if starts else None
        if run_start is None or run_end is None:
            return 0
        return max(0, run_end - run_start)

    @staticmethod
    def _parse_timestamp(value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            return value.timestamp()
        return datetime.fromisoformat(str(value)).timestamp()

    @staticmethod
    def _format_seconds(seconds):
        seconds = max(0, int(round(seconds)))
        return f"{seconds // 60}:{seconds % 60:02d}"
