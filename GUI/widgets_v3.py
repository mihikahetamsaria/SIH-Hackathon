"""
Small reusable widgets for main_window_v3.py.

- CircularProgress: the "66%" ring in the Immediate Guidance Dashboard.
  Drawn by hand with QPainter - a single gauge like this doesn't need a
  charting library, it's ~30 lines of arc-drawing.

- StepRow: one row in the left "Guided Task Procedure" list. Not a
  QListWidgetItem - it's a full QWidget (icon + title + timer chip, with
  an expandable AI-hint section underneath) so it can match the mockup's
  layout, which plain list items can't do on their own.

- StatusDot: one dot in the step-completion strip on the bottom bar.

Icons use qtawesome if it's installed (`pip install qtawesome`), and fall
back to plain Unicode glyphs if it isn't, so nothing crashes on a machine
that hasn't added the new dependency yet.
"""

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QFrame,
)

try:
    import qtawesome as qta
    HAVE_QTAWESOME = True
except ImportError:
    HAVE_QTAWESOME = False


def make_icon(name_fallback_char: str, qta_name: str, color: str, size: int = 16):
    """Returns a QIcon via qtawesome if available, else None - callers
    should fall back to a plain-text glyph (name_fallback_char) on None."""
    if HAVE_QTAWESOME:
        try:
            return qta.icon(qta_name, color=color)
        except Exception:
            return None
    return None


class CircularProgress(QWidget):
    """A ring gauge: track color behind, progress arc on top, % in the middle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        self._track_color = QColor("#333333")
        self._progress_color = QColor("#2ecc71")
        self._text_color = QColor("#e0e0e0")
        self.setMinimumSize(96, 96)

    def set_value(self, value: int):
        self._value = max(0, min(100, int(value)))
        self.update()

    def set_colors(self, track_hex: str, progress_hex: str, text_hex: str):
        self._track_color = QColor(track_hex)
        self._progress_color = QColor(progress_hex)
        self._text_color = QColor(text_hex)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        side = min(self.width(), self.height())
        pen_width = max(6, side // 12)
        margin = pen_width
        rect = QRectF(
            (self.width() - side) / 2 + margin,
            (self.height() - side) / 2 + margin,
            side - 2 * margin,
            side - 2 * margin,
        )

        track_pen = QPen(self._track_color, pen_width, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(rect, 0, 360 * 16)

        if self._value > 0:
            progress_pen = QPen(self._progress_color, pen_width, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(progress_pen)
            span = int(360 * (self._value / 100) * 16)
            painter.drawArc(rect, 90 * 16, -span)

        painter.setPen(self._text_color)
        font = QFont()
        font.setBold(True)
        font.setPointSize(max(10, side // 8))
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, f"{self._value}%")
        painter.end()


class StatusDot(QLabel):
    """One dot in the step-completion strip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self.set_status("pending", "#7a7a7a", "#333333")

    def set_status(self, status: str, color_hex: str, border_hex: str):
        filled = status in ("done", "in_progress", "skipped", "out_of_order")
        bg = color_hex if filled else "transparent"
        self.setStyleSheet(
            f"background-color: {bg}; border: 2px solid {border_hex if filled else color_hex}; "
            f"border-radius: 7px;"
        )


class StepRow(QFrame):
    """
    One row of the Guided Task Procedure list:
        [check/number icon]  Step name          [timer chip]
        (only when current) AI Hint: ...   DETAILS (expand) v
    """

    def __init__(self, step: dict, theme: dict, parent=None):
        super().__init__(parent)
        self.step = step
        self.theme = theme
        self.setObjectName("stepRow")
        self.setFrameShape(QFrame.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)

        top_row = QHBoxLayout()
        self.icon_label = QLabel()
        self.icon_label.setFixedWidth(22)
        top_row.addWidget(self.icon_label)

        self.title_label = QLabel(f"STEP {step['step_id']}: {step['name'].upper()}")
        self.title_label.setStyleSheet("font-weight: 700; font-size: 13px;")
        top_row.addWidget(self.title_label, stretch=1)

        self.timer_chip = QLabel("--:--")
        self.timer_chip.setStyleSheet(
            "padding: 2px 8px; border-radius: 8px; font-size: 12px; font-weight: 600;"
        )
        top_row.addWidget(self.timer_chip)
        outer.addLayout(top_row)

        self.hint_area = QWidget()
        hint_layout = QVBoxLayout(self.hint_area)
        hint_layout.setContentsMargins(30, 4, 0, 0)
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("font-size: 12px;")
        hint_layout.addWidget(self.hint_label)

        self.details_btn = QToolButton()
        self.details_btn.setText("DETAILS")
        self.details_btn.setCheckable(True)
        self.details_btn.setArrowType(Qt.DownArrow)
        self.details_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.details_btn.toggled.connect(self._toggle_details)
        hint_layout.addWidget(self.details_btn)

        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        self.details_label.setVisible(False)
        self.details_label.setStyleSheet("font-size: 12px; opacity: 0.85;")
        hint_layout.addWidget(self.details_label)

        outer.addWidget(self.hint_area)
        self.hint_area.setVisible(False)

        self.set_details_text(
            f"Object: {step.get('expected_object', '-')} &nbsp;|&nbsp; "
            f"Action: {step.get('expected_action', '-')} &nbsp;|&nbsp; "
            f"Hand: {step.get('expected_hand', '-')}"
        )

    def _toggle_details(self, checked: bool):
        self.details_label.setVisible(checked)
        self.details_btn.setArrowType(Qt.UpArrow if checked else Qt.DownArrow)

    def set_hint(self, text: str):
        self.hint_label.setText(f"<b>AI Hint:</b> {text}")

    def set_details_text(self, text: str):
        self.details_label.setText(text)

    def set_state(self, status: str, is_current: bool, timer_text: str, colors: dict):
        self.hint_area.setVisible(is_current)

        # `colors` here is status_colors: keyed by status name (done,
        # in_progress, skipped, out_of_order, pending) - not by
        # success/error/accent like the theme dict is. Look icons up by
        # status, and pull the "current step" border straight from theme.
        icon = None
        if status == "done":
            icon = make_icon("\u2713", "fa5s.check-circle", colors["done"])
            fallback = "\u2713"
        elif status in ("skipped", "out_of_order"):
            icon = make_icon("!", "fa5s.exclamation-circle", colors[status])
            fallback = "!"
        elif status == "in_progress":
            icon = make_icon("\u25b6", "fa5s.hand-paper", colors["in_progress"])
            fallback = "\u25b6"
        else:
            icon = make_icon("\u25cb", "fa5s.circle", colors["pending"])
            fallback = "\u25cb"

        if icon is not None:
            self.icon_label.setPixmap(icon.pixmap(18, 18))
        else:
            self.icon_label.setText(fallback)
            self.icon_label.setStyleSheet(f"color: {colors.get(status, colors['pending'])};")

        border = self.theme["accent"] if is_current else "transparent"
        bg = self.theme["panel_bg"]
        self.setStyleSheet(
            f"#stepRow {{ background-color: {bg}; border: 2px solid {border}; "
            f"border-radius: 10px; }}"
        )

        self.timer_chip.setText(timer_text)
        chip_color = colors.get(status, colors["pending"])
        self.timer_chip.setStyleSheet(
            f"padding: 2px 8px; border-radius: 8px; font-size: 12px; font-weight: 600; "
            f"background-color: {chip_color}; color: white;"
        )