"""
Theme picker, opened from the "Theme" menu item - same interaction as
"Log Files" and "Task Table": click it, a small window opens. Shared by
both v1 and v2 so theming behaves identically either way.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton

from theme import THEMES, hue_to_hex


class ThemeDialog(QDialog):
    """
    on_theme_changed(theme_dict) is called immediately whenever the user
    drags the slider or clicks a preset, so the main window can restyle
    itself live while this dialog is still open.
    """

    def __init__(self, current_theme: dict, on_theme_changed, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Theme")
        self.setFixedSize(340, 210)
        self.on_theme_changed = on_theme_changed
        self.current_theme = dict(current_theme)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Drag to change the accent color:"))
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setRange(0, 360)
        self.hue_slider.setValue(210)
        self.hue_slider.valueChanged.connect(self._on_hue_dragged)
        layout.addWidget(self.hue_slider)

        layout.addWidget(QLabel("Or pick a preset:"))
        preset_row = QHBoxLayout()
        for name in THEMES:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked=False, n=name: self._apply_preset(n))
            preset_row.addWidget(btn)
        layout.addLayout(preset_row)
        layout.addStretch()

    def _on_hue_dragged(self, hue_value: int):
        self.current_theme["accent"] = hue_to_hex(hue_value)
        self.on_theme_changed(dict(self.current_theme))

    def _apply_preset(self, name: str):
        self.current_theme = dict(THEMES[name])
        self.on_theme_changed(dict(self.current_theme))
