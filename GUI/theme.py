"""
Every color in the app lives here. Change a value in THEME and every
widget that uses it updates - nothing else in the codebase needs touching.

How this works:
- Qt widgets (QListWidget, QPushButton, etc.) are styled with QSS, which is
  basically CSS for Qt. build_stylesheet() below just formats a theme
  dict's values into a QSS string, applied once via app.setStyleSheet(...).
- The video overlay text/border (drawn with OpenCV, not Qt) reads the same
  dict, but as BGR tuples instead of hex strings, since that's what cv2
  wants. hex_to_bgr() does that conversion for you.
- v2 adds THEMES (named presets) and hue_to_hex() (for the draggable hue
  slider) so the theme can change live, not just by editing this file.
"""

import colorsys


def hex_to_bgr(hex_color: str) -> tuple:
    """'#2ecc71' -> (113, 204, 46)  (OpenCV wants BGR, not RGB)"""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def hue_to_hex(hue_degrees: float, saturation: float = 0.65, value: float = 0.85) -> str:
    """Turn a 0-360 hue (what a slider drags along) into a hex color."""
    r, g, b = colorsys.hsv_to_rgb((hue_degrees % 360) / 360.0, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# ---- Edit these to reskin the whole app (or pick a preset from THEMES) ----
THEME = {
    "background": "#121212",   # main window background
    "panel_bg": "#1e1e1e",     # sidebar / list / button background
    "border": "#333333",       # thin borders around panels
    "text": "#e0e0e0",         # default text color
    "accent": "#2b7de9",       # selection / progress bar / hover color
    "success": "#2ecc71",      # "done" / correct step color
    "warning": "#e67e22",      # "out_of_order" color
    "error": "#e74c3c",        # "skipped" color
    "pending": "#7a7a7a",      # "pending" / neutral color
}
# -----------------------------------------------------------------------------

# v2: named presets for the "click a swatch" quick-switch, in addition to
# the drag-a-slider live hue control. Feel free to add your own here.
THEMES = {
    "Midnight": dict(THEME),
    "Daylight": {
        "background": "#f2f2f2",
        "panel_bg": "#ffffff",
        "border": "#dcdcdc",
        "text": "#1a1a1a",
        "accent": "#2b7de9",
        "success": "#27ae60",
        "warning": "#e67e22",
        "error": "#e74c3c",
        "pending": "#999999",
    },
    "Ocean": {
        "background": "#0b1f26",
        "panel_bg": "#12303b",
        "border": "#1f4753",
        "text": "#dff4f7",
        "accent": "#20c9c9",
        "success": "#2ecc71",
        "warning": "#f5a623",
        "error": "#ff5c5c",
        "pending": "#5f8993",
    },
    "Sunset": {
        "background": "#1a1015",
        "panel_bg": "#2a1a20",
        "border": "#4a2a33",
        "text": "#f5e6e0",
        "accent": "#ff7847",
        "success": "#7ed08c",
        "warning": "#ffb347",
        "error": "#ff5c7a",
        "pending": "#9c7b7f",
    },
}


def get_status_colors(theme: dict) -> dict:
    return {
        "pending": theme["pending"],
        "in_progress": theme["accent"],
        "done": theme["success"],
        "skipped": theme["error"],
        "out_of_order": theme["warning"],
    }


STATUS_COLORS = get_status_colors(THEME)  # kept for v1's import


def build_stylesheet(theme: dict = THEME) -> str:
    return f"""
    QMainWindow, QWidget {{
        background-color: {theme['background']};
        color: {theme['text']};
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }}
    QMainWindow::separator {{
        background-color: {theme['background']};
        width: 4px;
        height: 4px;
    }}
    QMainWindow::separator:hover {{
        background-color: {theme['accent']};
    }}
    
    /* Menu Bar */
    QMenuBar {{
        background-color: {theme['panel_bg']};
        color: {theme['text']};
        padding: 4px;
        border-bottom: 1px solid {theme['border']};
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 6px 12px;
        border-radius: 4px;
    }}
    QMenuBar::item:selected {{
        background: {theme['border']};
    }}
    QMenu {{
        background-color: {theme['panel_bg']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: 6px;
        padding: 4px 0px;
    }}
    QMenu::item {{
        padding: 6px 24px 6px 16px;
        border-radius: 4px;
        margin: 0px 4px;
    }}
    QMenu::item:selected {{
        background-color: {theme['accent']};
        color: white;
    }}

    QLabel#sectionTitle {{
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        color: {theme['accent']};
        padding: 10px 2px 4px 2px;
    }}
    QLabel#bubbleLabel {{
        background-color: rgba(20, 20, 20, 220);
        color: #ffffff;
        border: 1px solid {theme['accent']};
        border-radius: 12px;
        padding: 12px;
        font-size: 13px;
    }}
    
    /* List Widget */
    QListWidget {{
        background-color: {theme['panel_bg']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        outline: none;
        padding: 6px;
    }}
    QListWidget::item {{
        padding: 12px 10px;
        border-radius: 6px;
        margin: 2px 0px;
        color: {theme['text']};
    }}
    QListWidget::item:selected {{
        background-color: {theme['accent']};
        color: white;
        font-weight: bold;
    }}
    QListWidget::item:hover:!selected {{
        background-color: {theme['border']};
    }}
    
    /* Buttons */
    QPushButton {{
        background-color: {theme['panel_bg']};
        border: 1px solid {theme['border']};
        border-radius: 6px;
        padding: 8px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {theme['accent']};
        border-color: {theme['accent']};
        color: white;
    }}
    QPushButton:disabled {{
        color: #666666;
        border-color: transparent;
    }}
    
    /* Progress Bar */
    QProgressBar {{
        background-color: {theme['background']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        height: 18px;
    }}
    QProgressBar::chunk {{
        background-color: {theme['accent']};
        border-radius: 7px;
    }}
    
    /* Sliders */
    QSlider::groove:horizontal {{
        height: 6px;
        background: {theme['border']};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {theme['accent']};
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }}
    
    /* Dock Widgets */
    QDockWidget {{
        titlebar-close-icon: none;
        color: {theme['text']};
        font-weight: 700;
        font-size: 13px;
    }}
    QDockWidget::title {{
        background-color: transparent;
        padding: 12px 10px 8px 10px;
        border-radius: 6px;
    }}
    
    /* Toolbars */
    QToolBar {{
        background-color: {theme['panel_bg']};
        border-bottom: 1px solid {theme['border']};
        padding: 6px;
        spacing: 8px;
    }}
    QToolButton {{
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 14px;
        font-weight: 600;
    }}
    QToolButton:hover {{
        background-color: {theme['accent']};
        color: white;
    }}
    """