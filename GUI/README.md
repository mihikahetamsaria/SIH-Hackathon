# HAR GUI (Parth's track)

Desktop GUI that polls `data/step_record_log.json` on a 1-second timer and
renders it. See Section 6 of the project doc.

**Layout:**
- **Left, main area:** live camera feed with the recognized action and
  correctness burned directly into the frame (colored border + text band,
  drawn with OpenCV) - the GUI doesn't do any recognition itself, it just
  visualizes whatever the Step-Record Log currently says.
- **Right sidebar:** every step listed with its instruction clip. Click any
  step to preview its clip inline. The matching clip **auto-plays** the
  moment that step is marked `skipped`/`out_of_order` in the log.
- Slim status strip on top (current/next step, progress) and an alert
  banner at the bottom that flashes on mistakes.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Run it

Terminal 1 - the GUI:
```bash
python main.py
```

Terminal 2 - the placeholder simulator (stands in for Mihika's Event
Sequencing module until it's ready; writes to the same log file/schema):
```bash
python placeholder_simulator.py
```

## Files

- `main.py` - entry point
- `main_window.py` - the GUI itself (PySide6 + OpenCV)
- `theme.py` - **all UI colors live here**, see "Changing the UI colors" below
- `log_reader.py` - safe JSON loading helpers
- `placeholder_simulator.py` - fakes the Event Sequencing module's output
- `mistake_review_dialog.py` - an alternate popup-style mistake review (not
  currently wired into main_window.py's v2 sidebar layout - kept in case
  you want a modal version later instead of/alongside the sidebar)
- `data/experiment_definition.json` - Section 5a contract (sample: red/yellow box task)
- `data/step_record_log.json` - Section 5b contract, live-updated by the simulator (or later, by Mihika's module)
- `data/reference_clips/` - instruction clips per step (`step_<id>.mp4`), shown in the sidebar

## Changing the UI colors

Open `theme.py`. Near the top there's:

```python
THEME = {
    "background": "#121212",
    "panel_bg": "#1e1e1e",
    "border": "#333333",
    "text": "#e0e0e0",
    "accent": "#2b7de9",
    "success": "#2ecc71",
    "warning": "#e67e22",
    "error": "#e74c3c",
    "pending": "#7a7a7a",
}
```

Just edit the hex codes. For example, to switch from the dark theme to a
light one:

```python
THEME = {
    "background": "#f5f5f5",
    "panel_bg": "#ffffff",
    "border": "#dddddd",
    "text": "#1a1a1a",
    "accent": "#2b7de9",
    "success": "#2ecc71",
    "warning": "#e67e22",
    "error": "#e74c3c",
    "pending": "#999999",
}
```

Save the file and re-run `python main.py` - no other file needs to change.

**Why one edit updates everything:** `theme.py` has a function
`build_stylesheet(THEME)` that turns the dictionary into a QSS string
(Qt's version of CSS). `main_window.py` applies it once, in `main()`:

```python
app.setStyleSheet(build_stylesheet(THEME))
```

Every Qt widget (the sidebar list, buttons, progress bar) picks its colors
from that one stylesheet. The video overlay (the border + text drawn on
the camera feed) isn't a Qt widget - it's pixels drawn with OpenCV - so it
reads `THEME` too, but through `hex_to_bgr()`, which converts a hex color
like `"#2ecc71"` into the `(B, G, R)` tuple OpenCV expects. That's why
`theme.py` is the *only* file you need to touch to reskin the app; both
the Qt side and the OpenCV overlay pull from the same dictionary instead
of having colors hardcoded in two places.

If you want a color that only affects one specific widget (not the whole
theme), you can still set `.setStyleSheet(...)` on that one widget
directly in `main_window.py`, same as before - `theme.py` just holds the
defaults everything else falls back to.

## Swapping in the real pipeline

Once Mihika's module is ready, it just needs to write to
`data/step_record_log.json` using the same schema as
`placeholder_simulator.py` - nothing in `main_window.py` changes.

No camera? The video panel just shows "No camera feed"; the sidebar and
alerts still work off the log.

## Troubleshooting

- **"No such file or directory: .../data/step_record_log.json"** - the
  `data` folder didn't come through when the project was copied; recreate
  it with `experiment_definition.json` and `step_record_log.json` inside.
- **Sidebar clips won't play** - QtMultimedia needs a codec backend. Try
  re-encoding clips as H.264 MP4, or install a codec pack (e.g. K-Lite on
  Windows) system-wide.
