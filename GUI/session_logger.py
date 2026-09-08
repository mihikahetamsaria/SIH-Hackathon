"""
Persists a dated archive of each run's Step-Record Log (with an added
video_path per step) so the GUI can show history across sessions, not
just whatever's live right now. Kept fully separate from
data/step_record_log.json (Section 5b's shared team contract) - this is
local bookkeeping for Parth's GUI track only, nobody else needs to read
or write it.
"""

import json
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def new_session_path(started_at: datetime) -> Path:
    stamp = started_at.strftime("%Y-%m-%d_%H%M%S")
    return LOGS_DIR / f"session_{stamp}.json"


def save_session(path: Path, started_at: datetime, steps: list):
    payload = {"session_started": started_at.isoformat(), "steps": steps}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def list_sessions() -> list:
    return sorted(LOGS_DIR.glob("session_*.json"), reverse=True)


def load_session(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
