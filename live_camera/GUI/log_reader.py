"""
Helpers for reading the shared JSON contracts (Section 5 of the project doc).

These are intentionally defensive: the Step-Record Log file may be mid-write
by another process (Mihika's module, or the placeholder simulator) when we
poll it, so a failed/partial read should never crash the GUI - we just skip
that tick and try again on the next timer fire.
"""

import json
from pathlib import Path
from typing import Optional


def load_json_safe(path: Path) -> Optional[dict]:
    """Load a JSON file, returning None (not raising) on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_experiment_definition(path: Path) -> Optional[dict]:
    return load_json_safe(path)


def load_step_record_log(path: Path) -> Optional[dict]:
    return load_json_safe(path)
