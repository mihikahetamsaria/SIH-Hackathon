"""
Fakes the Event Sequencing module's output so the GUI can be built/demoed
before that module exists (Section 6 / Section 8 step 4).

Run this in a second terminal alongside main.py. It writes to the same
data/step_record_log.json the GUI polls, using the exact schema from
Section 5b. Includes one deliberate "out_of_order" step so you can see
the alert banner and hear the voice alert fire.

    python placeholder_simulator.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
STEP_LOG_PATH = DATA_DIR / "step_record_log.json"

STEP_NAMES = {
    1: "Open container",
    2: "Remove red box",
    3: "Remove yellow box",
    4: "Place red box",
    5: "Place yellow box",
    6: "Close container",
}

# Each entry: (step_id, status, current_expected_step_id)
# Step 4 is deliberately triggered before step 2/3 finish cleanly to show
# an out_of_order case worth speaking aloud.
SCRIPT = [
    (1, "in_progress", 1),
    (1, "done", 2),
    (2, "in_progress", 2),
    (2, "done", 3),
    (4, "out_of_order", 3),  # tried to place red box before yellow box removed
    (3, "in_progress", 3),
    (3, "done", 4),
    (4, "in_progress", 4),
    (4, "done", 5),
    (5, "in_progress", 5),
    (5, "done", 6),
    (6, "in_progress", 6),
    (6, "done", None),
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_initial_state():
    return {
        "current_expected_step_id": 1,
        "steps": [
            {
                "step_id": sid,
                "name": name,
                "status": "pending",
                "ts_start": None,
                "ts_end": None,
                "corrective_action": None,
            }
            for sid, name in STEP_NAMES.items()
        ],
    }


def apply_event(state, step_id, status, current_expected):
    for s in state["steps"]:
        if s["step_id"] == step_id:
            s["status"] = status
            if status == "in_progress":
                s["ts_start"] = now_iso()
            elif status in ("done", "skipped", "out_of_order"):
                s["ts_end"] = now_iso()
            if status == "out_of_order":
                s["corrective_action"] = (
                    f"Step '{s['name']}' was attempted out of order."
                )
            else:
                s["corrective_action"] = None
    state["current_expected_step_id"] = current_expected


def write_state(state):
    with open(STEP_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def main():
    state = build_initial_state()
    write_state(state)
    print(f"Wrote initial state to {STEP_LOG_PATH}")

    for step_id, status, current_expected in SCRIPT:
        time.sleep(3)
        apply_event(state, step_id, status, current_expected)
        write_state(state)
        print(f"step {step_id} -> {status} (next expected: {current_expected})")

    print("Simulation complete.")


if __name__ == "__main__":
    main()
