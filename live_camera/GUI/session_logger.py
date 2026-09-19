import json
from datetime import datetime
from pathlib import Path

LOGS_DIR=Path(__file__).parent/"logs"
LOGS_DIR.mkdir(exist_ok=True)

def new_session_path(started_at):
    stamp=started_at.strftime("%Y-%m-%d_%H%M%S")
    return LOGS_DIR/f"session_{stamp}.json"

def save_session(path,started_at,steps,events=None):
    payload={
        "session_started":started_at.isoformat(),
        "steps":steps,
        "events":events or []
    }
    with open(path,"w",encoding="utf-8") as f:
        json.dump(payload,f,indent=2)

def list_sessions():
    return sorted(LOGS_DIR.glob("session_*.json"),reverse=True)

def load_session(path):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)