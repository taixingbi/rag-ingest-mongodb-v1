"""State management for incremental ingestion."""

import json
import os
from typing import Dict, Optional


STATE_FILE = "state.json"


def load_state() -> Dict[str, Dict[str, str]]:
    """Load ingestion state from state.json."""
    if not os.path.exists(STATE_FILE):
        return {}
    
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: Dict[str, Dict[str, str]]) -> None:
    """Save ingestion state to state.json."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_file_state(filepath: str, state: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Get state for a specific file."""
    return state.get(filepath)


def update_file_state(
    filepath: str,
    content_hash: str,
    mtime: str,
    state: Dict[str, Dict[str, str]],
) -> None:
    """Update state for a specific file."""
    state[filepath] = {
        "content_hash": content_hash,
        "mtime": mtime,
    }


def should_skip_file(
    filepath: str,
    current_hash: str,
    current_mtime: str,
    state: Dict[str, Dict[str, str]],
) -> bool:
    """Check if file should be skipped (unchanged since last ingest)."""
    file_state = get_file_state(filepath, state)
    if file_state is None:
        return False
    
    return (
        file_state.get("content_hash") == current_hash
        and file_state.get("mtime") == current_mtime
    )
