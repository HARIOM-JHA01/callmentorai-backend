"""
In-memory progress tracker for analysis pipeline sessions.

Stores ephemeral progress that only lives while a session is processing.
Automatically cleared when a session reaches a terminal state.
"""

from typing import TypedDict

_store: dict[str, "ProgressEntry"] = {}


class ProgressEntry(TypedDict):
    pct: int        # 0-100
    stage: str      # machine-readable key consumed by frontend i18n


def set_progress(session_id: str, pct: int, stage: str) -> None:
    _store[session_id] = {"pct": pct, "stage": stage}


def get_progress(session_id: str) -> ProgressEntry:
    return _store.get(session_id, {"pct": 0, "stage": "starting"})


def clear_progress(session_id: str) -> None:
    _store.pop(session_id, None)
