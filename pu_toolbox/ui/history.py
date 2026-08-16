"""Process-level run history for the Streamlit UI.

Kept outside ``st.session_state``: a browser refresh rebuilds the session,
which would otherwise discard recent runs. A process-level queue survives
refreshes and is cleared only when the Streamlit process restarts, matching
the documented contract in docs/user/howto/ui.md.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any

_MAX_ENTRIES = 20

_LOCK = Lock()
_HISTORY: deque[dict[str, Any]] = deque(maxlen=_MAX_ENTRIES)


def append(entry: dict[str, Any]) -> None:
    """Prepend one history entry, keeping at most 20."""
    with _LOCK:
        _HISTORY.appendleft(entry)


def snapshot() -> list[dict[str, Any]]:
    """Return a copy of the current history, newest first."""
    with _LOCK:
        return list(_HISTORY)


def clear_for_tests() -> None:
    """Reset history (test helper only)."""
    with _LOCK:
        _HISTORY.clear()
