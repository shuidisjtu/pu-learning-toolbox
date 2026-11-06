"""Current method routing defaults and interpretation for Survey experiments.

Capabilities and source metadata belong to the algorithm registry. The ledger
contains sampling/prior semantics, calibration defaults and method variants,
without historical review or evidence-file dependencies. Actual run choices
belong to that run's manifest (``run_view`` / ``calibration_applied``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

LEDGER_PATH = Path(__file__).with_name("method_ledger.json")

SAMPLING_ASSUMPTIONS = frozenset({"os", "ts", "both"})

#: Ledger notes open with either an ASCII or a full-width parenthesis; the head
#: is everything before it.
_NOTE_OPEN = re.compile(r"[（(]")


def load_ledger(path: str | Path = LEDGER_PATH) -> dict[str, Any]:
    """Load the survey method ledger; a payload without ``methods`` fails loud."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("methods"), dict):
        raise ValueError(f"{path} is not a valid method ledger file.")
    return value


def native_sampling_assumption(entry: dict[str, Any]) -> str:
    """Return one ledger entry's sampling assumption as a bare enum value.

    Only the head before the first parenthesis is the enum training views gate
    on; the trailing note stays documentation.
    """
    raw = entry.get("native_sampling_assumption") if isinstance(entry, dict) else None
    if not isinstance(raw, str):
        raise ValueError("method ledger entry is missing native_sampling_assumption")
    value = _NOTE_OPEN.split(raw, maxsplit=1)[0].strip().lower()
    if value not in SAMPLING_ASSUMPTIONS:
        raise ValueError(
            f"native_sampling_assumption must be one of {sorted(SAMPLING_ASSUMPTIONS)}; got {raw!r}"
        )
    return value
