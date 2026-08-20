# ruff: noqa: N803, N806

"""Windowed monitoring for repeated source-to-current PU shift audits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pu_toolbox.utils.serialization import json_safe

from .shift import PUShiftReport, analyze_pu_shift

AlertLevel = Literal["none", "info", "warning", "critical"]

__all__ = ["PUShiftMonitor", "ShiftWindow", "AlertLevel"]


@dataclass(frozen=True)
class ShiftWindow:
    """Serializable summary for one monitored target window."""

    window_id: str
    timestamp: str
    domain_auc: float
    severity: str
    adaptation_ready: bool
    effective_sample_fraction: float
    source_label_rate: float
    target_label_rate: float | None
    auc_change: float | None
    label_rate_change: float | None
    alert_level: AlertLevel
    alert_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "window_id": self.window_id,
                "timestamp": self.timestamp,
                "domain_auc": self.domain_auc,
                "severity": self.severity,
                "adaptation_ready": self.adaptation_ready,
                "effective_sample_fraction": self.effective_sample_fraction,
                "source_label_rate": self.source_label_rate,
                "target_label_rate": self.target_label_rate,
                "auc_change": self.auc_change,
                "label_rate_change": self.label_rate_change,
                "alert_level": self.alert_level,
                "alert_codes": list(self.alert_codes),
            }
        )


class PUShiftMonitor:
    """Compare successive target windows against a fixed reference domain.

    The monitor stores summaries, not raw features or labels. This makes the
    history safe to persist and keeps each window independently reproducible.
    """

    def __init__(
        self,
        X_reference: Any,
        y_reference_pu: Any,
        *,
        alpha: float = 0.1,
        cv: int = 5,
        random_state: int | None = 42,
        auc_jump_threshold: float = 0.1,
        label_rate_jump_threshold: float = 0.05,
    ) -> None:
        for name, value in {
            "auc_jump_threshold": auc_jump_threshold,
            "label_rate_jump_threshold": label_rate_jump_threshold,
        }.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number.")
        self.X_reference = X_reference
        self.y_reference_pu = y_reference_pu
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.auc_jump_threshold = float(auc_jump_threshold)
        self.label_rate_jump_threshold = float(label_rate_jump_threshold)
        self._history: list[ShiftWindow] = []

    @property
    def history(self) -> tuple[ShiftWindow, ...]:
        return tuple(self._history)

    def update(
        self,
        X_window: Any,
        *,
        window_id: str,
        y_window_pu: Any | None = None,
        timestamp: str | None = None,
    ) -> tuple[ShiftWindow, PUShiftReport]:
        """Audit and append one uniquely identified target window."""
        if not isinstance(window_id, str) or not window_id.strip():
            raise ValueError("window_id must be a non-empty string.")
        if any(item.window_id == window_id for item in self._history):
            raise ValueError(f"window_id {window_id!r} already exists in monitor history.")
        recorded_at = timestamp or datetime.now(timezone.utc).isoformat()
        try:
            datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be an ISO-8601 string.") from exc
        report = analyze_pu_shift(
            self.X_reference,
            self.y_reference_pu,
            X_window,
            y_target_pu=y_window_pu,
            alpha=self.alpha,
            cv=self.cv,
            random_state=self.random_state,
        )
        previous = self._history[-1] if self._history else None
        target_rate = report.sample_summary.get("target_labeled_positive_rate")
        auc_change = report.domain_auc - previous.domain_auc if previous else None
        label_rate_change = (
            target_rate - previous.target_label_rate
            if previous and target_rate is not None and previous.target_label_rate is not None
            else None
        )
        codes: list[str] = []
        if report.severity == "high":
            codes.append("high_domain_shift")
        if not report.adaptation_ready:
            codes.append("adaptation_not_ready")
        if auc_change is not None and abs(auc_change) >= self.auc_jump_threshold:
            codes.append("domain_auc_jump")
        if (
            label_rate_change is not None
            and abs(label_rate_change) >= self.label_rate_jump_threshold
        ):
            codes.append("label_rate_jump")
        level: AlertLevel = "none"
        if "high_domain_shift" in codes or "adaptation_not_ready" in codes:
            level = "critical"
        elif codes:
            level = "warning"
        elif report.severity == "moderate":
            level = "info"
        window = ShiftWindow(
            window_id=window_id,
            timestamp=recorded_at,
            domain_auc=report.domain_auc,
            severity=report.severity,
            adaptation_ready=report.adaptation_ready,
            effective_sample_fraction=report.weight_summary["effective_sample_fraction"],
            source_label_rate=report.sample_summary["source_labeled_positive_rate"],
            target_label_rate=target_rate,
            auc_change=auc_change,
            label_rate_change=label_rate_change,
            alert_level=level,
            alert_codes=tuple(codes),
        )
        self._history.append(window)
        return window, report

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "analysis_type": "pu_shift_monitor_history",
            "configuration": {
                "alpha": self.alpha,
                "cv": self.cv,
                "random_state": self.random_state,
                "auc_jump_threshold": self.auc_jump_threshold,
                "label_rate_jump_threshold": self.label_rate_jump_threshold,
            },
            "n_windows": len(self._history),
            "latest_alert_level": self._history[-1].alert_level if self._history else "none",
            "windows": [item.to_dict() for item in self._history],
        }

    def save_history(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination

    def load_history(self, path: str | Path) -> PUShiftMonitor:
        """Restore summaries into this monitor after validating the schema/config."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.load_history_payload(payload)

    def load_history_payload(self, payload: dict[str, Any]) -> PUShiftMonitor:
        """Restore an already-decoded history payload (useful for UI uploads)."""
        if not isinstance(payload, dict):
            raise ValueError("history payload must be a JSON object.")
        if payload.get("analysis_type") != "pu_shift_monitor_history":
            raise ValueError("history file is not a PU shift monitor history.")
        config = payload.get("configuration", {})
        expected = {
            "alpha": self.alpha,
            "cv": self.cv,
            "random_state": self.random_state,
            "auc_jump_threshold": self.auc_jump_threshold,
            "label_rate_jump_threshold": self.label_rate_jump_threshold,
        }
        if config != expected:
            raise ValueError("history configuration does not match this monitor.")
        restored: list[ShiftWindow] = []
        for item in payload.get("windows", []):
            values = dict(item)
            values["alert_codes"] = tuple(values["alert_codes"])
            restored.append(ShiftWindow(**values))
        if len({item.window_id for item in restored}) != len(restored):
            raise ValueError("history contains duplicate window_id values.")
        self._history = restored
        return self
