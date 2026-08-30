"""Portable JSON configuration shared by the CLI and graphical UI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .estimators.deep.vision import CNN_BACKBONES
from .workflows import DEFAULT_METRICS

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunConfiguration:
    """Serializable settings for one pipeline or tuning run."""

    classifier: str = "auto"
    classifier_params: dict[str, Any] = field(default_factory=dict)
    prior_estimator: str | None = "pen_l1"
    class_prior: float | None = None
    cv: int = 5
    metrics: tuple[str, ...] = tuple(DEFAULT_METRICS)
    random_state: int = 42
    architecture: str = "mlp"
    backbone: str = "cnn13"
    device: str = "auto"
    tuning_grid: dict[str, list[Any]] = field(default_factory=dict)
    scoring: str = "pu_zero_one_risk"
    comparison_classifiers: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RunConfiguration:
        """Validate and construct a configuration from decoded JSON."""
        if not isinstance(value, Mapping):
            raise ValueError("run configuration must be a JSON object.")
        version = value.get("schema_version", _SCHEMA_VERSION)
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported run configuration schema_version {version!r}; "
                f"expected {_SCHEMA_VERSION}."
            )
        allowed = {field.name for field in cls.__dataclass_fields__.values()} | {"schema_version"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown run configuration fields: {unknown}.")

        classifier = value.get("classifier", "auto")
        classifier_params = value.get("classifier_params", {})
        prior_estimator = value.get("prior_estimator", "pen_l1")
        class_prior = value.get("class_prior")
        cv = value.get("cv", 5)
        metrics = value.get("metrics", list(DEFAULT_METRICS))
        random_state = value.get("random_state", 42)
        architecture = value.get("architecture", "mlp")
        backbone = value.get("backbone", "cnn13")
        device = value.get("device", "auto")
        tuning_grid = value.get("tuning_grid", {})
        scoring = value.get("scoring", "pu_zero_one_risk")
        comparison_classifiers = value.get("comparison_classifiers", [])

        if not isinstance(classifier, str) or not classifier:
            raise ValueError("classifier must be a non-empty string.")
        _validate_mapping(classifier_params, "classifier_params")
        if prior_estimator is not None and (
            not isinstance(prior_estimator, str) or not prior_estimator
        ):
            raise ValueError("prior_estimator must be a non-empty string or null.")
        if class_prior is not None:
            if isinstance(class_prior, bool) or not isinstance(class_prior, int | float):
                raise ValueError("class_prior must be a number in (0, 1) or null.")
            class_prior = float(class_prior)
            if not 0.0 < class_prior < 1.0:
                raise ValueError("class_prior must be in (0, 1).")
        if isinstance(cv, bool) or not isinstance(cv, int) or cv < 2:
            raise ValueError("cv must be an integer >= 2.")
        if (
            not isinstance(metrics, list)
            or not metrics
            or any(not isinstance(item, str) or not item for item in metrics)
        ):
            raise ValueError("metrics must be a non-empty list of strings.")
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise ValueError("random_state must be an integer.")
        if architecture not in {"mlp", "cnn"}:
            raise ValueError("architecture must be 'mlp' or 'cnn'.")
        if backbone not in CNN_BACKBONES:
            raise ValueError("backbone must be 'cnn13', 'resnet18', or 'resnet50'.")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string.")
        _validate_mapping(tuning_grid, "tuning_grid")
        normalized_grid: dict[str, list[Any]] = {}
        for key, candidates in tuning_grid.items():
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"tuning_grid[{key!r}] must be a non-empty list.")
            normalized_grid[key] = list(candidates)
        if not isinstance(scoring, str) or not scoring:
            raise ValueError("scoring must be a non-empty string.")
        if not isinstance(comparison_classifiers, list) or any(
            not isinstance(name, str) or not name or name == "auto"
            for name in comparison_classifiers
        ):
            raise ValueError("comparison_classifiers must be a list of explicit names.")
        if comparison_classifiers and len(comparison_classifiers) < 2:
            raise ValueError("comparison_classifiers must contain at least two names.")
        if len(set(comparison_classifiers)) != len(comparison_classifiers):
            raise ValueError("comparison_classifiers must be unique.")
        if comparison_classifiers and tuning_grid:
            raise ValueError("model comparison and parameter tuning cannot run together.")
        if comparison_classifiers and classifier_params:
            raise ValueError("classifier_params are not supported for a model comparison.")
        config = cls(
            classifier=classifier,
            classifier_params=dict(classifier_params),
            prior_estimator=prior_estimator,
            class_prior=class_prior,
            cv=cv,
            metrics=tuple(metrics),
            random_state=random_state,
            architecture=architecture,
            backbone=backbone,
            device=device,
            tuning_grid=normalized_grid,
            scoring=scoring,
            comparison_classifiers=tuple(comparison_classifiers),
        )
        # Fail here with a readable message instead of later at download time.
        try:
            json.dumps(config.to_dict())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"run configuration contains non-JSON values: {exc}") from exc
        return config

    @classmethod
    def from_json(cls, text: str) -> RunConfiguration:
        """Decode a JSON document into a validated configuration."""
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"run configuration is not valid JSON: line {exc.lineno}, column {exc.colno}."
            ) from exc
        return cls.from_mapping(value)

    @classmethod
    def load(cls, path: str | Path) -> RunConfiguration:
        """Load a UTF-8 JSON configuration from disk."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "classifier": self.classifier,
            "classifier_params": dict(self.classifier_params),
            "prior_estimator": self.prior_estimator,
            "class_prior": self.class_prior,
            "cv": self.cv,
            "metrics": list(self.metrics),
            "random_state": self.random_state,
            "architecture": self.architecture,
            "backbone": self.backbone,
            "device": self.device,
            "tuning_grid": {key: list(values) for key, values in self.tuning_grid.items()},
            "scoring": self.scoring,
            "comparison_classifiers": list(self.comparison_classifiers),
        }

    def to_json(self) -> str:
        """Serialize with deterministic key ordering and UTF-8 text."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def _validate_mapping(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{name} keys must be non-empty strings.")
