from pu_toolbox.model_selection.split import (
    PUStratifiedKFold,
    PUStratifiedShuffleSplit,
)


def __getattr__(name: str):
    """Lazily expose tuning classes without creating a pipeline import cycle."""
    if name in {"PUTuner", "TuningResult", "TuningTrial"}:
        from pu_toolbox.model_selection import tuning

        return getattr(tuning, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PUStratifiedKFold",
    "PUStratifiedShuffleSplit",
    "PUTuner",
    "TuningResult",
    "TuningTrial",
]
