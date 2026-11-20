"""Evidence for an explicitly resolved training prior."""


def risk_pool_target(calibrated_view):
    """Name the role used by the objective, not the input sampling design."""
    return "positive_union_unlabeled" if calibrated_view else "native_unlabeled"


def training_prior_uses(*consumers, value, target, parameter, fit_override):
    """Record a fit or constructor fact once for each actual training use."""
    origin = "estimator" if fit_override is None else "fit"
    return tuple(
        {
            "consumer": consumer,
            "partition": "train",
            "target": target,
            "value": float(value),
            "source": f"{origin}.{parameter}",
        }
        for consumer in consumers
    )
