"""Explicit draft-only profile identity, separate from registered algorithms.

No registry aliases are registered. Legacy protocols keep method == algorithm;
only the named draft extension can attach an algorithm to a recipe/profile ID.
"""

from __future__ import annotations

import re

SCHEMA = "survey-profile-identity-1"
FIELD = "profile_identity_schema_version"
DRAFT_VERSION = "survey-v3.0-draft"
MANIFEST_FIELDS = (FIELD, "profile_id", "algorithm_id")
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def has_profile_identities(protocol):
    """Presence is explicit; an unknown/false marker is not a legacy protocol."""
    return FIELD in protocol


def validate_profile_identities(protocol):
    """Validate names and mappings before any data loading or model fitting."""
    if not has_profile_identities(protocol):
        profiles = protocol.get("method_profiles", {})
        if any(isinstance(value, dict) and "algorithm" in value for value in profiles.values()):
            raise ValueError(
                "profile algorithm mappings require the explicit draft identity schema"
            )
        return
    profiles = protocol["method_profiles"]
    if protocol[FIELD] != SCHEMA or protocol.get("protocol_version") != DRAFT_VERSION:
        raise ValueError("profile identity schema requires explicit survey-v3.0-draft")
    from ..core.base import BasePUClassifier
    from ..core.exceptions import RegistryError
    from ..registry import get_algorithm, get_algorithm_registry, register_all_builtin_methods

    register_all_builtin_methods()
    registered = get_algorithm_registry()
    reserved = {name: name for name in registered}
    reserved.update(
        {alias.lower(): name for name, meta in registered.items() for alias in meta.aliases}
    )
    reserved["pn_oracle"] = "pn_oracle"
    for profile_id, profile in profiles.items():
        if not isinstance(profile_id, str) or not _NAME.fullmatch(profile_id):
            raise ValueError("profile IDs must be safe lowercase identifiers")
        algorithm = profile.get("algorithm") if isinstance(profile, dict) else None
        if not isinstance(algorithm, str) or not _NAME.fullmatch(algorithm):
            raise ValueError("every draft profile requires an explicit canonical algorithm")
        if profile_id in reserved and reserved[profile_id] != algorithm:
            raise ValueError("profile cannot shadow a registered algorithm/oracle name")
        params = profile.get("params")
        if not isinstance(params, dict):
            raise ValueError("draft profile params must be an object")
        if profile.get("supervision", "pu_only") not in ("pu_only", "independent_clean_support"):
            raise ValueError("unknown profile supervision contract")
        if algorithm == "pn_oracle":
            if profile_id != "pn_oracle":
                raise ValueError("oracle profile aliases are not supported by this draft extension")
            continue
        if algorithm not in registered:
            raise ValueError(f"unknown canonical profile algorithm: {algorithm}")
        try:
            klass = get_algorithm(algorithm)
        except RegistryError as error:
            raise ValueError(
                f"profile algorithm has no available classifier: {algorithm}"
            ) from error
        if not isinstance(klass, type) or not issubclass(klass, BasePUClassifier):
            raise ValueError("survey profiles require a PU classifier, not a prior estimator")
        if (
            algorithm in ("pulns", "lagam")
            and profile.get("supervision") != "independent_clean_support"
        ):
            raise ValueError(
                "support-dependent algorithm requires independent clean support declaration"
            )
        if (
            algorithm == "self_pu"
            and "require_validation" in params
            and type(params["require_validation"]) is not bool
        ):
            raise ValueError("Self-PU require_validation must be an explicit JSON boolean")
        full_meta = algorithm == "self_pu" and params.get("require_validation") is True
        if full_meta and profile_id == "self_pu":
            raise ValueError("full-meta needs a separate profile ID; self_pu stays no-meta")
        if profile_id == "self_pu_full_meta" and (algorithm != "self_pu" or not full_meta):
            raise ValueError(
                "self_pu_full_meta cannot label the no-meta ablation or another method"
            )
        if full_meta and (
            profile.get("supervision") != "independent_clean_support"
            or params.get("distillation_scope") != "paper_population"
        ):
            raise ValueError(
                "full-meta profile needs independent clean support and paper_population"
            )


def profile_algorithm(protocol, profile_id):
    """Resolve a profile without changing the registry or guessing from its name."""
    validate_profile_identities(protocol)
    if not has_profile_identities(protocol):
        return profile_id
    return protocol["method_profiles"][profile_id]["algorithm"]


def profile_manifest_identity(protocol, profile_id):
    """Only the new extension adds fields; old manifest identities stay unchanged."""
    if not has_profile_identities(protocol):
        return {}
    return {
        FIELD: SCHEMA,
        "profile_id": profile_id,
        "algorithm_id": profile_algorithm(protocol, profile_id),
    }


def require_four_role_profile(protocol, profile_id):
    """Do not use an alias to bypass the still-missing formal support adapter."""
    if not has_profile_identities(protocol):
        return
    algorithm = profile_algorithm(protocol, profile_id)
    profile = protocol["method_profiles"][profile_id]
    if (
        algorithm in ("pulns", "lagam")
        or profile.get("supervision") == "independent_clean_support"
        or (algorithm == "self_pu" and profile["params"].get("require_validation") is True)
    ):
        raise ValueError(
            "clean-support profile requires the versioned five-role adapter; "
            "four-role execution unavailable"
        )
    if algorithm == "cvir":
        raise ValueError(
            "CVIR requires a versioned per-view prior/pool adapter; generic execution unavailable"
        )


def require_legacy_recipe_identity(protocol):
    """Registry v1 refs point at v2; this extension cannot pretend to bind them."""
    if has_profile_identities(protocol):
        raise ValueError("recipe registry v1 cannot bind the draft profile identity schema")


def valid_manifest_profile_identity(payload):
    """Check new artifact shape, not its algorithm mapping against a source matrix."""
    if not has_profile_identities(payload):
        return True
    if payload[FIELD] != SCHEMA or payload.get("protocol_version") != DRAFT_VERSION:
        return False
    unit = payload.get("execution_unit")
    if not isinstance(unit, dict) or type(payload.get("seed")) is not int:
        return False
    for field in ("profile_id", "algorithm_id"):
        value = payload.get(field)
        if not isinstance(value, str) or not _NAME.fullmatch(value):
            return False
    return payload["profile_id"] == unit.get("method")
