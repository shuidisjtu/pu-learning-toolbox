"""A v3 unit bound to the frozen registry records the binding and stays auditable."""

# ruff: noqa: F811

import copy
import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.survey_protocol import digest, load_protocol, resolve_protocol_path
from pu_toolbox.experiment.survey_recipe_audit import validate_consumed_bindings
from pu_toolbox.experiment.survey_recipe_registry import (
    bind_recipe_candidate,
    canonical_digest,
    load_registry,
)
from tests.unit.experiment._survey_script_helpers import (  # noqa: F401
    make_sar_splits,
    make_splits,
    run_manifests,
    survey_script,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "pu_toolbox/experiment/survey_recipe_registry_v1.json"

#: Every place a bound manifest repeats the identity it consumed.
CARRIED = (
    "recipe_registry_version",
    "recipe_registry_sha256",
    "recipe_profile_id",
    "recipe_candidate_id",
    "resolved_params_sha256",
    "inner_search_id",
)


def v3():
    return load_protocol(resolve_protocol_path("survey-v3.0"))


def expected_binding():
    protocol = v3()
    binding = bind_recipe_candidate(
        load_registry(REGISTRY),
        protocol,
        method="nnpu",
        dataset="spambase",
        training_path="native_2d",
    )
    binding.update(protocol_version=protocol["protocol_version"], protocol_sha256=digest(protocol))
    return binding


@pytest.fixture(scope="module")
def bound_run(tmp_path_factory, survey_script):
    root = tmp_path_factory.mktemp("bound-v3")
    data_dir = root / "splits"
    data_dir.mkdir()
    # 40 train positives so c=0.5 leaves enough labeled positives for nnpu's
    # two-positive minimum in both the train and validation roles.
    make_sar_splits(data_dir)
    (data_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "spambase",
                "seed": 0,
                "role_sizes": {"train": 60, "pu_val": 20, "clean_val": 20, "test": 20},
                "class_prior": {"population": 70 / 120},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out_dir = root / "results"
    code = survey_script.main(
        [
            str(data_dir),
            "--protocol",
            "survey-v3.0",
            "--dataset",
            "spambase",
            "--method",
            "nnpu",
            "--c",
            "0.5",
            "--seeds",
            "0",
            "--class-prior",
            "0.3",
            "--recipe-registry",
            str(REGISTRY),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    manifests = run_manifests(out_dir)
    assert len(manifests) == 1
    return json.loads(manifests[0].read_text(encoding="utf-8")), manifests[0]


def test_basic_bound_run_records_the_registry_identity(bound_run):
    manifest, manifest_path = bound_run
    binding = expected_binding()

    assert manifest["recipe_binding_status"] == "registry_bound"
    for field in CARRIED:
        assert manifest[field] == binding[field], field
    assert manifest["profile_identity_schema_version"] == "survey-profile-identity-1"
    assert manifest["profile_id"] == "nnpu"
    assert manifest["algorithm_id"] == "nnpu"
    # The run directory names the digest it was bound to, so two registries
    # cannot overwrite each other's results.
    assert canonical_digest(load_registry(REGISTRY)) in manifest_path.as_posix()


def test_param_candidates_and_selection_carry_the_same_binding(bound_run):
    manifest, _ = bound_run
    binding = expected_binding()

    assert manifest["candidate_runs"] and manifest["selection"]
    for child in [*manifest["candidate_runs"], *manifest["selection"].values()]:
        for field in CARRIED:
            assert child[field] == binding[field], field


def test_basic_bound_run_passes_the_aggregation_gate(bound_run):
    manifest, _ = bound_run
    validate_consumed_bindings([manifest], {"nnpu": expected_binding()}, protocol=v3())


def test_edge_tampered_binding_is_rejected(bound_run):
    manifest, _ = bound_run
    tampered = copy.deepcopy(manifest)
    tampered["selection"]["OA"]["resolved_params_sha256"] = "0" * 64

    with pytest.raises(ValueError):
        validate_consumed_bindings([tampered], {"nnpu": expected_binding()}, protocol=v3())


def test_determ_repeated_audits_are_identical_and_do_not_mutate(bound_run):
    manifest, _ = bound_run
    before = copy.deepcopy(manifest)
    expected = expected_binding()
    expected_before = copy.deepcopy(expected)

    validate_consumed_bindings([manifest], {"nnpu": expected}, protocol=v3())
    validate_consumed_bindings([manifest], {"nnpu": expected}, protocol=v3())

    assert manifest == before
    assert expected == expected_before


def test_edge_bound_run_without_expectations_is_refused(bound_run):
    manifest, _ = bound_run
    with pytest.raises(ValueError, match="batch audit"):
        validate_consumed_bindings([manifest], None, protocol=v3())
