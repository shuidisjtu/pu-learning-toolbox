"""KLDCE actual CPU plans preserve realized SCAR binding and explicit prior choice."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_classical_candidate_experiment as script  # noqa: E402

pytestmark = pytest.mark.unit


def product(tmp_path):
    directory = tmp_path / "input"
    directory.mkdir()
    for i, role in enumerate(script.ROLES):
        labels = np.r_[np.ones(11, int), np.zeros(25, int)]
        np.savez(
            directory / f"{role}.npz",
            X=np.random.RandomState(7 + i).normal(size=(36, 2)) + labels[:, None],
            y=labels,
            indices=np.arange(36) + i * 36,
        )
    return directory


def parameters(**overrides):
    return (
        dict(
            sigma="scale",
            reg_strength=1.0,
            centroid_radius=1.0,
            mom_groups=2,
            covariance_ridge=0.0,
            max_acs_iter=10,
            max_inner_iter=2000,
            inner_tol=1e-6,
            tol=1e-6,
        )
        | overrides
    )


def plan(directory, **overrides):
    return script.build_plan(
        directory,
        **(
            dict(
                method="kldce",
                dataset="synthetic",
                prior=0.3,
                prior_source="external population parameter",
                seed=3,
                view="os",
                parameters=parameters(),
                c_token="0.3",
                prior_policy="derived",
                kernel_payload_cap_bytes=64 * 2**20,
            )
            | overrides
        ),
    )


@pytest.mark.parametrize("policy", ["derived", "population_override"])
def test_basic_determ_actual_realized_binding_and_dual_selection_replay(tmp_path, policy):
    directory = product(tmp_path)
    planned = plan(directory, prior_policy=policy)
    assert planned == plan(directory, prior_policy=policy)
    binding = planned["flip_probability_binding"]
    assert binding["c_requested"] == 0.3 and binding["c_realized"] == 3 / 11
    assert binding["flip_probability"] == 1 - 3 / 11 != 1 - 0.3
    assert planned["work_units_per_attempt"] == 1 and planned["neural_epoch_count"] == 0
    expected_prior = 0.3 if policy == "population_override" else 3 / (36 * (1 - (1 - 3 / 11)))
    assert planned["kldce_preflight"]["effective_training_prior"] == expected_prior
    output = tmp_path / "output"
    summary = script.execute(planned, output)
    assert summary["accepted_engineering_run"] and summary["selected_models_replayed"]
    assert summary["warning_count"] == summary["retry_count"] == 0
    assert set(summary["selection"]) == {"PA", "OA"}
    assert summary["training_provenance"]["effective_training_prior"] == expected_prior
    assert summary["training_provenance"]["preflight"]["generation_binding"] == binding
    assert summary["training_provenance"]["acs_rounds_completed"] > 0
    assert summary["training_provenance"]["qp_solves_observed"] is None
    assert all(row["epoch"] is None for row in summary["selection"].values())
    assert not list(output.rglob("*.pt")) and not summary["formal_admission"]
    assert len(summary["attempt_receipts_sha256"]) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        dict(prior_policy=None),
        dict(kernel_payload_cap_bytes=None),
        dict(kernel_payload_cap_bytes=1),
        dict(view="ts"),
        dict(parameters=parameters(mom_groups=100)),
        dict(parameters=parameters(flip_probability=0.7)),
        dict(parameters=parameters(inner_tol=True)),
    ],
)
def test_param_missing_binding_or_invalid_budget_refused_without_fit(
    tmp_path, overrides, monkeypatch
):
    monkeypatch.setattr(
        script.KLDCEClassifier, "fit", lambda *a, **k: pytest.fail("plan reached KLDCE fit")
    )
    with pytest.raises(ValueError):
        plan(product(tmp_path), **overrides)


def test_edge_resealed_flip_or_policy_mismatch_refused_before_output(tmp_path):
    planned = plan(product(tmp_path))
    for field in ("flip_probability_binding", "kldce_prior_policy"):
        changed = copy.deepcopy(planned)
        if field == "flip_probability_binding":
            changed[field]["flip_probability"] = 0.7
        else:
            changed[field] = "population_override"
        changed["plan_sha256"] = script.strict_canonical_hash(
            {k: v for k, v in changed.items() if k != "plan_sha256"}
        )
        with pytest.raises(ValueError, match="identity changed"):
            script.execute(changed, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_edge_failed_fit_retries_keep_receipts_without_inventing_acs_progress(
    tmp_path, monkeypatch
):
    planned = plan(product(tmp_path))
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise TypeError("internal kernel failure")

    monkeypatch.setattr(script.KLDCEClassifier, "fit", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="all candidate runs failed"):
        script.execute(planned, output)
    failure = json.loads((output / "failure.json").read_text())
    assert len(calls) == 2 and len(failure["attempt_receipts"]) == 2
    for path in failure["attempt_receipts"]:
        receipt = json.loads((output / path).read_text())
        assert receipt["work_started"] == 1 and receipt["work_completed"] == 0
        assert receipt["acs_rounds_completed"] is None and receipt["qp_solves_observed"] is None


def test_basic_cli_only_plan_binds_realized_flip_without_constructing_a_fit(
    tmp_path, capsys, monkeypatch
):
    directory = product(tmp_path)
    requested = tmp_path / "parameters.json"
    requested.write_text(json.dumps(parameters()))
    monkeypatch.setattr(script.KLDCEClassifier, "fit", lambda *a, **k: pytest.fail("plan fitted"))
    result = script.main(
        [
            "--method",
            "kldce",
            "--data-dir",
            str(directory),
            "--dataset",
            "synthetic",
            "--class-prior",
            "0.3",
            "--prior-source",
            "external",
            "--seed",
            "3",
            "--view",
            "os",
            "--parameters",
            str(requested),
            "--c",
            "0.3",
            "--prior-policy",
            "derived",
            "--kernel-payload-cap-bytes",
            str(64 * 2**20),
        ]
    )
    planned = json.loads(capsys.readouterr().out)
    assert result == 0 and planned["flip_probability_binding"]["c_realized"] == 3 / 11
