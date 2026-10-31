"""CVIR image CLI pins explicit OS/TS, observed-count prior and stage evidence."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_candidate_image_experiment as script  # noqa: E402

pytestmark = pytest.mark.unit


def product(tmp_path):
    directory = tmp_path / "input"
    directory.mkdir()
    for i, role in enumerate(script.ROLES):
        np.savez(
            directory / f"{role}.npz",
            X=np.random.RandomState(i).randint(0, 256, (12, 3, 8, 8)).astype(np.uint8),
            y=np.arange(12) % 2,
            indices=np.arange(12) + i * 20,
        )
    return directory


def parameters(**overrides):
    return (
        dict(
            hidden_dim=4,
            warm_start_epochs=1,
            max_epochs=2,
            batch_size=16,
            learning_rate=0.001,
            data_device_policy="cpu_stream",
            input_sampling="os",
            device="cpu",
        )
        | overrides
    )


def plan(directory, **overrides):
    return script.build_plan(
        directory,
        **(
            dict(
                method="cvir",
                dataset="cifar10",
                prior=0.4,
                prior_source="synthetic population",
                seed=3,
                parameters=parameters(),
                checkpoint_bytes=64 * 2**20,
                c_token="0.5",
                view="ts",
            )
            | overrides
        ),
    )


@pytest.mark.parametrize("view", ["os", "ts"])
@pytest.mark.parametrize("warm", [0, 1])
def test_basic_determ_dual_selection_and_bound_actual_prior(tmp_path, view, warm):
    directory = product(tmp_path)
    requested = parameters(warm_start_epochs=warm)
    planned = plan(directory, view=view, parameters=requested)
    assert planned == plan(directory, view=view, parameters=requested)
    expected_prior = 0.4 if view == "ts" else (0.4 * 12 - 3) / 9
    assert planned["cvir_prior_binding"]["risk_pool_positive_prior"] == expected_prior
    assert not planned["cvir_prior_binding"]["uses_hidden_training_labels"]
    summary = script.execute(planned, tmp_path / "output")
    assert summary["accepted_engineering_run"] and summary["selected_models_replayed"]
    assert summary["warning_count"] == summary["retry_count"] == 0
    assert set(summary["selection"]) == {"PA", "OA"}
    assert 1 <= summary["retained_checkpoint_count"] <= 2
    facts = summary["training_provenance"]
    assert len(facts["epoch_cost_history"]) == warm + 2
    assert not facts["population_prior_forwarded_to_fit"]
    assert facts["population_prior_binding"]["risk_pool_positive_prior"] == expected_prior
    assert facts["bound_fit_receipt"]["calibration_applied"] is (view == "ts")
    assert summary["source_default_snapshot_projection"]["epochs"] == 21
    assert not summary["formal_admission"] and not summary["paper_numerical_reproduction"]


@pytest.mark.parametrize(
    "overrides",
    [
        dict(view=None),
        dict(parameters=parameters(unlabeled_positive_prior=0.4)),
        dict(parameters=parameters(prior_source="external U")),
        dict(parameters=parameters(batch_size=4)),
        dict(parameters=parameters(input_sampling="ts")),
    ],
)
def test_param_missing_view_prebound_prior_and_singleton_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        plan(product(tmp_path), **overrides)


def test_edge_changed_prior_binding_does_not_create_output(tmp_path):
    planned = plan(product(tmp_path))
    changed = copy.deepcopy(planned)
    changed["cvir_prior_binding"]["risk_pool_positive_prior"] = 0.9
    with pytest.raises(ValueError, match="plan changed"):
        script.execute(changed, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_param_non_cvir_method_cannot_override_native_bound_view(tmp_path):
    with pytest.raises(ValueError, match="training view differs"):
        plan(product(tmp_path), method="p3mix", view="ts")
