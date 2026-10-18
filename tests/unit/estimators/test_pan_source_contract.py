"""Paper Eq.7/Algorithm1 evidence, not execution of the related author code."""

# Match the public feature naming convention.
# ruff: noqa: N803, N806

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

torch = pytest.importorskip("torch")

from pu_toolbox import PANClassifier  # noqa: E402
from pu_toolbox.estimators.deep.pan import (  # noqa: E402
    pan_classifier_loss,
    pan_discriminator_loss,
)
from pu_toolbox.experiment.method_ledger import load_ledger  # noqa: E402
from pu_toolbox.experiment.survey_protocol import load_protocol  # noqa: E402

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
REVIEW_REF = "docs/research/pu_survey/data/pan_source_review_20261018.json"


@pytest.mark.parametrize("weight", [0.0001, 0.3, 1.0])
def test_basic_param_separate_pool_means_and_analytic_gradients_match_equation7(weight):
    dp = torch.tensor([-0.8, 0.2, 1.4], dtype=torch.float64, requires_grad=True)
    du = torch.tensor([-1.0, -0.2, 0.0, 0.8, 1.1], dtype=torch.float64, requires_grad=True)
    cu = torch.tensor([0.5, -0.6, 1.2, 0.1, -0.9], dtype=torch.float64, requires_grad=True)
    p, u, c = dp.detach().numpy(), du.detach().numpy(), cu.detach().numpy()
    d_value = (
        np.logaddexp(0, -p).mean()
        + np.logaddexp(0, u).mean()
        + weight * (c * (2 * expit(u) - 1)).mean()
    )
    loss = pan_discriminator_loss(dp, du, cu, weight=weight)
    assert float(loss.detach()) == pytest.approx(d_value, abs=1e-14)
    loss.backward()
    np.testing.assert_allclose(dp.grad, (expit(p) - 1) / len(p), atol=1e-14)
    expected_du = (expit(u) + 2 * weight * c * expit(u) * (1 - expit(u))) / len(u)
    np.testing.assert_allclose(du.grad, expected_du, atol=1e-14)
    assert cu.grad is None
    dp.grad = du.grad = None
    c_loss = pan_classifier_loss(cu, du, weight=weight)
    assert float(c_loss.detach()) == pytest.approx(
        -weight * (c * (2 * expit(u) - 1)).mean(), abs=1e-14
    )
    c_loss.backward()
    np.testing.assert_allclose(cu.grad, -weight * (2 * expit(u) - 1) / len(u), atol=1e-14)
    assert dp.grad is None and du.grad is None
    # Replicating either empirical pool does not reweight its expectation.
    repeated = pan_discriminator_loss(dp.repeat(2), du, cu, weight=weight)
    torch.testing.assert_close(repeated, loss, rtol=0, atol=1e-14)
    repeated = pan_discriminator_loss(dp, du.repeat(3), cu.repeat(3), weight=weight)
    torch.testing.assert_close(repeated, loss, rtol=0, atol=1e-14)


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("weight", [0.0001, 1.0])
def test_determ_actual_adam_steps_use_fresh_native_u_and_updated_detached_discriminator(
    monkeypatch, seed, weight
):
    rng = np.random.RandomState(9)
    X = rng.normal(size=(24, 3)).astype("float32")
    X[:, 0] = np.arange(24) / 10
    y = np.r_[np.ones(7, int), np.zeros(17, int)]
    torch.manual_seed(11)
    fitted = PANClassifier(
        model=torch.nn.Linear(3, 1),
        discriminator=torch.nn.Linear(3, 1),
        adversarial_weight=weight,
        max_epochs=1,
        batch_size=8,
        random_state=seed,
        device="cpu",
    )
    events, steps = [], []
    forward, step = torch.nn.Linear.forward, torch.optim.Adam.step
    sampling = np.random.RandomState(seed)
    sampling.randint(0, 2**31)  # Only the initial torch seed precedes role sampling.
    expected_batches = []
    for _ in range(3):
        p = X[sampling.choice(np.arange(7), 8)]
        du = X[sampling.choice(np.arange(7, 24), 8)]
        cu = X[sampling.choice(np.arange(7, 24), 8)]
        expected_batches.extend([du, p, du, cu, cu])

    def observed(layer, batch):
        out = forward(layer, batch)
        if layer is getattr(fitted, "model_", None):
            role = "C"
        elif layer is getattr(fitted, "discriminator_", None):
            role = "D"
        else:
            return out  # Constructor probes precede assignment and are not updates.
        events.append(
            (
                role,
                batch.detach().numpy().copy(),
                out.detach().numpy().reshape(-1),
                torch.is_grad_enabled(),
                layer.training,
            )
        )
        return out

    def checked(optimizer, *args, **kwargs):
        params = optimizer.param_groups[0]["params"]
        is_d = params[0] is fitted.discriminator_.weight
        offset = (len(steps) // 2) * 5
        if is_d:
            fixed_c, positive, unlabeled = events[offset : offset + 3]
            assert [event[0] for event in (fixed_c, positive, unlabeled)] == ["C", "D", "D"]
            assert [event[3:] for event in (fixed_c, positive, unlabeled)] == [
                (False, False),
                (True, True),
                (True, True),
            ]
            dp, du = expit(positive[2]), expit(unlabeled[2])
            p_grad = (dp - 1) / 8
            u_grad = (du + 2 * weight * fixed_c[2] * du * (1 - du)) / 8
            expected_w = positive[1].T @ p_grad + unlabeled[1].T @ u_grad
            expected_b = p_grad.sum() + u_grad.sum()
            inactive = fitted.model_
        else:
            assert steps[-1] == "D"
            fixed_d, classifier = events[offset + 3 : offset + 5]
            assert [fixed_d[0], classifier[0]] == ["D", "C"]
            assert [fixed_d[3:], classifier[3:]] == [(False, False), (True, True)]
            np.testing.assert_array_equal(fixed_d[1], classifier[1])
            # D's current weights are after the actual preceding Adam update.
            reward = classifier[1] @ fitted.discriminator_.weight.detach().numpy().T
            reward += fitted.discriminator_.bias.detach().numpy()
            np.testing.assert_allclose(fixed_d[2], reward.reshape(-1), atol=1e-7)
            c_grad = -weight * (2 * expit(reward.reshape(-1)) - 1) / 8
            expected_w, expected_b = classifier[1].T @ c_grad, c_grad.sum()
            inactive = fitted.discriminator_
        np.testing.assert_allclose(params[0].grad.numpy(), expected_w[None], rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(params[1].grad.numpy(), [expected_b], rtol=2e-5, atol=1e-7)
        before = [parameter.detach().clone() for parameter in inactive.parameters()]
        result = step(optimizer, *args, **kwargs)
        for parameter, old in zip(inactive.parameters(), before, strict=True):
            torch.testing.assert_close(parameter, old, rtol=0, atol=0)
        steps.append("D" if is_d else "C")
        return result

    monkeypatch.setattr(torch.nn.Linear, "forward", observed)
    monkeypatch.setattr(torch.optim.Adam, "step", checked)
    fitted.fit(X, y, class_prior=0.4)
    assert steps == ["D", "C"] * 3 and fitted.optimizer_steps_ == 6
    assert len(events) == len(expected_batches) == 15
    for event, batch in zip(events, expected_batches, strict=True):
        np.testing.assert_array_equal(event[1], batch)
    assert fitted._class_prior is None and fitted.calibration_applied_ is False


def test_edge_prediction_uses_classifier_not_adversary():
    X = np.random.RandomState(4).normal(size=(24, 3)).astype("float32")
    y = np.r_[np.ones(8), np.zeros(16)]
    fitted = PANClassifier(max_epochs=1, hidden_dim=4, batch_size=8, device="cpu").fit(X, y)
    expected = fitted.decision_function(X)
    with torch.no_grad():
        for parameter in fitted.discriminator_.parameters():
            parameter.add_(100)
    np.testing.assert_array_equal(fitted.decision_function(X), expected)


def test_basic_source_review_binds_current_code_tests_ledger_and_non_admission():
    review = json.loads((ROOT / REVIEW_REF).read_text(encoding="utf-8"))
    entry = load_ledger()["methods"]["pan"]
    assert entry["source_behavior_review_ref"] == REVIEW_REF
    assert review["method"] == "pan"
    assert review["review_status"] == "pending_method_owner_review"
    assert review["owner_signatures"] == []
    assert review["formal_admission"] is False
    assert "pan" not in load_protocol()["method_profiles"]
    assert review["source"]["repository"] == entry["code_version"]["upstream_url"]
    assert review["source"]["commit"] == entry["code_version"]["upstream_commit"]
    assert review["source"]["commit"] == "5be703639e4c1eb2c63d869ba150d84ebf0c116e"
    assert review["source"]["source_status"] == "official_related"
    assert {row["path"] for row in review["source"]["files"]} == {
        "main.py",
        "helper/pu_learning_dataset.py",
        "helper/cifar10_dataset.py",
        "helper/mnist_dataset.py",
        "README.md",
        "cfg/cifar10/uPU",
        "cfg/mnist/uPU",
        "cfg/news/uPU",
        "cfg/sddd/uPU",
    }
    assert review["source"]["executed_upstream_code"] is False
    license_observation = review["source"]["license_observation"]
    assert license_observation["status"] == "unknown"
    assert license_observation["permission_inferred_from_public_repository"] is False
    assert license_observation["upstream_code_or_weights_copied_executed_or_redistributed"] is False
    scope = review["engineering_scope"]
    assert scope["prediction_child_mode_restoration_fixed"] is True
    for name in (
        "fit_body_or_objective_changed",
        "historical_probe_or_frozen_result_changed",
        "new_real_data_or_GPU_run",
        "numeric_reproduction_or_formal_recipe_approval",
    ):
        assert scope[name] is False
    assert review["remaining_blockers"]
    assert {row["path"]: row["sha256"] for row in review["source"]["files"]} == (
        entry["code_version"]["source_files_sha256"]
    )
    source = ROOT / review["implementation_ref"]
    assert (
        hashlib.sha256(source.read_text(encoding="utf-8").encode()).hexdigest()
        == (review["implementation_sha256"])
    )
    assert review["calibration"] == {
        "run_view": entry["run_view"],
        "calibration_applied": entry["calibration_applied"],
        "population_prior_enters_target": False,
        "source_equivalence_claim": False,
    }
    for ref in review["behavior_test_refs"]:
        file, name = ref.split("::")
        nodes = ast.parse((ROOT / file).read_text(encoding="utf-8")).body
        assert any(isinstance(node, ast.FunctionDef) and node.name == name for node in nodes)
