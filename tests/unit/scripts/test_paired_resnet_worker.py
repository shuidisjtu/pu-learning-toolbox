"""Exercise the paired worker with real fits and small synthetic image roles."""

import json
from argparse import Namespace

import numpy as np
import pytest

from pu_toolbox.experiment import DatasetBundle, DatasetPart
from scripts import run_survey_paired_resnet as worker

pytestmark = [
    pytest.mark.unit,
    pytest.mark.filterwarnings("ignore:validation_data was not supplied"),
]


@pytest.mark.parametrize("arm", ["frozen", "trainable"])
@pytest.mark.parametrize(
    ("method", "consumer"),
    [
        ("robust_pu", "robust_pu_warmup_risk"),
        ("split_pu", "split_pu_teacher_risk"),
        ("holistic_pu", None),
        ("self_pu", "self_pu_training_risk"),
        ("self_pu_full_meta", "self_pu_training_risk"),
        ("dist_pu", "dist_pu_warmup_histogram"),
    ],
)
def test_basic_param_paired_probe_records_training_and_selection_priors(
    tmp_path, monkeypatch, method, consumer, arm
):
    torch = pytest.importorskip("torch")
    nn = torch.nn
    population_prior = 0.5
    rng = np.random.default_rng(19)

    def part(start, count, *, for_selection=True):
        return DatasetPart(
            rng.normal(size=(count, 1, 4, 4)).astype(np.float32),
            np.tile([0, 1], count // 2),
            "clean",
            np.arange(start, start + count),
            for_selection,
        )

    bundle = DatasetBundle(
        part(0, 24), part(24, 12), part(36, 12), part(48, 12, for_selection=False)
    )
    split_ref = {"dataset": "cifar10", "class_prior": {"population": population_prior}}
    support = product = None
    if method == "self_pu_full_meta":
        support = part(60, 8, for_selection=False)
        product = {
            "receipt_sha256": "synthetic",
            "support_adapter_options": dict(
                dataset="cifar10",
                source="synthetic independent support",
                prior_estimation_indices=[],
                original_train_count=len(bundle.train.X),
                support_removed_from_train=False,
                group_policy="not_applicable",
                preparation_seed=0,
                sampling_rule="fixed synthetic cohort",
            ),
        }
    torch.manual_seed(17)
    encoder = nn.Sequential(nn.Conv2d(1, 512, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    monkeypatch.setattr(worker, "_load", lambda _args: (bundle, split_ref, support, product, {}))
    monkeypatch.setattr(worker, "_source_commit", lambda _full: ("a" * 40, False))
    monkeypatch.setattr(
        worker, "prepare_image_bundle", lambda *_args, **_kwargs: (bundle, encoder, {}, None)
    )
    monkeypatch.setattr(worker, "prepare_image_support", lambda support, **_kwargs: support)
    monkeypatch.setattr(worker, "COMPONENT_LIMIT", 2 * 1024**2)
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _path: Namespace(free=10 * 1024**3))
    output = tmp_path / "probe"
    worker.run(
        Namespace(
            dataset="cifar10",
            seed=0,
            method=method,
            arm=arm,
            probe=True,
            device="cpu",
            output=output,
        )
    )

    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "accepted.json").read_text(encoding="utf-8"))
    priors = {row["consumer"]: row for row in manifest["research_settings"]["prior"]}
    selection = "PA+clean-support" if support is not None else "PA"
    if consumer is None:
        assert all(row["partition"] != "train" for row in priors.values())
        assert request["class_prior_training"] is None
    else:
        training = priors[consumer]
        expected_prior = population_prior
        target = "positive_union_unlabeled"
        if support is not None:
            observed, _ = worker.SCARGenerator().generate(
                bundle.train.X, bundle.train.labels, worker.C, 0
            )
            q = np.mean(observed == 1)
            expected_prior = (population_prior - q) / (1 - q)
            target = "native_unlabeled"
        assert training["target"] == target
        assert training["value"] == pytest.approx(expected_prior)
        assert training["value"] == request["class_prior_training"]
    assert priors["pa_risk"]["value"] == population_prior
    assert priors["pa_risk"]["protocol"] == selection
    assert set(receipt["metrics"]) == {selection, "OA"}
    assert all(row["checkpoint"] for row in manifest["selection"].values())


@pytest.mark.parametrize(
    "uses",
    [
        [],
        [{"target": "positive_union_unlabeled", "value": 0.5}],
        [{"target": "native_unlabeled", "value": 0.4}],
    ],
    ids=["missing-evidence", "wrong-pool", "wrong-value"],
)
def test_edge_prior_audit_refuses_missing_or_mismatched_evidence(uses):
    estimator = Namespace(
        _runner_prior_parameter="risk_prior",
        risk_prior=0.5,
        _prior_consumptions_=[
            {"partition": "train", "source": "estimator.risk_prior", **use} for use in uses
        ],
    )
    with pytest.raises(ValueError, match="did not consume risk_prior"):
        worker._assert_consumed_prior(estimator, "os")


def test_determ_self_pu_arms_share_the_seeded_head_and_parameters():
    torch = pytest.importorskip("torch")
    protocol = json.loads(worker.BASIS.read_text(encoding="utf-8"))
    runs = []
    for arm in ("frozen", "trainable"):
        args = Namespace(method="self_pu", seed=5, arm=arm, device="cpu", probe=True)
        _, params, head_hash = worker._model(args, protocol, torch.nn.Identity(), 0.5)
        runs.append((params, head_hash))

    assert runs[0] == runs[1]
    assert len(runs[0][1]) == 64
