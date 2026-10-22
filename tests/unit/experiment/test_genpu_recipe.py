# ruff: noqa: N803, N806
"""Actual fits, paid-stage accounting and early refusal of recipe drift."""

import copy
import functools
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.gen_pu import GenPUClassifier  # noqa: E402
from pu_toolbox.experiment.bundle import DatasetPart  # noqa: E402
from pu_toolbox.experiment.genpu_recipe import GenPURecipeTrainer, build_genpu_recipe  # noqa: E402
from pu_toolbox.experiment.image import fit_survey_image_preprocessing  # noqa: E402
from pu_toolbox.experiment.runner import ExperimentRunner  # noqa: E402
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA  # noqa: E402
from pu_toolbox.utils.serialization import strict_canonical_hash  # noqa: E402

pytestmark = pytest.mark.unit


def setup(**kwargs):
    return build_genpu_recipe(
        **(dict(prior=0.4, seed=0, device="cpu", gan_epochs=1, pn_epochs=3, batch_size=8) | kwargs)
    )


def parts():
    rng = np.random.RandomState(0)
    return tuple(
        DatasetPart(
            rng.normal(size=(n, 3)).astype("float32"),
            np.arange(n) % 2,
            "clean",
            np.arange(100 * i, 100 * i + n),
            i != 3,
        )
        for i, n in enumerate((24, 16, 12, 12))
    )


def config(recipe, **kwargs):
    return {
        "c": 0.5,
        "os_or_ts": "ts",
        "trainer": GenPURecipeTrainer(recipe),
        "checkpoint_bytes_per_component": 2**20,
        "reclaim_unselected_checkpoints": True,
    } | kwargs


def test_basic_real_single_fit_role_isolation_six_stage_cost_selection_and_retention(
    tmp_path, monkeypatch
):
    inputs = parts()
    before = [(part.X.copy(), part.labels.copy()) for part in inputs]
    original, calls = GenPUClassifier.fit, []

    @functools.wraps(original)
    def audit(self, X, y, **kwargs):
        calls.append((X.copy(), y.copy(), dict(kwargs)))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(GenPUClassifier, "fit", audit)
    model, recipe = setup()
    result = ExperimentRunner(
        seed=0,
        class_prior=0.4,
        protocols=[ProtocolPA(), ProtocolOA()],
        config=config(recipe),
        manifest_path=str(tmp_path / "manifest.json"),
    ).fit(model, *inputs)
    assert len(calls) == 1 and not result.failures
    assert set(calls[0][2]) == {"class_prior", "os_or_ts", "epoch_callback"}
    assert calls[0][2]["os_or_ts"] == "ts"
    np.testing.assert_array_equal(calls[0][0], inputs[0].X)
    assert calls[0][1].sum() == 6 < inputs[0].labels.sum()
    candidate = result.manifest["candidate_runs"][0]
    training = candidate["training_provenance"]
    assert training["recipe"] == recipe and not training["formal_admission"]
    assert training["stage_optimizer_steps"] == dict(
        positive_discriminator=3,
        negative_discriminator=3,
        unlabeled_discriminator=3,
        positive_generator=3,
        negative_generator=3,
        synthetic_pn=9,
    )
    assert training["optimizer_steps"] == 24
    assert len(training["six_network_state_bytes"]) == 6
    assert all(size > 0 for size in training["six_network_state_bytes"].values())
    assert all(0 < size <= 2**20 for size in training["checkpoint_bytes"])
    checkpoints = candidate["epoch_checkpoints"]
    assert [row["training_context"]["optimizer_steps"] for row in checkpoints] == [18, 21, 24]
    assert set(result.selected_models) == {"PA", "OA"}
    assert sum(not row["reclaimed"] for row in checkpoints) <= 2
    for row in checkpoints:
        assert Path(row["path"]).is_file() is (not row["reclaimed"])
        assert Path(row["path"]).is_relative_to(tmp_path / "checkpoints")
    assert not result.manifest["formal_eligible"]
    for part, (X, labels) in zip(inputs, before, strict=True):
        np.testing.assert_array_equal(part.X, X)
        np.testing.assert_array_equal(part.labels, labels)


@pytest.mark.parametrize("seed", [0, 7])
def test_determ_recording_does_not_change_plain_training(seed, tmp_path):
    model, recipe = setup(seed=seed)
    plain = copy.deepcopy(model)
    X = parts()[0].X
    y = np.r_[np.ones(6, int), np.zeros(18, int)]
    plain.fit(X, y, class_prior=0.4, os_or_ts="ts")
    trajectory = GenPURecipeTrainer(recipe, checkpoint_dir=tmp_path).fit(
        model, X, y, class_prior=0.4, os_or_ts="ts"
    )
    assert model.history_ == plain.history_ and model.optimizer_steps_ == plain.optimizer_steps_
    np.testing.assert_array_equal(model.decision_function(X), plain.decision_function(X))
    for name in model.stage_optimizer_steps_:
        attribute = "model_" if name == "synthetic_pn" else name + "_"
        for key, value in getattr(model, attribute).state_dict().items():
            torch.testing.assert_close(
                value, getattr(plain, attribute).state_dict()[key], rtol=0, atol=0
            )
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(X), plain.decision_function(X)
    )


@pytest.mark.parametrize(
    "options",
    [
        {"os_or_ts": "os"},
        {"candidates": [{}, {}]},
        {"candidates": [{"max_epochs": 2}]},
        {"survey_protocol": {}},
        {"recipe_registry": {}},
        {"capture_epoch_checkpoints": False},
        {"checkpoint_bytes_per_component": True},
    ],
)
def test_param_invalid_contract_fails_before_any_fit(options, monkeypatch):
    model, recipe = setup()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid contract reached fitting")

    monkeypatch.setattr(GenPUClassifier, "fit", forbidden)
    with pytest.raises(ValueError):
        config(recipe)["trainer"].validate_experiment(model, None, config(recipe, **options), [])


@pytest.mark.parametrize(
    "field,value",
    [
        ("formal_admission", True),
        ("formal_admission", 0),
        ("retention", "keep_all_GAN"),
        ("objective", "source_demo"),
    ],
)
def test_edge_rehashing_cannot_approve_or_change_fixed_semantics(field, value):
    model, recipe = setup()
    recipe[field] = value
    recipe["recipe_sha256"] = strict_canonical_hash(
        {k: v for k, v in recipe.items() if k != "recipe_sha256"}
    )
    with pytest.raises(ValueError, match="semantics"):
        GenPURecipeTrainer(recipe).validate_experiment(model, None, config(recipe), [])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prior": True},
        {"prior": np.nan},
        {"prior": 0},
        {"seed": True},
        {"batch_size": 0},
        {"gan_epochs": 1.0},
        {"pn_epochs": False},
        {"device": "auto"},
    ],
)
def test_param_invalid_explicit_requests(kwargs):
    with pytest.raises(ValueError):
        setup(**kwargs)


def test_edge_missing_cuda_is_not_a_cpu_success(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA explicitly requested"):
        setup(device="cuda")


def test_edge_changed_candidate_and_sar_are_rejected():
    model, recipe = setup()
    model.set_params(hidden_dim=128)
    trainer = GenPURecipeTrainer(recipe)
    with pytest.raises(ValueError, match="parameters differ"):
        trainer.validate_experiment(model, None, config(recipe), [])
    with pytest.raises(ValueError, match="SCAR only"):
        trainer.validate_generation({"mechanism": "scar"}, {"mechanism": "sar_lbe_a"})


def test_edge_selection_cannot_be_silently_reduced_to_terminal_oa():
    model, recipe = setup()
    with pytest.raises(ValueError, match="independent PA then OA"):
        GenPURecipeTrainer(recipe).validate_experiment(model, None, config(recipe), [ProtocolOA()])


def test_edge_actual_oversize_checkpoint_is_refused_and_kept_for_audit(tmp_path):
    model, recipe = setup()
    trainer = GenPURecipeTrainer(recipe, checkpoint_dir=tmp_path)
    trainer.validate_experiment(
        model, None, config(recipe, checkpoint_bytes_per_component=1), [ProtocolPA(), ProtocolOA()]
    )
    X = parts()[0].X
    y = np.r_[np.ones(6, int), np.zeros(18, int)]
    with pytest.raises(ValueError, match="exceeded declared byte bound"):
        trainer.fit(model, X, y, class_prior=0.4, os_or_ts="ts")
    assert len(list(tmp_path.rglob("*.pt"))) == 3


def test_basic_full_width_image_recipe_has_train_fitted_normalization_and_is_seeded():
    X = np.random.RandomState(1).randint(0, 256, (8, 3, 32, 32), dtype=np.uint8)
    _, image = fit_survey_image_preprocessing(X, dataset="cifar10", train_augmentation="none")
    before = torch.random.get_rng_state().clone()
    model, recipe = setup(image=image)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    assert (
        recipe["image_preprocessing"]["normalization"]["source"] == "train_only_channel_statistics"
    )
    assert recipe["image_preprocessing"]["backbone"]["small_input_stem"]
    assert sum(p.numel() for p in model.encoder.parameters()) > 11_000_000
    assert model.generator_output == "identity" and model.hidden_dim == 256
    GenPURecipeTrainer(recipe).validate_experiment(
        model, None, config(recipe), [ProtocolPA(), ProtocolOA()]
    )
