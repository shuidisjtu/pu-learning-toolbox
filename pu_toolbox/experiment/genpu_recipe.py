# ruff: noqa: N803
"""Explicit, single-candidate GenPU engineering recipe; not Survey admission.

The defaults of GenPU itself are untouched. This opt-in recipe fixes the loss,
network topology and selection stages while requiring an explicit compute
budget. Formal v3 units and P4.1-B approval remain separate obligations.
"""

from __future__ import annotations

import copy
import math
from numbers import Real
from pathlib import Path

from ..utils.serialization import strict_canonical_hash
from .checkpoints import EpochCheckpointTrainer
from .feature_adapter import encoder_state_sha256


def build_genpu_recipe(*, prior, seed, device, gan_epochs, pn_epochs, batch_size, image=None):
    """Build a fixed-topology TS recipe with explicit, unapproved run budgets.

    ``image`` is a train-fitted SurveyImagePreprocessing, or None for 2-D data.
    Pixels remain in the shared [0,1] input coordinates; normalization is inside
    all four independent CNNs. Identity outputs avoid clipping or silently
    imposing the source demo's different tanh input coordinates.
    """
    if (
        isinstance(prior, bool)
        or not isinstance(prior, Real)
        or not math.isfinite(prior)
        or not 0 < prior < 1
    ):
        raise ValueError("recipe requires a finite population prior in (0, 1)")
    for name, value in (
        ("gan_epochs", gan_epochs),
        ("pn_epochs", pn_epochs),
        ("batch_size", batch_size),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if device not in ("cpu", "cuda"):
        raise ValueError("recipe device must be cpu or cuda; no automatic fallback")
    if image is not None and image.train_augmentation != "none":
        raise ValueError("GenPU recipe requires train_augmentation=none")

    import torch

    from ..estimators.deep.gen_pu import GenPUClassifier
    from .image import build_survey_image_encoder

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable; refusing CPU fallback")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        encoder = build_survey_image_encoder(image) if image is not None else None
    parameters = {
        "class_prior": float(prior),
        "random_state": seed,
        "device": device,
        "hidden_dim": 256,
        "latent_dim": 100,
        "max_epochs": gan_epochs,
        "classifier_epochs": pn_epochs,
        "batch_size": batch_size,
        "learning_rate": 3e-4,
        "positive_weight": 1.0,
        "negative_weight": 1.0,
        "unlabeled_weight": 1.0,
        "generator_output": "identity",
    }
    recipe = _recipe_payload(
        parameters,
        image.to_manifest() if image is not None else None,
        encoder_state_sha256(encoder) if encoder is not None else None,
    )
    recipe["recipe_sha256"] = strict_canonical_hash(recipe)
    return GenPUClassifier(encoder=encoder, **parameters), recipe


def _recipe_payload(parameters, image, encoder_digest):
    """Single declaration used by construction and semantic validation."""
    return {
        "schema_version": "genpu-engineering-recipe-v1",
        "method": "genpu",
        "formal_admission": False,
        "formal_budget_approved": False,
        "owner_signatures": [],
        "adaptation": "benchmark-adapted_not_paper_numerical_reproduction",
        "parameters": parameters,
        "run_view": "ts-compatible",
        "objective": "paper_minimax_prior_weighted",
        "synthetic_pn_risk": "pi*mean(softplus(-C(Gp(z))))+(1-pi)*mean(softplus(C(Gn(z))))",
        "sampling": "Du_real_uniform_full_train_union; P_uniform_with_replacement",
        "optimizer": {"name": "Adam", "betas": [0.9, 0.999], "weight_decay": 0},
        "generator": {
            "hidden_layers": [256, 256],
            "activation": "LeakyReLU(0.2)",
            "output": "identity",
        },
        "heads": {"Dp": [], "Dn": [], "Du": [256, 256], "PN": [256, 256]},
        "initialization": (
            "seeded_torch_defaults; four_encoder_copies_from_same_untrained_template; fresh_PN"
        ),
        "encoder_initial_state_sha256": encoder_digest,
        "image_preprocessing": image,
        "generator_step_D_mode": (
            "weights_frozen_eval_BN_dropout; input_gradients_enabled; finally_restore"
        ),
        "selection": {
            "protocols": ["PA", "OA"],
            "eligible_stage": "synthetic_pn",
            "epochs": "all_completed_PN_epochs",
            "GAN_eligible": False,
        },
        "retention": "selected_PN_weights_only; all_epoch_metadata; no_GAN_or_optimizer_resume",
    }


class GenPURecipeTrainer(EpochCheckpointTrainer):
    """Enforce the recipe, stage costs and actual snapshot bounds before acceptance."""

    records_epoch_checkpoints = True

    def __init__(self, recipe, *, checkpoint_dir=None):
        super().__init__(checkpoint_dir=checkpoint_dir)
        self.recipe = copy.deepcopy(recipe)

    def _validate(self, estimator, view):
        from ..estimators.deep.gen_pu import GenPUClassifier

        payload = {key: value for key, value in self.recipe.items() if key != "recipe_sha256"}
        if self.recipe.get("recipe_sha256") != strict_canonical_hash(payload):
            raise ValueError("GenPU recipe identity changed")
        if not isinstance(estimator, GenPUClassifier) or view != "ts":
            raise ValueError("GenPU recipe requires GenPUClassifier and explicit ts view")
        image = self.recipe["image_preprocessing"]
        # A valid digest alone cannot authorize changes to stage/role semantics.
        p = self.recipe["parameters"]
        if strict_canonical_hash(payload) != strict_canonical_hash(
            _recipe_payload(p, image, self.recipe["encoder_initial_state_sha256"])
        ):
            raise ValueError("GenPU fixed recipe semantics changed")
        fixed = {
            "hidden_dim": 256,
            "latent_dim": 100,
            "learning_rate": 3e-4,
            "positive_weight": 1.0,
            "negative_weight": 1.0,
            "unlabeled_weight": 1.0,
            "generator_output": "identity",
        }
        if any(p.get(key) != value for key, value in fixed.items()):
            raise ValueError("GenPU fixed recipe parameters changed")
        if set(p) != set(fixed) | {
            "class_prior",
            "random_state",
            "device",
            "max_epochs",
            "classifier_epochs",
            "batch_size",
        }:
            raise ValueError("GenPU recipe parameter set changed")
        for key in ("max_epochs", "classifier_epochs", "batch_size"):
            if type(p[key]) is not int or p[key] < 1:
                raise ValueError("GenPU recipe has an invalid integer budget")
        actual = estimator.get_params(deep=False)
        if any(
            type(actual.get(key)) is not type(value) or actual[key] != value
            for key, value in p.items()
        ):
            raise ValueError("GenPU candidate parameters differ from recipe")
        encoder = estimator.encoder
        if (encoder is None) != (image is None):
            raise ValueError("GenPU image/encoder recipe mismatch")
        if (
            encoder is not None
            and encoder_state_sha256(encoder) != self.recipe["encoder_initial_state_sha256"]
        ):
            raise ValueError("GenPU encoder initialization differs from recipe")

    def validate_experiment(self, model, bundle, config, protocols):
        """Reject formal units, candidate overrides and missing storage estimates."""
        self._validate(model, config.get("os_or_ts"))
        if config.get("survey_protocol") is not None or config.get("recipe_registry") is not None:
            raise ValueError("GenPU engineering recipe is not a formal Survey recipe")
        if config.get("candidates", [{}]) != [{}]:
            raise ValueError("GenPU recipe accepts exactly one candidate without overrides")
        if config.get("capture_epoch_checkpoints", True) is not True:
            raise ValueError("GenPU recipe requires completed PN checkpoints")
        size = config.get("checkpoint_bytes_per_component")
        if type(size) is not int or size < 1:
            raise ValueError("GenPU recipe requires checkpoint_bytes_per_component")
        self._bytes_bound = size
        from .strategies import ProtocolOA, ProtocolPA

        if [type(protocol) for protocol in protocols] != [ProtocolPA, ProtocolOA]:
            raise ValueError("GenPU recipe requires independent PA then OA selection")

    def validate_generation(self, train_meta, val_meta):
        """No SAR adaptation is approved by this engineering recipe."""
        if any(meta.get("mechanism") != "scar" for meta in (train_meta, val_meta)):
            raise ValueError("GenPU recipe supports SCAR only")

    def bind_checkpoint_directory(self, root):
        """Share the runner's exact preflight filesystem and writer location."""
        self.checkpoint_dir = root

    def checkpoint_epoch_bound(self):
        """GAN has no deployable classifier and creates no candidate snapshots."""
        return self.recipe["parameters"]["classifier_epochs"]

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        """Train once without external validation; verify all six paid stages."""
        self._validate(estimator, os_or_ts)
        if class_prior != self.recipe["parameters"]["class_prior"]:
            raise ValueError("GenPU runner population prior differs from recipe")
        result = super().fit(estimator, X, y, class_prior=class_prior, os_or_ts=os_or_ts)
        steps = math.ceil(len(X) / estimator.batch_size)
        gan = steps * estimator.max_epochs
        expected = dict.fromkeys(
            (
                "positive_discriminator",
                "negative_discriminator",
                "unlabeled_discriminator",
                "positive_generator",
                "negative_generator",
            ),
            gan,
        )
        expected["synthetic_pn"] = steps * estimator.classifier_epochs
        if estimator.stage_optimizer_steps_ != expected or estimator.optimizer_steps_ != sum(
            expected.values()
        ):
            raise ValueError("GenPU six-stage optimizer accounting differs from recipe")
        if len(result.checkpoints) != estimator.classifier_epochs:
            raise ValueError("GenPU recipe requires every completed PN checkpoint")
        for index, checkpoint in enumerate(result.checkpoints, 1):
            context = checkpoint.training_context
            if context != {
                "stage": "synthetic_pn",
                "stage_epoch": index,
                "round_index": None,
                "optimizer_steps": 5 * gan + steps * index,
            }:
                raise ValueError("GenPU stage eligibility or cumulative cost differs from recipe")
            if (
                hasattr(self, "_bytes_bound")
                and Path(checkpoint.path).stat().st_size > self._bytes_bound
            ):
                raise ValueError("GenPU checkpoint exceeded declared byte bound")
        estimator.training_provenance_ = {
            "recipe": copy.deepcopy(self.recipe),
            "steps_per_epoch": steps,
            "stage_optimizer_steps": expected,
            "optimizer_steps": estimator.optimizer_steps_,
            "checkpoint_count": len(result.checkpoints),
            "checkpoint_bytes": [
                Path(checkpoint.path).stat().st_size for checkpoint in result.checkpoints
            ],
            "six_network_state_bytes": {
                name: sum(
                    t.numel() * t.element_size()
                    for t in getattr(estimator, name).state_dict().values()
                )
                for name in (
                    "positive_generator_",
                    "negative_generator_",
                    "positive_discriminator_",
                    "negative_discriminator_",
                    "unlabeled_discriminator_",
                    "model_",
                )
            },
            "external_validation_forwarded_to_fit": False,
            "formal_admission": False,
        }
        return result
