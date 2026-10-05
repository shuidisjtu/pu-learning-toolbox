"""Built-in algorithm registry — native paper methods.

Each entry captures canonical metadata (name, aliases, family, scenario,
assumption, source status, upstream URL, license, etc.) so that the
registry browser and documentation generators have complete
information even before training logic is implemented.

See the method cards under ``docs/research/method_cards/`` for per-method source
status, and ``docs/user/concepts/method_selection.md`` §§2–5 for the algorithm
family taxonomy.
"""

from __future__ import annotations

from ..core.exceptions import RegistryError
from ..core.tags import (
    AlgorithmFamily as Fam,
)
from ..core.tags import (
    Assumption as Asm,
)
from ..core.tags import (
    Backend,
    Maturity,
)
from ..core.tags import (
    ImplementationStatus as Impl,
)
from ..core.tags import (
    Scenario as Scn,
)
from ..core.tags import (
    SourceStatus as Src,
)
from ..core.tags import (
    TrainingCost as Cost,
)
from .metadata import AlgorithmMetadata
from .registry import register_method

# ═════════════════════════════════════════════════════════════════════
# Canonical method list -- this ordering is the authority; the registry is
# the code-side source of truth for algorithm metadata.
# ═════════════════════════════════════════════════════════════════════

# NOTE: a bound entry must NOT repeat the fields its estimator class declares --
# the class is the source, and `_sync_class_metadata_to_registry` overwrites the
# literal at registration, so a literal there is computed and then discarded.
# An entry with no bound class (`api_only`) is the opposite: the literal IS the
# source, so it must declare all eight entry fields (the `_SYNC_FIELDS` members
# that entries carry).
# `tests/test_builtin_methods.py::test_static_entries_do_not_redeclare_class_fields`
# enforces both halves.
# The five `_SYNC_FIELDS` members an entry never carries -- `native_architectures`,
# `input_ndims`, `encoder_parameter`, `trains_encoder`, `label_semantics` -- come
# from the class or the dataclass default.  An `api_only` entry has no class to
# declare them, so it takes the defaults; the guard's first clause rejects one
# written here (known boundary, recorded in docs/dev/single_source_map.md).

_BUILTIN: list[AlgorithmMetadata] = [
    # ── 1. Class-Prior Estimation ──────────────────────────────────
    AlgorithmMetadata(
        # NOTE: km1/km2 are NOT aliases here -- they denote
        # KernelMeanPriorEstimator(variant=...) in PUPipeline (a different
        # algorithm), so binding them to ClassPriorEstimator would repeat
        # the kldce alias bug.
        name="class_prior_estimation",
        aliases=["cpe", "pen_l1", "pe"],
        deprecated_aliases=["pe"],
        family=Fam.CLASS_PRIOR_ESTIMATION,
        paper="Class-Prior Estimation for Learning from Positive and Unlabeled Data",
        scenario=[Scn.SINGLE_TRAINING_SET, Scn.CASE_CONTROL],
        assumption=[Asm.SCAR],
        requires_class_prior=False,
        supports_sparse=False,
        supports_gpu=False,
        backend=Backend.NUMPY,
        maturity=Maturity.STABLE,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_RELATED,
        upstream_url="http://www.mcduplessis.com/index.php/software/",
        license="unknown",
        training_cost=Cost.LOW,  # convex scipy solve over a coarse grid
    ),
    # ── 2. ReCPE ───────────────────────────────────────────────────
    AlgorithmMetadata(
        name="recpe",
        aliases=["re_cpe", "rethinking_cpe"],
        paper="Rethinking Class-Prior Estimation for Positive-Unlabeled Learning",
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://github.com/a5507203/Rethinking-Class-Prior-Estimation-for-Positive-Unlabeled-Learning",
        license="MIT",
        training_cost=Cost.LOW,  # convex scipy solve
    ),
    # ── 3. Elkan-Noto ──────────────────────────────────────────────
    AlgorithmMetadata(
        name="elkan_noto",
        aliases=["en", "elkan-noto", "elkan_noto_calibration"],
        paper="Learning Classifiers from Only Positive and Unlabeled Data",
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://github.com/pulearn/pulearn",
        license="BSD-3-Clause",
        training_cost=Cost.LOW,  # sklearn LogisticRegression wrapper
    ),
    # ── 4. Convex PU / uPU ─────────────────────────────────────────
    AlgorithmMetadata(
        name="upu",
        aliases=["convex_pu", "unbiased_pu", "u-pu"],
        paper="Convex Formulation for Learning from Positive and Unlabeled Data",
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://github.com/t-sakai-kure/pywsl",
        license="MIT",
        training_cost=Cost.LOW,  # convex objective, scipy minimize
    ),
    # ── 5. nnPU ────────────────────────────────────────────────────
    AlgorithmMetadata(
        name="nnpu",
        aliases=["non_negative_pu", "nn-pu", "nnPU"],
        paper="Positive-Unlabeled Learning with Non-Negative Risk Estimator",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/kiryor/nnPUlearning",
        license="MIT",
        training_cost=Cost.MEDIUM,  # torch, fixed 200 epochs
    ),
    # ── 6. PNU ─────────────────────────────────────────────────────
    AlgorithmMetadata(
        name="pnu",
        aliases=["pnu_classifier", "pn-pu-nu"],
        paper=(
            "Semi-supervised Classification Based on Classification "
            "from Positive and Unlabeled Data"
        ),
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://github.com/t-sakai-kure/pywsl",
        license="MIT",
        training_cost=Cost.LOW,  # closed-form linear solve
    ),
    # ── 7. Centroid Estimation / LDCE / KLDCE ──────────────────────
    AlgorithmMetadata(
        name="centroid_pu",
        aliases=["ldce", "centroid_estimation"],
        paper="Loss Decomposition and Centroid Estimation for Positive and Unlabeled Learning",
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://gcatnjust.github.io/ChenGong/code/CEGE_PAMI20.rar",
        license="unknown",
        training_cost=Cost.MEDIUM,  # fixed 10000-iteration alternating scheme
    ),
    AlgorithmMetadata(
        name="kldce",
        aliases=["kernelized_ldce"],
        paper=(
            "Loss Decomposition and Centroid Estimation for Positive and "
            "Unlabeled Learning (kernelized version, RBF)"
        ),
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://gcatnjust.github.io/ChenGong/code/CEGE_PAMI20.rar",
        license="unknown",
        training_cost=Cost.MEDIUM,  # 100 iterations, QP oracle on RBF
    ),
    # ── 8. LLSVM ───────────────────────────────────────────────────
    AlgorithmMetadata(
        name="llsvm",
        aliases=["large_margin_svm", "label_calibrated_svm"],
        paper=(
            "Large-Margin Label-Calibrated Support Vector Machines "
            "for Positive and Unlabeled Learning"
        ),
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://gcatnjust.github.io/ChenGong/code/LLSVM_TNNLS19.rar",
        license="unknown",
        training_cost=Cost.HIGH,  # fixed 3000-epoch non-convex SGD
    ),
    # ── 9. Dist-PU ─────────────────────────────────────────────────
    AlgorithmMetadata(
        name="dist_pu",
        aliases=["distribution_pu", "distpu"],
        paper="Dist-PU: Positive-Unlabeled Learning from a Label Distribution Perspective",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective",
        license="MIT",
        training_cost=Cost.MEDIUM,  # torch, 100 epochs
    ),
    # ── 9a. Variational PU (experimental tabular adapter) ───────────
    AlgorithmMetadata(
        name="vpu",
        aliases=["variational_pu"],
        paper="A Variational Approach for Learning from Positive and Unlabeled Data",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/HC-Feynman/vpu",
        license="MIT",
        training_cost=Cost.MEDIUM,
    ),
    # ── 9b. PULDA (experimental tabular adapter) ───────────────────
    AlgorithmMetadata(
        name="pulda",
        aliases=["label_distribution_alignment"],
        paper="Positive-Unlabeled Learning with Label Distribution Alignment",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/jiangyangby/PULDA",
        license="MIT",
        training_cost=Cost.HIGH,
    ),
    # ── 10. PUSB ───────────────────────────────────────────────────
    AlgorithmMetadata(
        name="pusb",
        aliases=["biased_pu", "selection_bias_pu", "nnPUSB"],
        paper="Learning from Positive and Unlabeled Data with a Selection Bias",
        requires_class_prior=False,
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://github.com/MasaKat0/PUlearning",
        license="MIT",
        training_cost=Cost.MEDIUM,  # sklearn LR, max_iter=1000
    ),
    # ── 10a. PUSB kernel (official-aligned RBF adapter) ─────────────
    AlgorithmMetadata(
        name="pusb_kernel",
        aliases=["kernelized_pusb"],
        paper=(
            "Learning from Positive and Unlabeled Data with a Selection "
            "Bias (kernelized version, RBF)"
        ),
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://github.com/MasaKat0/PUlearning",
        license="MIT",
        training_cost=Cost.HIGH,  # full (sigma x reg) grid CV + refit
    ),
    # ── 11. LBE ────────────────────────────────────────────────────
    AlgorithmMetadata(
        name="lbe",
        aliases=["labeling_bias", "labeling_bias_estimation"],
        paper="Instance-Dependent Positive and Unlabeled Learning with Labeling Bias Estimation",
        requires_class_prior=False,
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://gcatnjust.github.io/ChenGong/code/LBE_TPAMI21.rar",
        license="needs_review",
        training_cost=Cost.MEDIUM,  # sklearn LR, max_iter=1000 + EM loop
    ),
    # ── 12. Self-PU ────────────────────────────────────────────────
    AlgorithmMetadata(
        name="self_pu",
        aliases=["self_pu_classifier"],
        paper="Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/VITA-Group/Self-PU",
        license="MIT",
        training_cost=Cost.MEDIUM,  # torch, 200 epochs
    ),
    # ── 13. InfoMax PU ─────────────────────────────────────────────
    AlgorithmMetadata(
        name="infomax_pu",
        aliases=["information_theoretic_pu", "pu_representation"],
        paper="Information-Theoretic Representation Learning for Positive-Unlabeled Classification",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url=None,
        license=None,
        training_cost=Cost.HIGH,  # torch, ~600 epochs (two-stage)
    ),
    # ── 14. Weighted Contrastive PU ────────────────────────────────
    AlgorithmMetadata(
        name="weighted_contrastive_pu",
        aliases=["wcon_pu", "wconpu", "contrastive_pu"],
        paper=(
            "Weighted Contrastive Learning with Hard Negative Mining "
            "for Positive and Unlabeled Learning"
        ),
        supports_sparse=False,
        supports_gpu=True,
        upstream_url=None,
        license=None,
        training_cost=Cost.HIGH,  # torch, 100 epochs
    ),
    # ── 15. DGPU ───────────────────────────────────────────────────
    AlgorithmMetadata(
        name="dgpu",
        aliases=["discriminative_generative_pu"],
        paper="Discriminative-Generative Positive and Unlabeled Learning",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url=None,
        license=None,
        training_cost=Cost.MEDIUM,  # torch, 200 epochs + generative sampling
    ),
    # ── 16. GradPU ──────────────────────────────────────────────────
    AlgorithmMetadata(
        name="gradpu",
        aliases=["grad_pu"],
        paper="GradPU: Positive-Unlabeled Learning via Gradient Penalty and Positive Upweighting",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url=None,
        license=None,
        training_cost=Cost.HIGH,  # second-order input-gradient penalty
    ),
    # ── 17. PU Extra Trees ──────────────────────────────────────────
    AlgorithmMetadata(
        name="puet",
        aliases=["pu_extra_trees"],
        paper=(
            "Positive-Unlabeled Learning using Random Forests "
            "via Recursive Greedy Risk Minimization"
        ),
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://github.com/jonathanwilton/PUExtraTrees",
        license="MIT",
        training_cost=Cost.HIGH,  # default 100-tree CPU forest
    ),
    AlgorithmMetadata(
        name="robust_pu",
        aliases=["robust-pu"],
        paper="Robust Positive-Unlabeled Learning via Noise Negative Sample Self-correction",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/woriazzc/Robust-PU",
        license="unknown",
        training_cost=Cost.HIGH,
    ),
    AlgorithmMetadata(
        name="cvir",
        aliases=["conditional_value_ignoring_risk"],
        paper="Mixture Proportion Estimation and PU Learning: A Modern Approach",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/acmi-lab/PU_learning",
        license="Apache-2.0",
        training_cost=Cost.MEDIUM,
    ),
    AlgorithmMetadata(
        name="split_pu",
        aliases=["split-pu"],
        paper="Split-PU: Hardness-aware Training Strategy for Positive-Unlabeled Learning",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/loadder/SplitPU_MM2022",
        license="unknown",
        training_cost=Cost.HIGH,
    ),
    AlgorithmMetadata(
        name="lagam",
        aliases=["la_gam"],
        paper="Positive-Unlabeled Learning by Latent Group-Aware Meta Disambiguation",
        requires_clean_support=True,
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/llong-cs/LaGAM",
        license="unknown",
        training_cost=Cost.HIGH,
    ),
]


def register_all_builtin_methods() -> int:
    """Register all paper methods and bind native implementations.

    Returns the number of methods newly registered.  Idempotent —
    methods already present in the registry are skipped, so calling
    this repeatedly (e.g. from the recommender and the workflow
    pipeline entry points) is safe without clearing the registry.

    Native implementations (those with ``implementation_status=NATIVE``)
    are automatically bound to their registry entries.
    """
    count = 0
    for meta in _BUILTIN:
        try:
            register_method(meta)
            count += 1
        except RegistryError:
            # Already registered by an earlier caller; skip.
            continue

    # Bind native estimator classes to their registry entries.
    _bind_native_classes()
    return count


def _bind_native_classes() -> None:
    """Lazy-import and bind native estimator classes.

    Called automatically by :func:`register_all_builtin_methods`.
    Safe to call repeatedly — each method checks individually
    whether it is already bound.

    When adding a new NATIVE method, add an entry to
    ``_NATIVE_IMPORTS`` below.
    """
    from .registry import _CLASSES, bind_estimator_class

    _native_imports: list[tuple[str, str, str]] = [
        # (canonical_name, module_path, class_name)
        ("elkan_noto", "..estimators.classic.elkan_noto", "ElkanNotoClassifier"),
        ("upu", "..estimators.risk.upu", "UPUClassifier"),
        ("nnpu", "..estimators.risk.nnpu", "NonNegativePUClassifier"),
        ("pnu", "..estimators.risk.pnu", "PNUClassifier"),
        ("recpe", "..prior.recpe", "ReCPEEstimator"),
        ("class_prior_estimation", "..prior.pen_l1", "ClassPriorEstimator"),
        ("dist_pu", "..estimators.risk.dist_pu", "DistPUClassifier"),
        ("vpu", "..estimators.risk.vpu", "VPUClassifier"),
        ("pulda", "..estimators.risk.pulda", "PULDAClassifier"),
        ("pusb", "..estimators.bias_aware.pusb", "PUSBClassifier"),
        ("pusb_kernel", "..estimators.bias_aware.pusb_kernel", "PUSBKernelClassifier"),
        ("lbe", "..estimators.bias_aware.lbe", "LBEClassifier"),
        ("centroid_pu", "..estimators.risk.ldce", "LDCEClassifier"),
        ("kldce", "..estimators.risk.kldce", "KLDCEClassifier"),
        ("llsvm", "..estimators.classic.llsvm", "LLSVMClassifier"),
        ("self_pu", "..estimators.deep.self_pu", "SelfPUClassifier"),
        ("infomax_pu", "..estimators.deep.infomax_pu", "InfoMaxPUClassifier"),
        (
            "weighted_contrastive_pu",
            "..estimators.deep.weighted_contrastive_pu",
            "WeightedContrastivePUClassifier",
        ),
        ("dgpu", "..estimators.deep.dgpu", "DGPUClassifier"),
        ("gradpu", "..estimators.deep.grad_pu", "GradPUClassifier"),
        ("puet", "..estimators.risk.puet", "PUExtraTreesClassifier"),
        ("robust_pu", "..estimators.deep.robust_pu", "RobustPUClassifier"),
        ("cvir", "..estimators.risk.cvir", "CVIRClassifier"),
        ("split_pu", "..estimators.deep.split_pu", "SplitPUClassifier"),
        ("lagam", "..estimators.deep.lagam", "LaGAMClassifier"),
    ]

    for canonical_name, module_path, class_name in _native_imports:
        if canonical_name in _CLASSES:
            continue  # Already bound
        import importlib

        mod = importlib.import_module(module_path, __package__)
        cls = getattr(mod, class_name)
        bind_estimator_class(canonical_name, cls)
