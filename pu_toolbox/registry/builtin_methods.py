"""Built-in algorithm registry — 16 native paper methods.

Each entry captures canonical metadata (name, aliases, family, scenario,
assumption, source status, upstream URL, license, etc.) so that the
registry browser and documentation generators have complete
information even before training logic is implemented.

See ``docs/resources_optimized.md`` for the full source inventory and
``docs/method_selection.md`` §§2–5 for the algorithm family taxonomy.
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
# Canonical method list (order follows resources_optimized.md §4)
# ═════════════════════════════════════════════════════════════════════

_BUILTIN: list[AlgorithmMetadata] = [
    # ── 1. Class-Prior Estimation ──────────────────────────────────
    AlgorithmMetadata(
        # NOTE: km1/km2 are NOT aliases here -- they denote
        # KernelMeanPriorEstimator(variant=...) in PUPipeline (a different
        # algorithm), so binding them to ClassPriorEstimator would repeat
        # the kldce alias bug.
        name="class_prior_estimation",
        aliases=["cpe", "pen_l1", "pe"],
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
        family=Fam.CLASS_PRIOR_ESTIMATION,
        paper="Rethinking Class-Prior Estimation for Positive-Unlabeled Learning",
        scenario=[Scn.SINGLE_TRAINING_SET, Scn.CASE_CONTROL],
        assumption=[Asm.SCAR],
        requires_class_prior=False,
        supports_sparse=False,
        supports_gpu=False,
        backend=Backend.NUMPY,
        maturity=Maturity.STABLE,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_EXACT,
        upstream_url="https://github.com/a5507203/Rethinking-Class-Prior-Estimation-for-Positive-Unlabeled-Learning",
        license="MIT",
        training_cost=Cost.LOW,  # convex scipy solve
    ),
    # ── 3. Elkan-Noto ──────────────────────────────────────────────
    AlgorithmMetadata(
        name="elkan_noto",
        aliases=["en", "elkan-noto", "elkan_noto_calibration"],
        family=Fam.CLASSIC_CALIBRATION,
        paper="Learning Classifiers from Only Positive and Unlabeled Data",
        scenario=[Scn.SINGLE_TRAINING_SET],
        assumption=[Asm.SCAR],
        requires_class_prior=False,
        supports_sparse=False,
        supports_gpu=False,
        backend=Backend.SKLEARN,
        maturity=Maturity.STABLE,
        implementation_status=Impl.NATIVE,
        source_status=Src.THIRD_PARTY_ONLY,
        upstream_url="https://github.com/pulearn/pulearn",
        license="BSD-3-Clause",
        training_cost=Cost.LOW,  # sklearn LogisticRegression wrapper
    ),
    # ── 4. Convex PU / uPU ─────────────────────────────────────────
    AlgorithmMetadata(
        name="upu",
        aliases=["convex_pu", "unbiased_pu", "u-pu"],
        family=Fam.RISK_ESTIMATION,
        paper="Convex Formulation for Learning from Positive and Unlabeled Data",
        scenario=[Scn.CASE_CONTROL],
        assumption=[Asm.SCAR],
        requires_class_prior=False,
        supports_sparse=False,
        supports_gpu=False,
        backend=Backend.NUMPY,
        maturity=Maturity.STABLE,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_BUNDLE,
        upstream_url="https://github.com/t-sakai-kure/pywsl",
        license="MIT",
        training_cost=Cost.LOW,  # convex objective, scipy minimize
    ),
    # ── 5. nnPU ────────────────────────────────────────────────────
    AlgorithmMetadata(
        name="nnpu",
        aliases=["non_negative_pu", "nn-pu", "nnPU"],
        family=Fam.RISK_ESTIMATION,
        paper="Positive-Unlabeled Learning with Non-Negative Risk Estimator",
        scenario=[Scn.CASE_CONTROL],
        assumption=[Asm.SCAR],
        requires_class_prior=True,
        supports_sparse=False,
        supports_gpu=True,
        backend=Backend.TORCH,
        maturity=Maturity.STABLE,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_EXACT,
        upstream_url="https://github.com/kiryor/nnPUlearning",
        license="MIT",
        training_cost=Cost.MEDIUM,  # torch, fixed 200 epochs
    ),
    # ── 6. PNU ─────────────────────────────────────────────────────
    AlgorithmMetadata(
        name="pnu",
        aliases=["pnu_classifier", "pn-pu-nu"],
        family=Fam.RISK_ESTIMATION,
        paper=(
            "Semi-supervised Classification Based on Classification "
            "from Positive and Unlabeled Data"
        ),
        scenario=[Scn.CASE_CONTROL],
        assumption=[Asm.SCAR],
        requires_class_prior=True,
        supports_sparse=False,
        supports_gpu=False,
        backend=Backend.NUMPY,
        maturity=Maturity.RESEARCH,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_EXACT,
        upstream_url="https://github.com/t-sakai-kure/pywsl",
        license="MIT",
        training_cost=Cost.LOW,  # closed-form linear solve
    ),
    # ── 7. Centroid Estimation / LDCE / KLDCE ──────────────────────
    AlgorithmMetadata(
        name="centroid_pu",
        aliases=["ldce", "centroid_estimation"],
        family=Fam.RISK_ESTIMATION,
        paper="Loss Decomposition and Centroid Estimation for Positive and Unlabeled Learning",
        scenario=[Scn.SINGLE_TRAINING_SET],
        assumption=[Asm.SCAR],
        requires_class_prior=False,
        supports_sparse=False,
        supports_gpu=False,
        backend=Backend.NUMPY,
        maturity=Maturity.RESEARCH,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_RELATED,
        upstream_url="https://gcatnjust.github.io/ChenGong/code/CEGE_PAMI20.rar",
        license="unknown",
        training_cost=Cost.MEDIUM,  # fixed 100-iteration alternating scheme
    ),
    AlgorithmMetadata(
        name="kldce",
        aliases=["kernelized_ldce"],
        family=Fam.RISK_ESTIMATION,
        paper=(
            "Loss Decomposition and Centroid Estimation for Positive and "
            "Unlabeled Learning (kernelized version, RBF)"
        ),
        scenario=[Scn.SINGLE_TRAINING_SET],
        assumption=[Asm.SCAR],
        requires_class_prior=False,
        supports_sparse=False,
        supports_gpu=False,
        backend=Backend.NUMPY,
        maturity=Maturity.RESEARCH,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_RELATED,
        upstream_url="https://gcatnjust.github.io/ChenGong/code/CEGE_PAMI20.rar",
        license="unknown",
        training_cost=Cost.MEDIUM,  # 100 iterations, QP oracle on RBF
    ),
    # ── 8. LLSVM ───────────────────────────────────────────────────
    AlgorithmMetadata(
        name="llsvm",
        aliases=["large_margin_svm", "label_calibrated_svm"],
        family=Fam.RISK_ESTIMATION,
        paper=(
            "Large-Margin Label-Calibrated Support Vector Machines "
            "for Positive and Unlabeled Learning"
        ),
        scenario=[Scn.CASE_CONTROL],
        assumption=[Asm.SCAR, Asm.SAR],
        requires_class_prior=True,
        supports_sparse=False,
        supports_gpu=False,
        backend=Backend.NUMPY,
        maturity=Maturity.RESEARCH,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_EXACT,
        upstream_url="https://gcatnjust.github.io/ChenGong/code/LLSVM_TNNLS19.rar",
        license="unknown",
        training_cost=Cost.HIGH,  # fixed 3000-epoch non-convex SGD
    ),
    # ── 9. Dist-PU ─────────────────────────────────────────────────
    AlgorithmMetadata(
        name="dist_pu",
        aliases=["distribution_pu", "distpu"],
        family=Fam.RISK_ESTIMATION,
        paper="Dist-PU: Positive-Unlabeled Learning from a Label Distribution Perspective",
        scenario=[Scn.CASE_CONTROL],
        assumption=[Asm.SCAR],
        requires_class_prior=True,
        supports_sparse=False,
        supports_gpu=True,
        backend=Backend.TORCH,
        maturity=Maturity.RESEARCH,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_EXACT,
        upstream_url="https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective",
        license="MIT",
        training_cost=Cost.MEDIUM,  # torch, 100 epochs
    ),
    # ── 10. PUSB ───────────────────────────────────────────────────
    AlgorithmMetadata(
        name="pusb",
        aliases=["biased_pu", "selection_bias_pu", "nnPUSB"],
        family=Fam.BIAS_AWARE,
        paper="Learning from Positive and Unlabeled Data with a Selection Bias",
        scenario=[Scn.SELECTION_BIASED],
        assumption=[Asm.SAR],
        requires_class_prior=False,
        supports_sparse=False,
        supports_gpu=False,
        backend=Backend.SKLEARN,
        maturity=Maturity.RESEARCH,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_EXACT,
        upstream_url="https://github.com/MasaKat0/PUlearning",
        license="MIT",
        training_cost=Cost.MEDIUM,  # sklearn LR, max_iter=1000
    ),
    # ── 11. LBE ────────────────────────────────────────────────────
    AlgorithmMetadata(
        name="lbe",
        aliases=["labeling_bias", "labeling_bias_estimation"],
        family=Fam.BIAS_AWARE,
        paper="Instance-Dependent Positive and Unlabeled Learning with Labeling Bias Estimation",
        scenario=[Scn.SINGLE_TRAINING_SET, Scn.SELECTION_BIASED],
        assumption=[Asm.SAR],
        requires_class_prior=False,
        supports_sparse=False,
        supports_gpu=False,
        backend=Backend.SKLEARN,
        maturity=Maturity.RESEARCH,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_EXACT,
        upstream_url="https://gcatnjust.github.io/ChenGong/code/LBE_TPAMI21.rar",
        license="needs_review",
        training_cost=Cost.MEDIUM,  # sklearn LR, max_iter=1000 + EM loop
    ),
    # ── 12. Self-PU ────────────────────────────────────────────────
    AlgorithmMetadata(
        name="self_pu",
        aliases=["self_pu_classifier"],
        family=Fam.DEEP_PU,
        paper="Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training",
        scenario=[Scn.CASE_CONTROL],
        assumption=[Asm.SCAR],
        requires_class_prior=True,
        supports_sparse=False,
        supports_gpu=True,
        backend=Backend.TORCH,
        maturity=Maturity.RESEARCH,
        implementation_status=Impl.NATIVE,
        source_status=Src.OFFICIAL_EXACT,
        upstream_url="https://github.com/VITA-Group/Self-PU",
        license="MIT",
        training_cost=Cost.MEDIUM,  # torch, 200 epochs
    ),
    # ── 13. InfoMax PU ─────────────────────────────────────────────
    AlgorithmMetadata(
        name="infomax_pu",
        aliases=["information_theoretic_pu", "pu_representation"],
        family=Fam.DEEP_PU,
        paper="Information-Theoretic Representation Learning for Positive-Unlabeled Classification",
        scenario=[Scn.CASE_CONTROL],
        assumption=[Asm.SCAR],
        requires_class_prior=False,
        supports_sparse=False,
        supports_gpu=True,
        backend=Backend.TORCH,
        maturity=Maturity.RESEARCH,
        implementation_status=Impl.NATIVE,
        source_status=Src.NOT_FOUND,
        upstream_url=None,
        license=None,
        training_cost=Cost.HIGH,  # torch, ~600 epochs (two-stage)
    ),
    # ── 14. Weighted Contrastive PU ────────────────────────────────
    AlgorithmMetadata(
        name="weighted_contrastive_pu",
        aliases=["wcon_pu", "wconpu", "contrastive_pu"],
        family=Fam.DEEP_PU,
        paper=(
            "Weighted Contrastive Learning with Hard Negative Mining "
            "for Positive and Unlabeled Learning"
        ),
        scenario=[Scn.CASE_CONTROL],
        assumption=[Asm.SCAR],
        requires_class_prior=True,
        supports_sparse=False,
        supports_gpu=True,
        backend=Backend.TORCH,
        maturity=Maturity.RESEARCH,
        implementation_status=Impl.NATIVE,
        source_status=Src.NOT_FOUND,
        upstream_url=None,
        license=None,
        training_cost=Cost.HIGH,  # torch, 800 epochs
    ),
    # ── 15. DGPU ───────────────────────────────────────────────────
    AlgorithmMetadata(
        name="dgpu",
        aliases=["discriminative_generative_pu"],
        family=Fam.DEEP_PU,
        paper="Discriminative-Generative Positive and Unlabeled Learning",
        scenario=[Scn.CASE_CONTROL, Scn.SELECTION_BIASED],
        assumption=[Asm.SCAR, Asm.SAR],
        requires_class_prior=True,
        supports_sparse=False,
        supports_gpu=True,
        backend=Backend.TORCH,
        maturity=Maturity.EXPERIMENTAL,
        implementation_status=Impl.NATIVE,
        source_status=Src.NOT_FOUND,
        upstream_url=None,
        license=None,
        training_cost=Cost.MEDIUM,  # torch, 200 epochs + generative sampling
    ),
]


def register_all_builtin_methods() -> int:
    """Register all 16 paper methods and bind native implementations.

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
        ("pusb", "..estimators.bias_aware.pusb", "PUSBClassifier"),
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
    ]

    for canonical_name, module_path, class_name in _native_imports:
        if canonical_name in _CLASSES:
            continue  # Already bound
        import importlib

        mod = importlib.import_module(module_path, __package__)
        cls = getattr(mod, class_name)
        bind_estimator_class(canonical_name, cls)
