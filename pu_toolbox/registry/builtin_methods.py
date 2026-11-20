"""Built-in algorithm metadata paired with concrete estimator import paths."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from importlib import import_module
from typing import TYPE_CHECKING

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
    Scenario as Scn,
)
from ..core.tags import (
    SourceStatus as Src,
)
from ..core.tags import (
    TrainingCost as Cost,
)
from .metadata import AlgorithmMetadata

if TYPE_CHECKING:
    from .registry import _EstimatorClass

# Estimator attributes own the capabilities they declare; literals supply the rest.
_BUILTIN: dict[tuple[str, str], AlgorithmMetadata] = {
    # ── 1. Class-Prior Estimation ──────────────────────────────────
    ("..prior.pen_l1", "ClassPriorEstimator"): AlgorithmMetadata(
        # km1/km2 name KernelMeanPriorEstimator variants, not penL1 aliases.
        name="class_prior_estimation",
        aliases=["cpe", "pen_l1", "pe"],
        deprecated_aliases=["pe"],
        family=Fam.CLASS_PRIOR_ESTIMATION,
        paper="Class-Prior Estimation for Learning from Positive and Unlabeled Data",
        scenario=[Scn.SINGLE_TRAINING_SET, Scn.CASE_CONTROL],
        assumption=[Asm.SCAR],
        supports_sparse=False,
        supports_gpu=False,
        backend=Backend.NUMPY,
        maturity=Maturity.STABLE,
        source_status=Src.OFFICIAL_RELATED,
        upstream_url="http://www.mcduplessis.com/index.php/software/",
        license="unknown",
        training_cost=Cost.LOW,  # convex scipy solve over a coarse grid
    ),
    # ── 2. ReCPE ───────────────────────────────────────────────────
    ("..prior.recpe", "ReCPEEstimator"): AlgorithmMetadata(
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
    ("..estimators.classic.elkan_noto", "ElkanNotoClassifier"): AlgorithmMetadata(
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
    ("..estimators.risk.upu", "UPUClassifier"): AlgorithmMetadata(
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
    ("..estimators.risk.nnpu", "NonNegativePUClassifier"): AlgorithmMetadata(
        name="nnpu",
        aliases=["non_negative_pu", "nn-pu"],
        paper="Positive-Unlabeled Learning with Non-Negative Risk Estimator",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/kiryor/nnPUlearning",
        license="MIT",
        training_cost=Cost.MEDIUM,  # torch, fixed 200 epochs
    ),
    # ── 6. PNU ─────────────────────────────────────────────────────
    ("..estimators.risk.pnu", "PNUClassifier"): AlgorithmMetadata(
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
    ("..estimators.risk.ldce", "LDCEClassifier"): AlgorithmMetadata(
        name="centroid_pu",
        aliases=["ldce", "centroid_estimation"],
        paper="Loss Decomposition and Centroid Estimation for Positive and Unlabeled Learning",
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://gcatnjust.github.io/ChenGong/code/CEGE_PAMI20.rar",
        license="unknown",
        training_cost=Cost.MEDIUM,  # fixed 10000-iteration alternating scheme
    ),
    ("..estimators.risk.kldce", "KLDCEClassifier"): AlgorithmMetadata(
        name="kldce",
        aliases=["kernelized_ldce"],
        paper=(
            "Loss Decomposition and Centroid Estimation for Positive and "
            "Unlabeled Learning (kernelized version, RBF)"
        ),
        supports_sparse=False,
        supports_gpu=False,
        upstream_url=None,
        license="unknown",
        training_cost=Cost.MEDIUM,  # 100 iterations, QP oracle on RBF
    ),
    # ── 8. LLSVM ───────────────────────────────────────────────────
    ("..estimators.classic.llsvm", "LLSVMClassifier"): AlgorithmMetadata(
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
    ("..estimators.risk.dist_pu", "DistPUClassifier"): AlgorithmMetadata(
        name="dist_pu",
        aliases=["distribution_pu", "distpu"],
        paper="Dist-PU: Positive-Unlabeled Learning from a Label Distribution Perspective",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective",
        license="MIT",
        training_cost=Cost.MEDIUM,  # torch; optional two-stage 120-epoch mini-batch adapter
    ),
    # ── 9a. Variational PU (experimental tabular adapter) ───────────
    ("..estimators.risk.vpu", "VPUClassifier"): AlgorithmMetadata(
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
    ("..estimators.risk.pulda", "PULDAClassifier"): AlgorithmMetadata(
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
    ("..estimators.bias_aware.pusb", "PUSBClassifier"): AlgorithmMetadata(
        name="pusb",
        aliases=["biased_pu", "selection_bias_pu", "nnPUSB"],
        paper="Learning from Positive and Unlabeled Data with a Selection Bias",
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://github.com/MasaKat0/PUlearning",
        license="MIT",
        training_cost=Cost.MEDIUM,  # sklearn LR, max_iter=1000
    ),
    # ── 10a. PUSB kernel (official-aligned RBF adapter) ─────────────
    ("..estimators.bias_aware.pusb_kernel", "PUSBKernelClassifier"): AlgorithmMetadata(
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
    ("..estimators.bias_aware.lbe", "LBEClassifier"): AlgorithmMetadata(
        name="lbe",
        aliases=["labeling_bias", "labeling_bias_estimation"],
        paper="Instance-Dependent Positive and Unlabeled Learning with Labeling Bias Estimation",
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://gcatnjust.github.io/ChenGong/code/LBE_TPAMI21.rar",
        license="needs_review",
        training_cost=Cost.MEDIUM,  # sklearn LR, max_iter=1000 + EM loop
    ),
    ("..estimators.deep.lbe_neural", "NeuralLBEClassifier"): AlgorithmMetadata(
        name="lbe_neural",
        aliases=["neural_lbe"],
        paper="Instance-Dependent Positive and Unlabeled Learning with Labeling Bias Estimation",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://gcatnjust.github.io/ChenGong/code/LBE_TPAMI21.rar",
        license="needs_review",
        training_cost=Cost.MEDIUM,
    ),
    # ── 12. Self-PU ────────────────────────────────────────────────
    ("..estimators.deep.self_pu", "SelfPUClassifier"): AlgorithmMetadata(
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
    ("..estimators.deep.infomax_pu", "InfoMaxPUClassifier"): AlgorithmMetadata(
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
    (
        "..estimators.deep.weighted_contrastive_pu",
        "WeightedContrastivePUClassifier",
    ): AlgorithmMetadata(
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
    ("..estimators.deep.dgpu", "DGPUClassifier"): AlgorithmMetadata(
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
    ("..estimators.deep.grad_pu", "GradPUClassifier"): AlgorithmMetadata(
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
    ("..estimators.risk.puet", "PUExtraTreesClassifier"): AlgorithmMetadata(
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
    ("..estimators.deep.robust_pu", "RobustPUClassifier"): AlgorithmMetadata(
        name="robust_pu",
        aliases=["robust-pu"],
        paper="Robust Positive-Unlabeled Learning via Noise Negative Sample Self-correction",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/woriazzc/Robust-PU",
        license="unknown",
        training_cost=Cost.HIGH,
    ),
    ("..estimators.risk.cvir", "CVIRClassifier"): AlgorithmMetadata(
        name="cvir",
        aliases=["conditional_value_ignoring_risk"],
        paper="Mixture Proportion Estimation and PU Learning: A Modern Approach",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/acmi-lab/PU_learning",
        license="Apache-2.0",
        training_cost=Cost.MEDIUM,
    ),
    ("..estimators.deep.split_pu", "SplitPUClassifier"): AlgorithmMetadata(
        name="split_pu",
        aliases=["split-pu"],
        paper="Split-PU: Hardness-aware Training Strategy for Positive-Unlabeled Learning",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/loadder/SplitPU_MM2022",
        license="unknown",
        training_cost=Cost.HIGH,
    ),
    ("..estimators.deep.tedn", "TEDnClassifier"): AlgorithmMetadata(
        name="tedn",
        aliases=["transform_estimate_discard", "transform-estimate-discard"],
        paper="Mixture Proportion Estimation and PU Learning: A Modern Approach (TEDn)",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/acmi-lab/PU_learning",
        license="Apache-2.0",
        training_cost=Cost.HIGH,
    ),
    ("..estimators.deep.lagam", "LaGAMClassifier"): AlgorithmMetadata(
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
    ("..estimators.deep.gen_pu", "GenPUClassifier"): AlgorithmMetadata(
        name="genpu",
        aliases=["gen_pu", "generative_adversarial_pu"],
        paper="Generative Adversarial Positive-Unlabelled Learning",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://qibinzhao.github.io/assets/publications/IJCAI_2018_HouMing/gen_pu_demo.zip",
        license="unknown",
        training_cost=Cost.HIGH,
    ),
    ("..estimators.deep.holistic_pu", "HolisticPUClassifier"): AlgorithmMetadata(
        name="holistic_pu",
        aliases=["holisticpu", "holistic"],
        paper=(
            "Beyond Myopia: Learning from Positive and Unlabeled Data through "
            "Holistic Predictive Trends"
        ),
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/wxr99/HolisticPU",
        license="unknown",
        training_cost=Cost.HIGH,
    ),
    ("..estimators.deep.pan", "PANClassifier"): AlgorithmMetadata(
        name="pan",
        aliases=["predictive_adversarial_pu"],
        paper="Predictive Adversarial Learning from Positive and Unlabeled Data",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url="https://github.com/morning-dews/PAN",
        license="needs_review",
        training_cost=Cost.HIGH,
    ),
    ("..estimators.deep.p3mix", "P3MixClassifier"): AlgorithmMetadata(
        name="p3mix",
        aliases=["p3_mix", "positive_unlabeled_partially_positive_mixup"],
        paper="Who Is Your Right Mixup Partner in Positive and Unlabeled Learning?",
        supports_sparse=False,
        supports_gpu=True,
        upstream_url=None,
        license="unknown",
        training_cost=Cost.HIGH,
    ),
    ("..estimators.classic.rank_pruning", "RankPruningClassifier"): AlgorithmMetadata(
        name="rp",
        aliases=["rank_pruning", "rankpruning"],
        paper=(
            "Learning with Confident Examples: Rank Pruning for Robust Classification "
            "with Noisy Labels"
        ),
        supports_sparse=False,
        supports_gpu=False,
        upstream_url="https://github.com/cgnorthcutt/rankpruning",
        license="MIT",
        training_cost=Cost.MEDIUM,
    ),
    ("..estimators.deep.pulns", "PULNSClassifier"): AlgorithmMetadata(
        name="pulns",
        aliases=["negative_selector_pu"],
        paper="PULNS: Positive-Unlabeled Learning with Effective Negative Sample Selector",
        requires_clean_support=True,
        supports_sparse=False,
        supports_gpu=True,
        upstream_url=None,
        license="needs_review",
        training_cost=Cost.HIGH,
    ),
}


def _load_builtin_methods() -> Iterator[tuple[AlgorithmMetadata, _EstimatorClass]]:
    for (module_path, class_name), metadata in _BUILTIN.items():
        module = import_module(module_path, __package__)
        yield deepcopy(metadata), getattr(module, class_name)
