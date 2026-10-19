"""Source/behavior review registration contract tests.
Each audited method must link its ledger entry to the review document that
records the source-behavior findings, and that document must keep the exact
facts the ledger claims.  Splitting these from the core ledger/registry
consistency checks keeps each file under the test-quality size limit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pu_toolbox.registry import get_metadata

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.contract
def test_ts_calibration_methods_link_to_their_scope_reviews(
    ledger_methods: dict[str, dict],
) -> None:
    """Audited TS hooks stay linked to the applicable non-signing review."""
    review_refs = {
        **dict.fromkeys(
            {"nnpu", "upu", "pusb_kernel", "dist_pu", "self_pu"},
            "docs/research/pu_survey/reviews/rv6_ts_calibration_technical_review_20261019.md",
        ),
        **dict.fromkeys(
            {"pulda", "puet", "gradpu"},
            "docs/research/pu_survey/reviews/additional_ts_calibration_review_20261019.md",
        ),
        **dict.fromkeys(
            {"robust_pu", "split_pu", "genpu"},
            "docs/research/pu_survey/reviews/additional_ts_calibration_review_20261019b.md",
        ),
        "vpu": "docs/research/pu_survey/admission/vpu_sampling_audit.md",
    }
    display_names = {
        "pulda": "PULDA",
        "puet": "PUET",
        "gradpu": "GradPU",
        "robust_pu": "Robust-PU",
        "split_pu": "Split-PU",
        "genpu": "GEN-PU",
    }
    for name, expected in review_refs.items():
        review_path = _ROOT / expected
        assert review_path.is_file()
        review = review_path.read_text(encoding="utf-8")
        assert ledger_methods[name]["calibration_review_ref"] == expected
        assert f"`{name}`" in review or display_names.get(name, "") in review
        non_signing = "签署" in review and any(
            phrase in review for phrase in ("不代表", "不代替", "不是")
        )
        single_party_audit = "单方技术审计" in review and "合作者复核未获得" in review
        assert non_signing or single_party_audit


@pytest.mark.contract
def test_self_pu_population_scope_source_review_is_registered(
    ledger_methods: dict[str, dict],
) -> None:
    """The reviewed paper-aligned scope is documented without changing pilot default."""
    method = ledger_methods["self_pu"]
    review_path = (
        _ROOT / "docs/research/pu_survey/admission/selfpu_consistency_scope_review_20261019.md"
    )
    assert "distillation_scope=paper_population" in method["implementation_status"]
    assert method["calibration_applied"] is True
    assert method["calibration_hooked"] is True
    assert (
        method["evidence"]["consistency_scope_review"] == review_path.relative_to(_ROOT).as_posix()
    )
    uncertainty = "\n".join(method["uncertainty"])
    assert "distillation_scope=untrusted_u" in uncertainty
    assert "paper_population" in uncertainty
    review = review_path.read_text(encoding="utf-8")
    assert "式 (13)" in review and "式 (16)" in review
    assert 'distillation_scope="untrusted_u"' in review
    assert 'distillation_scope="paper_population"' in review


@pytest.mark.contract
def test_kldce_source_behavior_review_is_registered(ledger_methods: dict[str, dict]) -> None:
    """KLDCE source audit discloses its generalized pair-SMO adaptation."""
    method = ledger_methods["kldce"]
    expected = "docs/research/pu_survey/admission/kldce_source_behavior_review_20261019.md"
    review_path = _ROOT / expected
    assert method["source_behavior_review_ref"] == expected
    assert method["evidence"]["source_behavior_review"] == expected
    assert review_path.is_file()
    review = review_path.read_text(encoding="utf-8")
    assert "不是附录 Algorithm 1 中 α/γ 两组分别选择" in review
    assert "不可称作逐行或逐组复现" in review
    assert "不改变冻结矩阵、历史结果或准入状态" in review


@pytest.mark.contract
def test_pusb_linear_baseline_source_scope_is_registered(
    ledger_methods: dict[str, dict],
) -> None:
    """The linear P/U baseline cannot inherit the kernel method's exact-source claim."""
    method = ledger_methods["pusb"]
    expected = "docs/research/pu_survey/admission/pusb_linear_baseline_review_20261019.md"
    review_path = _ROOT / expected
    assert get_metadata("pusb").source_status.value == "official_related"
    assert ledger_methods["pusb"]["code_version"]["source_status"].startswith("official_related")
    assert method["source_behavior_review_ref"] == expected
    assert method["evidence"]["source_behavior_review"] == expected
    assert get_metadata("pusb_kernel").source_status.value == "official_related"
    assert review_path.is_file()
    review = review_path.read_text(encoding="utf-8")
    assert "观测来源标签的概率" in review
    assert "冻结实验结果、比较矩阵和准入状态不变" in review


@pytest.mark.contract
def test_pusb_kernel_source_behavior_review_is_registered(
    ledger_methods: dict[str, dict],
) -> None:
    """The kernel adaptation and its paper-reproduction limits are traceable."""
    method = ledger_methods["pusb_kernel"]
    expected = "docs/research/pu_survey/admission/pusb_kernel_source_behavior_review_20261019.md"
    review_path = _ROOT / expected
    assert method["source_behavior_review_ref"] == expected
    assert method["evidence"]["source_behavior_review"] == expected
    assert review_path.is_file()
    review = review_path.read_text(encoding="utf-8")
    assert "跨目标选参" in review
    assert "paper_claim=false" in review
    assert "不改算法、冻结矩阵、实验产物或准入状态" in review


@pytest.mark.contract
@pytest.mark.parametrize(
    ("method_name", "lock_key", "expected_commit", "review_ref"),
    [
        (
            "nnpu",
            "nnpu",
            "a92c5365c07490c871e338c07f6e31d3eae2ca37",
            "docs/research/pu_survey/admission/nnpu_source_behavior_review_20261019.md",
        ),
        (
            "dist_pu",
            "dist_pu",
            "cb74be1a87176fd38270873c06374e53905b7354",
            "docs/research/pu_survey/admission/dist_pu_source_behavior_review_20261019.md",
        ),
        (
            "pusb_kernel",
            "pusb",
            "3401b77ccdd653d39f4f3a6258a42c7938fa9ede",
            "docs/research/pu_survey/admission/pusb_kernel_source_behavior_review_20261019.md",
        ),
        (
            "self_pu",
            "self_pu",
            "a0e332ae4f8110e2490d597876e36bf837e1060f",
            "docs/research/pu_survey/admission/selfpu_source_behavior_review_20261019.md",
        ),
    ],
)
def test_method_upstream_pin_matches_locked_source_and_behavior_review(
    ledger_methods: dict[str, dict],
    method_name: str,
    lock_key: str,
    expected_commit: str,
    review_ref: str,
) -> None:
    """Structured method pins stay aligned with the frozen source lock and audit."""
    lock_path = _ROOT / "benchmarks/assigned_methods/configs/official_sources.lock.json"
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))["sources"][lock_key]
    method = ledger_methods[method_name]
    assert source_lock["status"] == "locked"
    assert source_lock["commit"] == expected_commit
    assert method["code_version"]["upstream_commit"] == expected_commit
    assert method["source_behavior_review_ref"] == review_ref
    assert expected_commit in (_ROOT / review_ref).read_text(encoding="utf-8")


@pytest.mark.contract
def test_linear_pusb_does_not_inherit_kernel_source_commit(
    ledger_methods: dict[str, dict],
) -> None:
    """The related upstream repository pin belongs to the kernel method, not baseline."""
    assert "upstream_commit" not in ledger_methods["pusb"]["code_version"]


@pytest.mark.contract
def test_vpu_upstream_pin_matches_schedule_and_behavior_audits(
    ledger_methods: dict[str, dict],
) -> None:
    """VPU's structured source pin stays bound to its immutable audit snapshot."""
    method = ledger_methods["vpu"]
    schedule_path = _ROOT / "docs/research/pu_survey/data/vpu_schedule_source_20261015.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    expected_commit = "603bcc1e628795f57a5ac87e5b0b977273b7cf91"
    assert schedule["source"]["commit"] == expected_commit
    assert method["code_version"]["upstream_commit"] == expected_commit
    assert expected_commit in (_ROOT / "docs/research/method_cards/VPU.md").read_text(
        encoding="utf-8"
    )
    assert expected_commit in (
        _ROOT / "docs/research/pu_survey/admission/vpu_d21_d22_review_20261018.md"
    ).read_text(encoding="utf-8")


@pytest.mark.contract
def test_upu_source_behavior_review_binds_locked_bundle_without_overclaim(
    ledger_methods: dict[str, dict],
) -> None:
    """The uPU bundle snapshot is pinned without implying every loss is upstream."""
    method = ledger_methods["upu"]
    expected = "docs/research/pu_survey/admission/upu_source_behavior_review_20261019.md"
    review_path = _ROOT / expected
    assert method["source_behavior_review_ref"] == expected
    assert method["evidence"]["source_behavior_review"] == expected
    assert review_path.is_file()
    review = review_path.read_text(encoding="utf-8")
    commit = "79a5bf104a53cb6d3498918e435bd88ac5331543"
    lock_path = _ROOT / "benchmarks/assigned_methods/configs/official_sources.lock.json"
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))["sources"]["upu"]
    assert source_lock["status"] == "locked"
    assert source_lock["commit"] == commit
    assert method["code_version"]["upstream_commit"] == commit
    assert "PU_SL" in review and "pu_mr.py:12-20,26-47" in review
    assert "C-DH" in review and "C-LL" in review and "squared" in review
    assert "不代表 toolbox 的所有 loss 分支或实验与其等价" in review


@pytest.mark.contract
def test_nnpu_source_behavior_review_tracks_branch_and_batch_differences(
    ledger_methods: dict[str, dict],
) -> None:
    """nnPU branch semantics and its source-version/batching boundaries are explicit."""
    method = ledger_methods["nnpu"]
    expected = "docs/research/pu_survey/admission/nnpu_source_behavior_review_20261019.md"
    review_path = _ROOT / expected
    assert method["source_behavior_review_ref"] == expected
    assert method["evidence"]["source_behavior_review"] == expected
    assert review_path.is_file()
    review = review_path.read_text(encoding="utf-8")
    assert "反传 `-γ r`" in review
    assert "合并的 `XYtrain`" in review
    commit = "a92c5365c07490c871e338c07f6e31d3eae2ca37"
    lock_path = _ROOT / "benchmarks/assigned_methods/configs/official_sources.lock.json"
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))["sources"]["nnpu"]
    assert source_lock["commit"] == commit
    assert method["code_version"]["upstream_commit"] == commit
    assert commit in review
    assert "尚无该仓库 immutable commit" not in review


@pytest.mark.contract
def test_lbe_source_review_ref_binds_official_archive_identity_and_behavior(
    ledger_methods: dict[str, dict],
) -> None:
    """The audited author bundle is traceable without implying clean-room equivalence."""
    method = ledger_methods["lbe"]
    expected = "docs/research/pu_survey/data/lbe_source_review_20261012.json"
    assert method["source_behavior_review_ref"] == expected
    assert method["evidence"]["source_behavior_review"] == expected
    review = json.loads((_ROOT / expected).read_text(encoding="utf-8"))
    source = review["official_upstream_verification_20261019"]
    assert source["matches_locked_official_sha256"] is True
    assert source["download_sha256"] == method["code_version"]["official_archive_sha256"]
    assert source["matches_user_supplied_archive"] is False
    assert review["equation_findings"]
    assert method["implementation_status"].startswith("NATIVE")
    assert "不代表 clean-room 实现等价" in method["code_version"]["source_status"]


@pytest.mark.contract
def test_self_pu_official_source_pin_is_registered_without_reproduction_claim(
    ledger_methods: dict[str, dict],
) -> None:
    """Pin the official source snapshot without implying toolbox reproduction."""
    method = ledger_methods["self_pu"]
    lock_path = _ROOT / "benchmarks/assigned_methods/configs/official_sources.lock.json"
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))["sources"]["self_pu"]
    expected_commit = "a0e332ae4f8110e2490d597876e36bf837e1060f"
    assert source_lock["status"] == "locked"
    assert source_lock["commit"] == expected_commit
    assert method["code_version"]["upstream_commit"] == expected_commit
    assert expected_commit in method["code_version"]["source_status"]
    assert method["code_version"]["source_status"].split("（", 1)[0] == "official_exact"
    assert "观察到的 master commit" in "\n".join(method["uncertainty"])
    review_path = _ROOT / "docs/research/pu_survey/admission/selfpu_source_pin_review_20261019.md"
    assert method["source_version_review_ref"] == review_path.relative_to(_ROOT).as_posix()
    assert method["evidence"]["source_version_review"] == review_path.relative_to(_ROOT).as_posix()
    review = review_path.read_text(encoding="utf-8")
    assert expected_commit in review
    assert "不表示工具箱 clean-room 实现与上游逐行相同" in review


@pytest.mark.contract
def test_self_pu_locked_source_behavior_audit_is_linked_and_keeps_test_isolated(
    ledger_methods: dict[str, dict],
) -> None:
    """Bind the source behavior finding to the exact locked official revision."""
    method = ledger_methods["self_pu"]
    expected = "docs/research/pu_survey/admission/selfpu_source_behavior_review_20261019.md"
    assert method["source_behavior_review_ref"] == expected
    assert method["evidence"]["source_behavior_review"] == expected
    review = (_ROOT / expected).read_text(encoding="utf-8")
    commit = "a0e332ae4f8110e2490d597876e36bf837e1060f"
    assert commit in review
    assert "split='test'" in review
    assert "负预测熵目标" in review
    assert "测试标签参与 checkpoint 选择" in review
    assert "不得为追求" in review and "测试样本重新接入训练" in review


@pytest.mark.contract
def test_existing_candidate_source_audits_are_explicitly_linked_from_ledger(
    ledger_methods: dict[str, dict],
) -> None:
    """Keep the existing source/behavior evidence discoverable per candidate."""
    expected = {
        "pulda": ("docs/research/method_cards/PULDA.md", "entropyMinimization"),
        "puet": ("docs/research/method_cards/PUET.md", "tree.py:293-298"),
        "gradpu": (
            "docs/research/pu_survey/admission/gradpu_source_review_20261005.md",
            "pending_method_owner_review",
        ),
        "split_pu": ("docs/research/method_cards/Split-PU.md", "测试监控不是 test-best 选模"),
        "vpu": (
            "docs/research/pu_survey/admission/vpu_sampling_audit.md",
            "including positive and unlabeled",
        ),
    }
    for name, (relative_path, marker) in expected.items():
        method = ledger_methods[name]
        assert method["source_behavior_review_ref"] == relative_path
        assert method["evidence"]["source_behavior_review"]
        review_path = _ROOT / relative_path
        assert review_path.is_file()
        assert marker in review_path.read_text(encoding="utf-8")
