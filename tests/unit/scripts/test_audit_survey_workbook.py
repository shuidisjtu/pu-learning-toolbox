"""Synthetic Excel checks; the privately supplied workbook is not a test dependency."""

import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import pytest

from pu_toolbox.experiment.pilot_plan import planned_runs
from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.unit

openpyxl = pytest.importorskip("openpyxl")

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "workbook_audit", ROOT / "scripts/audit_survey_workbook.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

RUN_HEADERS = [
    "batch",
    "dataset",
    "method",
    "training_path",
    "run_view",
    "mechanism",
    "c",
    "seed",
    "execution_mode",
    "method_variant",
    "calibration_applied",
    "comparability_group",
    "protocol_deviation",
    "formal_eligible",
    "formal_blockers",
    "pa_accuracy",
    "pa_auc",
    "pa_auc_unavailable_reason",
    "oa_accuracy",
    "oa_auc",
    "oa_auc_unavailable_reason",
    "single_config_seconds",
    "single_config_candidates",
    "tuning_seconds",
    "peak_gpu_memory_mb",
    "python_version",
    "torch_version",
    "cuda_runtime_version",
    "gpu_name",
    "platform",
]
SUMMARY_HEADERS = [
    "status",
    "dataset",
    "training_path",
    "comparability_group",
    "run_view",
    "mechanism",
    "method",
    "selection_protocol",
    "c",
    "n_observed",
    "mean_percent",
    "sample_std_percent",
    "auc_mean_percent",
    "auc_n_observed",
    "missing_seeds",
    "reasons",
    "run_single_config_seconds_sum",
    "run_tuning_seconds_sum",
    "run_peak_gpu_memory_mb",
]
COVERAGE_HEADERS = [
    "batch",
    "dataset",
    "training_path",
    "methods",
    "manifests",
    "view_distribution",
    "calibration_distribution",
    "mechanism_distribution",
    "protocol_deviation",
    "peak_gpu_memory_unmeasured",
    "recorded_failures",
]


@pytest.fixture
def receipt(tmp_path):
    from collections import Counter, defaultdict

    protocol = load_protocol()
    units = {
        (r["dataset"], r["method"], r["training_path"]): r for r in protocol["execution_units"]
    }
    index = json.loads(
        (ROOT / "docs/research/pu_survey/data/p2_2_artifacts_index.json").read_text()
    )
    book = openpyxl.Workbook()
    book.remove(book.active)
    for name, header in (
        ("runs", RUN_HEADERS),
        ("P2.2 汇总", SUMMARY_HEADERS),
        ("coverage", COVERAGE_HEADERS),
    ):
        book.create_sheet(name).append(header)
    book.create_sheet("readme").append(
        [
            index["identity"]["protocol_sha256"]
            + " "
            + index["identity"]["comparison_sha256"]
            + " "
            + index["produced_by_commit"]
        ]
    )
    raw = []
    grouped = defaultdict(list)
    for run in planned_runs(protocol):
        oracle = run.is_oracle
        os = run.method in {"lbe", "pn_oracle"}
        row = dict(
            batch=audit.batch_for(run.dataset, run.method, run.training_path),
            dataset=run.dataset,
            method=run.method,
            training_path=run.training_path,
            run_view="os-compatible" if os else "ts-compatible",
            mechanism="pn_oracle" if oracle else run.mechanism,
            c=run.c_token,
            seed=run.seed,
            execution_mode="versioned_pilot",
            method_variant="without_clean_validation_meta_reweighting"
            if run.method == "self_pu"
            else "shared_spec_engineering",
            calibration_applied=not os,
            comparability_group=units[(run.dataset, run.method, run.training_path)][
                "comparability_group"
            ],
            protocol_deviation="c_grid" if oracle else None,
            formal_eligible=not oracle,
            formal_blockers="protocol_deviation" if oracle else None,
            single_config_seconds=1.0,
            single_config_candidates=1,
            tuning_seconds=1.1,
            peak_gpu_memory_mb=None if run.method in {"lbe", "upu", "pusb_kernel"} else 12.0,
        )
        for selection in ("PA", "OA"):
            if selection == "PA" and (oracle or run.mechanism != "scar"):
                continue
            row[selection.lower() + "_accuracy"] = 0.8 + run.seed * 0.01
            row[selection.lower() + "_auc"] = 0.9 + run.seed * 0.01
            key = tuple(
                selection
                if k == "selection_protocol"
                else audit.token(row.get(k))
                if k == "c"
                else row[k]
                for k in audit.KEY_FIELDS
            )
            grouped[key].append(row)
        raw.append(row)
        book["runs"].append([row.get(k) for k in RUN_HEADERS])
    for key, rows in grouped.items():
        row = dict(zip(audit.KEY_FIELDS, key, strict=True))
        values = [r[row["selection_protocol"].lower() + "_accuracy"] for r in rows]
        row.update(
            status="partial" if row["method"] == "pn_oracle" else "formal",
            reasons="formal_blockers_present,protocol_deviation"
            if row["method"] == "pn_oracle"
            else None,
            n_observed=5,
            mean_percent=statistics.mean(values) * 100,
            sample_std_percent=statistics.stdev(values) * 100,
            auc_mean_percent=92.0,
            auc_n_observed=5,
            run_single_config_seconds_sum=5.0,
            run_tuning_seconds_sum=5.5,
            run_peak_gpu_memory_mb=rows[0]["peak_gpu_memory_mb"],
        )
        book["P2.2 汇总"].append([row.get(k) for k in SUMMARY_HEADERS])
    for batch in index["inputs"]["batches"]:
        rows = [r for r in raw if r["batch"] == batch]
        row = dict(
            batch=batch,
            dataset=rows[0]["dataset"],
            training_path=rows[0]["training_path"],
            manifests=len(rows),
            peak_gpu_memory_unmeasured=sum(r["peak_gpu_memory_mb"] is None for r in rows),
            recorded_failures=0,
        )
        for field, source in (
            ("methods", "method"),
            ("view_distribution", "run_view"),
            ("calibration_distribution", "calibration_applied"),
            ("mechanism_distribution", "mechanism"),
        ):
            row[field] = ", ".join(
                f"{k} = {v}" for k, v in Counter(str(r[source]) for r in rows).items()
            )
        deviations = sum(bool(r["protocol_deviation"]) for r in rows)
        row["protocol_deviation"] = f"c_grid = {deviations}" if deviations else "none"
        book["coverage"].append([row.get(k) for k in COVERAGE_HEADERS])
    path = tmp_path / "synthetic.xlsx"
    index_path = tmp_path / "index.json"

    def save():
        book.save(path)
        index["release_table"].update(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        index_path.write_text(json.dumps(index))

    save()
    yield book, path, index_path, save
    book.close()


def test_basic_determ_valid_workbook_reconstructs_183_rows(receipt):
    _, path, index, _ = receipt
    report = audit.audit_workbook(path, index)
    assert report["ok"], report["problems"]
    assert audit.audit_workbook(path, index) == report
    assert report["summary_states"] == {"formal": 180, "partial": 3}
    assert report["selection_rows"] == {"PA": 54, "OA": 129}
    assert report["run_cost_groups"] == 129
    assert report["gpu_unmeasured"] == 315
    assert report["source_manifests_independently_verified"] is False


@pytest.mark.parametrize(
    "sheet,column,value,expected",
    [
        ("P2.2 汇总", "mean_percent", 0.1, "mean_percent delta"),
        ("P2.2 汇总", "sample_std_percent", 0.1, "sample_std_percent delta"),
        ("P2.2 汇总", "status", "blocked", "summary status"),
        ("runs", "c", "0.99", "run keys differ"),
        ("runs", "calibration_applied", False, "calibration flag"),
        ("coverage", "manifests", 214, "coverage manifests"),
    ],
)
def test_param_tampering_is_not_accepted(receipt, sheet, column, value, expected):
    book, path, index, save = receipt
    headers = next(book[sheet].values)
    book[sheet].cell(2, headers.index(column) + 1, value)
    save()  # A matching file digest must not excuse incorrect content.
    report = audit.audit_workbook(path, index)
    assert not report["ok"]
    assert any(expected in message for message in report["problems"])


def test_edge_four_decimal_training_rounding_is_allowed(receipt):
    book, path, index, save = receipt
    headers = next(book["P2.2 汇总"].values)
    book["P2.2 汇总"].cell(2, headers.index("run_single_config_seconds_sum") + 1, 5.0002)
    save()
    assert audit.audit_workbook(path, index)["ok"]


def test_file_digest_mismatch_is_rejected(receipt):
    _, path, index, _ = receipt
    recorded = json.loads(index.read_text())
    recorded["release_table"]["sha256"] = "0" * 64
    index.write_text(json.dumps(recorded))
    assert not audit.audit_workbook(path, index)["ok"]
