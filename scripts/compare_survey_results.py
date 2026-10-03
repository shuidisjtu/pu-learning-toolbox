#!/usr/bin/env python
r"""Attach the pre-registered comparison matrix to a finished numeric summary.

Reads the ``summary.json`` :mod:`summarize_survey_results` writes and resolves every
row through the frozen matrix, writing ``comparison_summary.json`` /
``comparison_summary.md`` and ``unresolved_items.csv``.

This entry point owns the walk, the tally and the rendering; the rules are the
library's.  ``survey_comparison.comparison_context`` states a unit's version, digest,
mapping and class, and ``build_comparison_report`` adjudicates a numeric one -- both
are called rather than reimplemented, because a second copy of the eligibility gate
is a second answer to the same question.

Four things this must not do, each pinned by tests:

* **Not invent a value for a unit the matrix does not cover.**  A row whose identity
  resolves to no mapping is an unresolved item, not a comparison whose anchor list
  came back empty.
* **Not present a number our own protocol will not stand behind.**  A row the matrix
  calls numeric is adjudicated only when our protocol calls it *formal*: a partial
  row has a mean, and what it lost is exactly the standing a numeric comparison
  needs.  The class is still recorded, so nothing is hidden by the withholding.
* **Not restate the pre-registration.**  The six eligibility classes, the anchors and
  the arithmetic come from the matrix module; nothing here re-derives them, because a
  copied threshold is a second threshold.
* **Not fold an unresolved class into a numeric column.**  ``blocked_pending_pa_criterion``,
  ``no_direct_anchor``, ``background_only``, ``magnitude_and_trend`` and
  ``non_runnable`` keep their own names in their own section.

Run:  uv run python scripts/compare_survey_results.py --summary <summary.json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from pu_toolbox.experiment.survey_comparison import (  # noqa: E402
    ELIGIBILITY_CLASSES,
    build_comparison_report,
    comparison_context,
    comparison_digest,
    load_comparison_protocol,
)
from pu_toolbox.experiment.survey_provenance import (  # noqa: E402
    build_provenance,
    render_provenance_lines,
)
from pu_toolbox.experiment.survey_summary import SummaryError, to_result_summary  # noqa: E402

SCHEMA_VERSION = "survey-comparison-report-1"

#: The one class a numeric rule exists for.  Named so the gate below reads as a
#: statement about the pre-registration rather than a magic string.
NUMERIC = "numeric"

#: Recorded for a row the matrix does not cover at all.  Deliberately not a member of
#: :data:`ELIGIBILITY_CLASSES`: it is a defect in this result set or in the matrix's
#: coverage, not a seventh eligibility the pre-registration granted.
UNRESOLVED = "unresolved"

#: The report statuses that are a numeric verdict.  ``not_comparable`` is a report the
#: matrix produced for a class with no rule behind it, so counting it as an
#: adjudication would make every blocked unit look decided -- the exact reading the
#: whole attachment exists to prevent.  Named once because the tally and the Markdown
#: heading are two renderings of one fact: deriving them separately is how the heading
#: came to say 170 while the JSON said 0.
ADJUDICATED_STATUSES: tuple[str, ...] = ("consistent", "investigate")


def is_adjudicated(entry: dict[str, Any]) -> bool:
    """Whether a row carries a numeric verdict rather than a statement of its class."""
    return entry["report"] is not None and entry["report"]["status"] in ADJUDICATED_STATUSES


def compare_row(row: dict[str, Any], *, comparison: dict[str, Any]) -> dict[str, Any]:
    """One report row's standing against the frozen matrix.

    ``context`` is always present once the row resolves -- it is the library's
    identity-level statement and costs nothing.  ``report`` is present only when the
    row is both numeric and formal; otherwise ``note`` says which of the two gates
    stopped it, so a reader can tell a coverage gap from a withholding.
    """
    identity = row["result_identity"]
    entry: dict[str, Any] = {
        "result_identity": identity,
        "row_key": row["row_key"],
        "row_status": row["status"],
        "eligibility": UNRESOLVED,
        "mapping_id": None,
        "context": None,
        "report": None,
        "note": None,
    }
    try:
        context = comparison_context(identity, comparison=comparison)
    except ValueError as exc:
        entry["note"] = str(exc)
        return entry
    entry["context"] = context
    entry["mapping_id"] = context["mapping_id"]
    entry["eligibility"] = context["eligibility"]
    if row["status"] != "formal":
        entry["note"] = (
            f"row status is {row['status']!r}: the mean exists, the standing a numeric "
            "comparison needs does not"
        )
        return entry
    try:
        entry["report"] = build_comparison_report(to_result_summary(row), comparison=comparison)
    except (SummaryError, ValueError) as exc:
        # A row the matrix calls numeric that cannot produce a summary -- fewer than
        # two repeats, a missing spread, an uncertain anchor -- is a finding, not a
        # comparison that silently carries no numbers.
        entry["note"] = f"not evaluated: {exc}"
    return entry


def matrix_state(comparison: dict[str, Any]) -> dict[str, Any]:
    """The matrix's own unresolved state, so it travels with the report."""
    anchors = comparison["anchors"]
    mappings = comparison["mappings"]
    return {
        "comparison_version": comparison["comparison_version"],
        "comparison_sha256": comparison_digest(comparison),
        "review_status": comparison["review_status"],
        "formal_blockers": list(comparison["formal_blockers"]),
        "anchors": len(anchors),
        "anchors_pending_review": sum(
            1 for anchor in anchors if anchor["review_state"] != "accepted"
        ),
        "mappings": len(mappings),
        "mappings_pending_review": sum(
            1 for mapping in mappings if mapping["review_state"] != "accepted"
        ),
        "decision_rules": dict(comparison["decision_rules"]),
    }


def _carried_provenance(summary: dict[str, Any], *, comparison: dict[str, Any]) -> dict[str, Any]:
    """The summary's own identity carried forward, plus this matrix's digest.

    This entry point reads no whitelist -- its input is the summary -- so its
    input result roots are the ones that summary recorded, and its code commit
    is the commit that produced it.  Re-deriving either here would let this
    report disagree with the report it attached itself to.  A summary written
    before this block existed carries none of it, and this report then says so
    rather than presenting an absence as a reading.
    """
    origin = summary.get("provenance") or {}
    return build_provenance(
        protocol_version=summary.get("protocol_version"),
        protocol_sha256=summary.get("protocol_sha256"),
        comparison_version=comparison.get("comparison_version"),
        comparison_sha256=comparison_digest(comparison),
        result_roots=origin.get("input_result_roots"),
        source_roots=origin.get("source_roots"),
        code_commit=origin.get("code_commit"),
        code_commit_dirty=origin.get("code_commit_dirty"),
    )


def build_comparison(summary: dict[str, Any], *, comparison: dict[str, Any]) -> dict[str, Any]:
    """Resolve every row, then report what could and could not be compared."""
    rows = [compare_row(row, comparison=comparison) for row in summary.get("rows", [])]
    by_class = Counter(entry["eligibility"] for entry in rows)
    by_status = Counter(entry["report"]["status"] for entry in rows if entry["report"] is not None)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": None,
        "comparison_digest": comparison_digest(comparison),
        "eligibility_classes": list(ELIGIBILITY_CLASSES),
        "summary_schema_version": summary.get("schema_version"),
        "summary_protocol_sha256": summary.get("protocol_sha256"),
        "provenance": _carried_provenance(summary, comparison=comparison),
        "matrix_state": matrix_state(comparison),
        "coverage": {
            "rows": len(rows),
            # A verdict is a numeric one; ``not_comparable`` is a report the matrix
            # produced, and counting it as adjudicated would make every blocked unit
            # look decided.
            "adjudicated": sum(1 for entry in rows if is_adjudicated(entry)),
            "not_comparable": by_status.get("not_comparable", 0),
            "withheld_not_formal": sum(
                1 for entry in rows if entry["note"] and "row status is" in entry["note"]
            ),
            "by_eligibility": {name: by_class.get(name, 0) for name in ELIGIBILITY_CLASSES}
            | {UNRESOLVED: by_class.get(UNRESOLVED, 0)},
            "by_verdict": dict(sorted(by_status.items())),
        },
        "rows": rows,
    }


def unresolved_items(report: dict[str, Any]) -> list[dict[str, str]]:
    """Every row a reader must not read as an adjudicated numeric comparison."""
    items = [
        {
            "dataset": entry["result_identity"]["dataset"],
            "training_path": entry["result_identity"]["training_path"],
            "method": entry["result_identity"]["method"],
            "labeling_mechanism": entry["result_identity"]["labeling_mechanism"],
            "c_token": entry["result_identity"]["c_token"],
            "selection_protocol": entry["result_identity"]["selection_protocol"],
            "row_status": entry["row_status"],
            "eligibility": entry["eligibility"],
            "reason": entry["note"]
            or (entry["context"] or {}).get("rationale", "")
            or "no numeric rule applies",
        }
        for entry in report["rows"]
        if entry["report"] is None or entry["report"]["status"] == "not_comparable"
    ]
    pending = report["matrix_state"]
    if pending["anchors_pending_review"] or pending["mappings_pending_review"]:
        items.append(
            {
                "dataset": "-",
                "training_path": "-",
                "method": "-",
                "labeling_mechanism": "-",
                "c_token": "-",
                "selection_protocol": "-",
                "row_status": pending["review_status"],
                "eligibility": "matrix_review_pending",
                "reason": (
                    f"{pending['anchors_pending_review']} of {pending['anchors']} anchors and "
                    f"{pending['mappings_pending_review']} of {pending['mappings']} mappings await "
                    f"review; formal_blockers {pending['formal_blockers'] or 'none'}"
                ),
            }
        )
    return items


def _render_markdown(report: dict[str, Any]) -> str:
    state = report["matrix_state"]
    coverage = report["coverage"]
    lines = [
        "# P2.2 文献对照附着",
        "",
        f"- 对照矩阵：`{state['comparison_version']}` / `{state['comparison_sha256']}`",
        f"- 矩阵复核状态：`{state['review_status']}`；正式阻断 {state['formal_blockers'] or '无'}",
        f"- 锚点 {state['anchors']}（待复核 {state['anchors_pending_review']}）；"
        f"映射 {state['mappings']}（待复核 {state['mappings_pending_review']}）",
        f"- 数值汇总：`{report['summary_schema_version']}` / `{report['summary_protocol_sha256']}`",
    ]
    # Identity before verdicts, and carried forward rather than re-read: this
    # report's inputs are whatever that summary was built from.
    lines.extend(render_provenance_lines(report.get("provenance")))
    lines.extend(
        [
            f"- 行数 {coverage['rows']}；已裁决 {coverage['adjudicated']}；"
            f"因非 formal 而保留 {coverage['withheld_not_formal']}",
            "",
            "> 规则与算术读自 `survey_comparison.py`，本文件只汇总与渲染。",
            "> **未裁决不等于一致**：没有数值规则、或本协议未判 formal 的单元，"
            "一律保留其原始类别。",
            "",
            "## 按 eligibility 分布",
            "",
            "| 类别 | 行数 |",
            "|---|---|",
        ]
    )
    lines.extend(
        f"| {name} | {count} |" for name, count in sorted(coverage["by_eligibility"].items())
    )
    adjudicated = [entry for entry in report["rows"] if is_adjudicated(entry)]
    lines.extend(["", f"## 数值裁决（{len(adjudicated)} 行）", ""])
    if not adjudicated:
        # Why a heading of zero is informative, not empty: the two numbers that
        # account for it are printed rather than left for the reader to reconstruct.
        lines.extend(
            [
                "（无。没有任何单元同时满足矩阵判 `numeric` 与本协议判 `formal`；"
                f"其中 `numeric` 类别 "
                f"{coverage['by_eligibility'].get(NUMERIC, 0)} 行，"
                f"`not_comparable` {coverage['not_comparable']} 行。）",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "> 单位：percentage points；`锚` 为文献锚点 id，`外部` 为其发表均值；"
                "阈值读自矩阵的 `decision_rules`。",
                "",
                "| 数据 | 方法 | 机制 | c | 协议 | 映射 | 锚 | 结论 | 我方 | 外部 | 差 | 阈值 |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for entry in adjudicated:
            identity = entry["result_identity"]
            verdict = entry["report"]
            for anchor in verdict.get("anchor_comparisons", []):
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            identity["dataset"],
                            identity["method"],
                            identity["labeling_mechanism"],
                            identity["c_token"],
                            identity["selection_protocol"],
                            verdict["mapping_id"],
                            anchor["anchor_id"],
                            anchor["status"],
                            f"{verdict['mean_ours_pp']:.2f}",
                            str(verdict["n_ours"]),
                            f"{anchor['mean_anchor_pp']:.2f}",
                            f"{anchor['delta_pp']:.2f}",
                            f"{anchor['limit_pp']:.2f}",
                        )
                    )
                    + " |"
                )
    lines.extend(["", "## 未决项", ""])
    items = unresolved_items(report)
    for item in items[:40]:
        lines.append(
            f"- `{item['eligibility']}` — {item['dataset']}/{item['training_path']}/"
            f"{item['method']}/{item['labeling_mechanism']}/c={item['c_token']}/"
            f"{item['selection_protocol']}：{item['reason']}"
        )
    if len(items) > 40:
        lines.append(f"- ……还有 {len(items) - 40} 条，见 `unresolved_items.csv`")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write comparison_summary.json / .md and unresolved_items.csv."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    json_path = target / "comparison_summary.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    written.append(json_path)

    md_path = target / "comparison_summary.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    written.append(md_path)

    items = unresolved_items(report)
    csv_path = target / "unresolved_items.csv"
    columns = [
        "dataset",
        "training_path",
        "method",
        "labeling_mechanism",
        "c_token",
        "selection_protocol",
        "row_status",
        "eligibility",
        "reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(items)
    written.append(csv_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", required=True, help="summary.json to attach the matrix to")
    parser.add_argument("--out-dir", required=True, help="where 03_comparison artifacts go")
    args = parser.parse_args(argv)
    try:
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        report = build_comparison(summary, comparison=load_comparison_protocol())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    for path in write_report(report, args.out_dir):
        print(f"wrote {path}")
    coverage = report["coverage"]
    print(
        f"rows: {coverage['rows']}; adjudicated: {coverage['adjudicated']}; "
        f"unresolved: {len(unresolved_items(report))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
