"""Write result summaries, workbooks, diagnostic cards and conclusions."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .survey_conclusions import build_conclusions, conclusion_rows, render_conclusions
from .survey_diagnostics import build_cards, render_cards
from .survey_workbook import write_workbook


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    return path


def export(
    report: dict[str, Any],
    out_dir: str | Path,
    *,
    literature: Sequence[Mapping[str, Any]] = (),
    native_view: Mapping[str, str] | None = None,
) -> list[Path]:
    """Export a finished report; method and literature facts are optional inputs."""
    cards = build_cards(report, native_view=native_view or report.get("native_views", {}))
    conclusions = build_conclusions(report, cards, literature=literature)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    workbook = write_workbook(
        report, target / "survey_results.xlsx", conclusions=conclusion_rows(conclusions)
    )
    cards_md = target / "diagnostic_cards.md"
    cards_md.write_text(render_cards(cards), encoding="utf-8")
    conclusions_md = target / "conclusions.md"
    conclusions_md.write_text(render_conclusions(conclusions), encoding="utf-8")
    return [
        workbook,
        cards_md,
        _write_json(target / "diagnostic_cards.json", cards),
        conclusions_md,
        _write_json(target / "conclusions.json", conclusions),
    ]


def write_summary(report: Mapping[str, Any], out_dir: str | Path) -> list[Path]:
    """Write five summary artifacts without loading execution or audit configuration."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = _write_json(target / "summary.json", report)
    rows = report.get("rows", [])
    columns = (
        "dataset",
        "training_path",
        "method",
        "run_view",
        "mechanism",
        "selection_protocol",
        "c_token",
        "mean",
        "std",
        "seeds_observed",
        "missing_seeds",
        "status",
        "reasons",
        "cost_id",
    )
    records = []
    lines = [
        "# Experiment results",
        "",
        f"Report mode: {report.get('report_mode', 'provided_summary')}",
        f"Overall: {report.get('overall', 'unavailable')}",
        "Formal admission is not inferred by exporting or collecting results.",
        "",
        "| Dataset | Method | Path | View | Mechanism | Selection | c | Accuracy | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        metric = row["metric"]
        record = {
            **row["result_identity"],
            **row["row_key"],
            **{key: metric.get(key) for key in ("mean", "std", "seeds_observed", "missing_seeds")},
            "status": row["status"],
            "reasons": row.get("reasons", []),
            "cost_id": row.get("cost_id"),
        }
        records.append(record)
        mean, std = metric.get("mean"), metric.get("std")
        value = "unavailable" if mean is None else f"{mean * 100:.2f}"
        if std is not None:
            value += f" ± {std * 100:.2f}"
        cells = [str(record.get(key, "")) for key in columns[:7]]
        lines.append("| " + " | ".join([*cells, value, row["status"]]) + " |")
    lines.extend(["", "Costs are recorded once per fit in summary.json/run_costs.", ""])
    for source in report.get("sources", []):
        if source.get("state") != "available":
            lines.append(f"- {source['state']}: {source['path']}")
    markdown = target / "summary.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = target / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in record.items()
                }
            )
    coverage = target / "coverage.csv"
    with coverage.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kind", "identity", "status", "missing_seeds", "reasons"])
        for row in rows:
            writer.writerow(
                [
                    "row",
                    json.dumps(row["row_key"], sort_keys=True),
                    row["status"],
                    json.dumps(row["metric"].get("missing_seeds", [])),
                    json.dumps(row.get("reasons", [])),
                ]
            )
        for source in report.get("sources", []):
            writer.writerow(
                [
                    "source",
                    source["path"],
                    source["state"],
                    "",
                    json.dumps(source.get("formal_blockers", [])),
                ]
            )
    refusals = _write_json(target / "refusals.json", report.get("refusals", []))
    return [json_path, markdown, csv_path, coverage, refusals]
