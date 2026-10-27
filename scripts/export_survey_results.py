#!/usr/bin/env python
"""Export the delivery workbook and the per-method diagnostic cards.

Reads the report ``summarize_survey_results.py`` builds -- either a finished
``summary.json`` or, in one step, the batch whitelist config it is built from --
and writes three files into ``--out-dir``:

* ``survey_results.xlsx``: the four protocol §5.1 groups on their own sheets (mean
  ± std over the protocol seeds, additional metrics, cost), plus ``diagnostic``,
  ``coverage`` and ``readme``.  Nothing is ranked.
* ``diagnostic_cards.md`` / ``diagnostic_cards.json``: one card per method, signals
  from the execution plan §1.4.5.

This script re-derives no number: layout lives in ``survey_workbook`` and the
signals in ``survey_diagnostics``, both pure functions of the report.

Run:  uv run python scripts/export_survey_results.py --config <config.json> --out-dir <dir>
      uv run python scripts/export_survey_results.py --summary <summary.json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ``scripts/`` is not a package, so siblings are reached through this file's own
# directory.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import audit_survey_batches as audit  # noqa: E402
import summarize_survey_results as summarizer  # noqa: E402

from pu_toolbox.experiment.method_ledger import (  # noqa: E402
    load_ledger,
    native_sampling_assumption,
)
from pu_toolbox.experiment.survey_audit import AuditError  # noqa: E402
from pu_toolbox.experiment.survey_diagnostics import build_cards, render_cards  # noqa: E402
from pu_toolbox.experiment.survey_protocol import (  # noqa: E402
    load_protocol,
    resolve_protocol_path,
)
from pu_toolbox.experiment.survey_provenance import (  # noqa: E402
    resolve_code_commit,
    with_code_commit,
)
from pu_toolbox.experiment.survey_summary import SummaryError  # noqa: E402
from pu_toolbox.experiment.survey_workbook import WorkbookError, write_workbook  # noqa: E402

WORKBOOK_NAME = "survey_results.xlsx"
CARDS_MD_NAME = "diagnostic_cards.md"
CARDS_JSON_NAME = "diagnostic_cards.json"


def native_views() -> dict[str, str]:
    """Each ledger method's native sampling view (``os`` / ``ts`` / ``both``)."""
    methods = load_ledger()["methods"]
    return {name: native_sampling_assumption(entry) for name, entry in methods.items()}


def export(report: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write the workbook and the cards for *report*; returns the paths written."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    cards = build_cards(report, native_view=native_views())
    written = [write_workbook(report, target / WORKBOOK_NAME)]
    md_path = target / CARDS_MD_NAME
    md_path.write_text(render_cards(cards), encoding="utf-8")
    json_path = target / CARDS_JSON_NAME
    json_path.write_text(
        json.dumps(cards, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    return [*written, md_path, json_path]


def _report_from_config(config_path: str, protocol_name: str, out_dir: Path) -> dict[str, Any]:
    config = audit.load_config(config_path)
    protocol = load_protocol(resolve_protocol_path(protocol_name))
    report = summarizer.build_summary(config, protocol=protocol)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    commit, dirty = resolve_code_commit()
    report["provenance"] = with_code_commit(report.get("provenance"), commit, dirty=dirty)
    for path in summarizer.write_report(report, out_dir):
        print(f"wrote {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="batch-root whitelist JSON; the summary is built first")
    source.add_argument("--summary", help="a summary.json written by summarize_survey_results.py")
    parser.add_argument("--out-dir", required=True, help="where the workbook and cards go")
    parser.add_argument(
        "--protocol",
        default="survey-v3.0",
        help="protocol the tree was run under, used with --config (shipped name or path)",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    try:
        if args.config:
            report = _report_from_config(args.config, args.protocol, out_dir)
        else:
            report = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        for path in export(report, out_dir):
            print(f"wrote {path}")
    except (
        AuditError,
        SummaryError,
        WorkbookError,
        audit.ConfigError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"overall: {report.get('overall')}; rows: {len(report.get('rows', []))}")
    return 0 if report.get("overall") != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
