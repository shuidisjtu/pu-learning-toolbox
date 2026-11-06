# pu_toolbox/experiment/survey_conclusions.py

"""Conclusion analysis with explicit diagnostic signal attribution.

The delivery asks for conclusions that name the signals they rest on and keep
three kinds apart.  This module builds that artifact from two inputs the exporter
already has -- the summary report and the per-method diagnostic cards -- and adds
nothing else:

* **本实验观测** -- generated from a card's flagged observations; carries the
  numbers and the context each one came from.
* **推断** -- generated beside an observation; cites the observation id and states
  the diagnostic follow-up direction for that signal.
* **文献事实** -- never generated from run numbers.  It appears only when the
  caller supplies a registered fact with a source, so a paper claim cannot be
  manufactured by this module.

Four rules keep the artifact honest:

* A conclusion without a signal is refused: every conclusion names its basis.
* An observation without evidence is refused; an inference that cites no
  observation is refused.
* A literature fact without a source is refused, and it may carry no run
  evidence -- the run numbers belong to the observation kind.
* Signals the cards cannot compute are listed with their reasons instead of
  being dropped.

The builder is a pure function of its inputs: same report, same cards, same
literature registry, same output.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .survey_diagnostics import DIRECTIONS, NOT_COMPUTED
from .survey_diagnostics import SIGNALS as COMPUTED_SIGNALS

#: The three distinct kinds of conclusion evidence.
KINDS: tuple[str, ...] = ("文献事实", "本实验观测", "推断")

#: Every diagnostic signal the conclusion layer can attribute, with its card label.
#: The seven computed ones come from the diagnostics module; the five the cards
#: cannot read keep their ids so a conclusion can never cite a made-up signal.
SIGNALS: dict[str, str] = {
    "mechanism_scar_to_sar": "机制对比 SCAR→SAR",
    "view_os_ts": "视图对比 OS↔TS",
    "backbone_path": "backbone / 表征",
    "class_prior_semantics": "class prior 语义",
    "c_sensitivity": "c 敏感性",
    "pn_oracle_alignment": "PN oracle 对齐",
    "pa_oa_gap": "PA / OA 差距",
    "seed_variance": "seed 间方差",
    "cross_dataset_trend": "跨数据集趋势",
    "external_anchor_deviation": "外部锚点偏差",
    "native_assumption_split": "原生适用 / 假设违背分组",
    "backbone_rank_reversal": "排名颠倒",
}

SCHEMA_VERSION = "survey-conclusions-1"


class ConclusionError(ValueError):
    """A conclusion that may not be written: missing kind, basis or evidence."""


def _label(signal: str) -> str:
    return SIGNALS[signal]


def _where(context: Mapping[str, Any]) -> str:
    return (
        "/".join(
            str(context.get(field, "?"))
            for field in ("dataset", "training_path", "run_view", "selection_protocol")
        )
        + "/c="
        + str(context.get("c_token", "?"))
    )


def _value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, Mapping):
        return "/".join(f"{key}:{_value(inner)}" for key, inner in value.items())
    return str(value)


def _figures(item: Mapping[str, Any]) -> str:
    parts = []
    for key, value in item.items():
        if key in ("context", "flag"):
            continue
        parts.append(f"{key}={_value(value)}")
    return ", ".join(parts)


def _observation(method: str, signal: str, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    flagged = [item for item in items if item["flag"]]
    return {
        "conclusion_id": f"obs:{method}:{signal}",
        "kind": "本实验观测",
        "method": method,
        "scope": {"rows_in_signal": len(items), "flagged_contexts": len(flagged)},
        "signals": [signal],
        "statement": (
            f"{method} 的{_label(signal)}在 {len(items)} 个可比较上下文中"
            f"有 {len(flagged)} 个触发异常形态；标记只是排查提示，不判定实现有错。"
        ),
        "evidence": flagged,
        "flag": True,
    }


def _inference(
    method: str, signal: str, observation_id: str, rows_in_signal: int
) -> dict[str, Any]:
    return {
        "conclusion_id": f"inf:{method}:{signal}",
        "kind": "推断",
        "method": method,
        "scope": {"rows_in_signal": rows_in_signal},
        "signals": [signal],
        "statement": f"{method} 的{_label(signal)}出现异常，待查方向：{DIRECTIONS[signal]}。",
        "evidence": [],
        "cites": [observation_id],
        "follow_up": DIRECTIONS[signal],
        "flag": True,
    }


def _literature_fact(fact: Mapping[str, Any]) -> dict[str, Any]:
    try:
        fact_id, method, statement = fact["fact_id"], fact["method"], fact["statement"]
        signals, source = list(fact["signals"]), fact["source"]
    except (KeyError, TypeError) as exc:
        raise ConclusionError(f"literature fact is missing a required field: {exc}") from exc
    return {
        "conclusion_id": f"lit:{fact_id}",
        "kind": "文献事实",
        "method": method,
        "scope": {"registered": True},
        "signals": signals,
        "statement": statement,
        "source": source,
        "evidence": [],
        "flag": False,
    }


def validate_conclusion(
    record: Mapping[str, Any], *, by_id: Mapping[str, Mapping[str, Any]] | None = None
) -> None:
    """Refuse conclusions that lack a valid kind, signal or evidence."""
    kind = record.get("kind")
    if kind not in KINDS:
        raise ConclusionError(f"kind must be one of {KINDS}; got {kind!r}")
    signals = record.get("signals") or []
    if not signals:
        raise ConclusionError(f"{record.get('conclusion_id')!r} names no signal")
    if len(set(signals)) != len(signals):
        raise ConclusionError(f"{record.get('conclusion_id')!r} repeats a signal: {signals}")
    unknown = sorted(set(signals) - set(SIGNALS))
    if unknown:
        raise ConclusionError(f"{record.get('conclusion_id')!r} cites unknown signals: {unknown}")
    if not record.get("method"):
        raise ConclusionError(f"{record.get('conclusion_id')!r} names no method")

    def refuse(reason: str) -> None:
        raise ConclusionError(f"{record.get('conclusion_id')!r} {reason}")

    if kind == "文献事实":
        if not record.get("source"):
            refuse("is a literature fact without a source")
        if record.get("evidence"):
            refuse("is a literature fact carrying run evidence")
        if record.get("cites"):
            refuse("is a literature fact citing runs")
    elif kind == "本实验观测":
        if not record.get("evidence"):
            refuse("is an observation without evidence")
        if record.get("source"):
            refuse("is an observation with a source")
        if record.get("cites"):
            refuse("is an observation citing a conclusion")
    else:
        cites = record.get("cites") or []
        if not cites:
            refuse("is an inference without an observation to cite")
        if not record.get("follow_up"):
            refuse("is an inference without a direction")
        if by_id is not None:
            for cited in cites:
                target = by_id.get(cited)
                if target is None:
                    refuse(f"cites unknown conclusion {cited!r}")
                if target.get("kind") != "本实验观测":
                    refuse(f"cites {cited!r}, which is not an observation")


def build_conclusions(
    report: Mapping[str, Any],
    cards: Mapping[str, Mapping[str, Any]],
    *,
    literature: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Every conclusion the cards support, plus what could not be attributed."""
    conclusions: list[dict[str, Any]] = [_literature_fact(fact) for fact in literature]
    flagged_methods: set[str] = set()
    quiet_methods: list[str] = []
    for method in sorted(cards):
        card = cards[method]
        fired = False
        for signal in COMPUTED_SIGNALS:
            items = card["signals"][signal]
            if not items:
                continue
            observation = _observation(method, signal, items)
            if not observation["evidence"]:
                continue
            conclusions.append(observation)
            conclusions.append(_inference(method, signal, observation["conclusion_id"], len(items)))
            fired = True
        if fired:
            flagged_methods.add(method)
        else:
            quiet_methods.append(method)

    not_attributable = [
        {"signal": signal, "reason": reason} for signal, reason in NOT_COMPUTED.items()
    ]
    by_id = {record["conclusion_id"]: record for record in conclusions}
    if len(by_id) != len(conclusions):
        raise ConclusionError("duplicate conclusion id")
    for record in conclusions:
        validate_conclusion(record, by_id=by_id)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_from": {
            "protocol_version": report.get("protocol_version"),
            "protocol_sha256": report.get("protocol_sha256"),
            "rows": len(report.get("rows", [])),
        },
        "conclusions": conclusions,
        "not_attributable": not_attributable,
        "flagged_methods": sorted(flagged_methods),
        "quiet_methods": quiet_methods,
        "literature_facts": sum(1 for record in conclusions if record["kind"] == "文献事实"),
    }


def _signals_text(record: Mapping[str, Any]) -> str:
    return ", ".join(_label(signal) for signal in record["signals"])


def render_conclusions(bundle: Mapping[str, Any]) -> str:
    """The conclusions as Markdown: what each one rests on, and what it may not say."""
    lines = [
        "# 结论分析（执行计划 §1.4.5 信号归因）",
        "",
        "本文件由导出器从汇总报告与诊断卡生成，不重算任何数；每条结论都列出依据的信号。",
        "",
        "## 三类结论",
        "",
        "- **文献事实**：来自已登记的论文陈述，必须带来源；本工具不会把运行数字写成文献事实。",
        "- **本实验观测**：来自完整 seed 的报告行，附带上下文与数值；异常标记只是排查提示。",
        "- **推断**：由观测引出的待查方向，必须引用产生它的观测。",
        "",
        "规则：没有标注信号的结论不写入；单 seed 或缺 seed 的行不进任何信号；"
        "标记不判定实现有错，只提示回看实现、选模与数据视图。",
        "",
        "## 概览",
        "",
        f"- 结论总数：{len(bundle['conclusions'])}"
        f"（文献事实 {bundle['literature_facts']}，"
        f"本实验观测 {sum(1 for c in bundle['conclusions'] if c['kind'] == '本实验观测')}，"
        f"推断 {sum(1 for c in bundle['conclusions'] if c['kind'] == '推断')}）。",
        f"- 触发标记的方法：{', '.join(bundle['flagged_methods']) or '无'}。",
        f"- 未触发任何标记的方法：{', '.join(bundle['quiet_methods']) or '无'}。",
        f"- 报告版本：{bundle['generated_from'].get('protocol_version') or '未记录'}；"
        f"报告行数：{bundle['generated_from'].get('rows')}。",
        "",
        "## 结论",
        "",
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in bundle["conclusions"]:
        grouped.setdefault(record["method"], []).append(record)
    if not grouped:
        lines.append("本报告没有任何方法触发可归因的异常，故不产生结论。")
        lines.append("")
    for method in sorted(grouped):
        lines.append(f"### {method}")
        lines.append("")
        for record in grouped[method]:
            kind = record["kind"]
            head = (
                f"- **{kind}**（{record['conclusion_id']}；信号：{_signals_text(record)}）"
                f"：{record['statement']}"
            )
            lines.append(head)
            if kind == "本实验观测":
                for item in record["evidence"]:
                    lines.append(f"  - 证据：{_where(item['context'])}：{_figures(item)}")
            elif kind == "推断":
                lines.append(f"  - 依据：{', '.join(record['cites'])}")
            else:
                lines.append(f"  - 来源：{record['source']}")
        lines.append("")

    lines.append("## 未归因的信号")
    lines.append("")
    lines.append("这些信号在报告或台账里没有可读的数据，本文件不为其生成结论：")
    lines.append("")
    for item in bundle["not_attributable"]:
        lines.append(f"- {_label(item['signal'])}（{item['signal']}）：{item['reason']}")
    lines.append("")
    if not bundle["literature_facts"]:
        lines.append("本次未提供文献登记，故本文件不含「文献事实」。")
        lines.append("")
    return "\n".join(lines)


def _evidence_summary(record: Mapping[str, Any], limit: int = 3) -> str:
    """The evidence as one cell: the first few contexts, then a count, never all of them."""
    items = record["evidence"]
    parts = [f"{_where(item['context'])}：{_figures(item)}" for item in items[:limit]]
    if len(items) > limit:
        parts.append(f"…等 {len(items)} 条")
    return "；".join(parts)


def conclusion_rows(bundle: Mapping[str, Any]) -> list[list[Any]]:
    """Rows for the workbook sheet, in the same order as the Markdown."""
    rows: list[list[Any]] = []
    for record in bundle["conclusions"]:
        if record["kind"] == "推断":
            extra = "依据：" + ", ".join(record["cites"])
        elif record["kind"] == "文献事实":
            extra = "来源：" + str(record["source"])
        else:
            extra = _evidence_summary(record)
        rows.append(
            [
                record["conclusion_id"],
                record["kind"],
                record["method"],
                _signals_text(record),
                record["statement"],
                extra,
            ]
        )
    return rows
