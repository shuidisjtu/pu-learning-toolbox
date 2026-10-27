# pu_toolbox/experiment/survey_diagnostics.py

"""Per-method diagnostic cards from the summary report (execution plan §1.4.5).

A card lists, for one method, what each signal of the §1.4.5 table shows across the
datasets, paths, views and ``c`` values the report holds, and flags the shapes the
table calls anomalous.  Three rules keep it from over-claiming:

* **A flag is a prompt to look, never a verdict.**  The thresholds below are the
  card's own heuristics, not protocol numbers; a flag says "go back to the
  implementation, the selection and the data view", and the rendered card says so.
* **Only complete rows count.**  A row short of the protocol's seeds is excluded and
  counted, because a single-seed reading does not enter a conclusion table.
* **What cannot be read from the report is listed, not skipped.**  The class-prior,
  cross-dataset-trend and external-anchor signals need manifest fields or the
  comparison matrix, so every card names them as not computed and says why.  The
  card has no anchor signal at all, so a method with no direct anchor is never
  described as agreeing or disagreeing with a paper.

Every observation is a ``本实验观测``; the direction to check is an ``推断``.  A
``文献事实`` never appears here because the card cites no source.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

#: Signals the card computes from report rows, in §1.4.5 order.
SIGNALS: tuple[str, ...] = (
    "mechanism_scar_to_sar",
    "view_os_ts",
    "backbone_path",
    "c_sensitivity",
    "pn_oracle_alignment",
    "pa_oa_gap",
    "seed_variance",
)

#: Signals §1.4.5 lists that the report rows cannot answer, and why.
NOT_COMPUTED: Mapping[str, str] = {
    "class_prior_semantics": (
        "needs c_realized, c_requested and the three priors from the manifest; "
        "the summary report keeps only the aggregated accuracy"
    ),
    "cross_dataset_trend": (
        "needs the direction a paper claims for the method, which is a literature "
        "fact the report does not hold"
    ),
    "external_anchor_deviation": (
        "needs the comparison matrix; a method with no direct anchor must not be "
        "read as agreeing or disagreeing with a paper"
    ),
    "native_assumption_split": (
        "the ledger records no structured field saying which methods are native to "
        "SAR, so the native-versus-violating contrast is not formed; "
        "mechanism_scar_to_sar covers only the 'SAR is better than SCAR' half"
    ),
    "backbone_rank_reversal": (
        "needs the exception-method tags (private network) and a peer ranking; "
        "backbone_path only shows one method's spread across two training paths"
    ),
}

#: Heuristic thresholds, in accuracy points.  They only decide when to print a flag.
CLIFF_POINTS = 10.0  # one adjacent-c step larger than this is a cliff
OS_TS_NO_EFFECT_POINTS = 0.5  # |TS - OS| below this: calibration changed nothing
PATH_GAP_POINTS = 5.0  # same method, two training paths, apart by more than this
PA_OA_GAP_POINTS = 5.0  # OA ahead of PA by more than this
ORACLE_GAP_SPREAD_POINTS = 15.0  # gap to the oracle varying this much across datasets

_C_ORDER = ("0.1", "0.3", "0.5")

_DIRECTIONS: Mapping[str, str] = {
    "mechanism_scar_to_sar": "标注机制处理、权重/score 更新是否真的随机制变化",
    "view_os_ts": "校准实现、未标记风险项是否被真正替换",
    "backbone_path": "表征适配、私有网络的架构影响",
    "c_sensitivity": "标记频率机制、风险估计对 n_L 的依赖",
    "pn_oracle_alignment": "泄漏、评测口径或划分一致性",
    "pa_oa_gap": "选模健康度、是否依赖真实标签",
    "seed_variance": "初始化与调参稳定性",
}


def _complete(row: Mapping[str, Any]) -> bool:
    metric = row["metric"]
    return not metric.get("missing_seeds") and metric.get("mean") is not None


def _points(row: Mapping[str, Any]) -> float:
    return float(row["metric"]["mean"]) * 100.0


def _std_points(row: Mapping[str, Any]) -> float:
    std = row["metric"].get("std")
    return 0.0 if std is None else float(std) * 100.0


def _context(row: Mapping[str, Any]) -> dict[str, str]:
    key = row["row_key"]
    return {
        "dataset": row["result_identity"]["dataset"],
        "training_path": row["result_identity"]["training_path"],
        "run_view": key["run_view"],
        "mechanism": key["mechanism"],
        "selection_protocol": key["selection_protocol"],
        "c_token": str(key["c_token"]),
    }


def _obs(row_context: Mapping[str, Any], flag: bool, **values: Any) -> dict[str, Any]:
    return {"context": dict(row_context), "flag": flag, **values}


def _index(rows: list[Mapping[str, Any]], *fields: str) -> dict[tuple, Mapping[str, Any]]:
    """Rows keyed by the named context fields; the first wins, so order must be stable."""
    index: dict[tuple, Mapping[str, Any]] = {}
    for row in rows:
        ctx = _context(row)
        index.setdefault(tuple(ctx[field] for field in fields), row)
    return index


def _sorted_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: tuple(_context(row).values()) + (row["row_key"]["method"],))


def _mechanism(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scar = _index(
        [r for r in rows if r["row_key"]["mechanism"] == "scar"],
        "dataset",
        "training_path",
        "run_view",
        "selection_protocol",
        "c_token",
    )
    out = []
    for row in rows:
        ctx = _context(row)
        if not ctx["mechanism"].startswith("sar") or ctx["selection_protocol"] != "OA":
            continue
        base = scar.get(
            (ctx["dataset"], ctx["training_path"], ctx["run_view"], "OA", ctx["c_token"])
        )
        if base is None:
            continue
        delta = _points(row) - _points(base)
        out.append(
            _obs(
                ctx,
                delta > 0,
                scar_points=_points(base),
                sar_points=_points(row),
                delta_points=delta,
            )
        )
    return out


def _views(rows: list[Mapping[str, Any]], native: str | None) -> list[dict[str, Any]]:
    by_os = _index(
        [r for r in rows if r["row_key"]["run_view"] == "os-compatible"],
        "dataset",
        "training_path",
        "mechanism",
        "selection_protocol",
        "c_token",
    )
    out = []
    for row in rows:
        ctx = _context(row)
        if ctx["run_view"] != "ts-compatible":
            continue
        base = by_os.get(
            (
                ctx["dataset"],
                ctx["training_path"],
                ctx["mechanism"],
                ctx["selection_protocol"],
                ctx["c_token"],
            )
        )
        if base is None:
            continue
        delta = _points(row) - _points(base)
        flag = native == "ts" and (delta < 0 or abs(delta) < OS_TS_NO_EFFECT_POINTS)
        out.append(
            _obs(
                ctx,
                flag,
                os_points=_points(base),
                ts_points=_points(row),
                delta_points=delta,
                native_view=native,
            )
        )
    return out


def _paths(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        ctx = _context(row)
        groups[
            (
                ctx["dataset"],
                ctx["run_view"],
                ctx["mechanism"],
                ctx["selection_protocol"],
                ctx["c_token"],
            )
        ].append(row)
    out = []
    for key in sorted(groups):
        members = groups[key]
        if len({_context(r)["training_path"] for r in members}) < 2:
            continue
        values = {_context(r)["training_path"]: _points(r) for r in members}
        spread = max(values.values()) - min(values.values())
        ctx = _context(members[0])
        ctx["training_path"] = "/".join(sorted(values))
        out.append(
            _obs(
                ctx,
                spread > PATH_GAP_POINTS,
                points_by_path=dict(sorted(values.items())),
                spread_points=spread,
            )
        )
    return out


def _c_curves(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        ctx = _context(row)
        if ctx["mechanism"] != "scar" or ctx["c_token"] not in _C_ORDER:
            continue
        groups[(ctx["dataset"], ctx["training_path"], ctx["run_view"], ctx["selection_protocol"])][
            ctx["c_token"]
        ] = row
    out = []
    for key in sorted(groups):
        curve = groups[key]
        tokens = [c for c in _C_ORDER if c in curve]
        if len(tokens) < 3:
            continue
        values = [_points(curve[c]) for c in tokens]
        noise = max(_std_points(curve[c]) for c in tokens)
        steps = [b - a for a, b in zip(values, values[1:], strict=False)]
        non_monotone = any(step < -noise for step in steps)
        cliff = any(abs(step) > CLIFF_POINTS for step in steps)
        insensitive = max(values) - min(values) < max(noise, 1e-9)
        ctx = _context(curve[tokens[0]])
        ctx["c_token"] = "/".join(tokens)
        out.append(
            _obs(
                ctx,
                non_monotone or cliff or insensitive,
                points_by_c=dict(zip(tokens, values, strict=True)),
                non_monotone=non_monotone,
                cliff=cliff,
                insensitive=insensitive,
            )
        )
    return out


def _oracle(
    rows: list[Mapping[str, Any]], oracle_rows: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    oracle = {}
    for row in oracle_rows:
        ctx = _context(row)
        if ctx["selection_protocol"] == "OA":
            oracle.setdefault((ctx["dataset"], ctx["training_path"]), row)
    out = []
    gaps = []
    for row in rows:
        ctx = _context(row)
        base = oracle.get((ctx["dataset"], ctx["training_path"]))
        if base is None or ctx["selection_protocol"] != "OA" or ctx["mechanism"] != "scar":
            continue
        gap = _points(base) - _points(row)
        gaps.append(gap)
        out.append(
            _obs(
                ctx,
                gap < 0,
                oracle_points=_points(base),
                method_points=_points(row),
                gap_points=gap,
            )
        )
    if gaps and max(gaps) - min(gaps) > ORACLE_GAP_SPREAD_POINTS:
        for item in out:
            item["flag"] = True
            item["gap_varies_across_rows"] = True
    return out


def _pa_oa(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pa = _index(
        [r for r in rows if r["row_key"]["selection_protocol"] == "PA"],
        "dataset",
        "training_path",
        "run_view",
        "mechanism",
        "c_token",
    )
    out = []
    for row in rows:
        ctx = _context(row)
        if ctx["selection_protocol"] != "OA":
            continue
        base = pa.get(
            (
                ctx["dataset"],
                ctx["training_path"],
                ctx["run_view"],
                ctx["mechanism"],
                ctx["c_token"],
            )
        )
        if base is None:
            continue
        gap = _points(row) - _points(base)
        out.append(
            _obs(
                ctx,
                gap > PA_OA_GAP_POINTS,
                oa_points=_points(row),
                pa_points=_points(base),
                gap_points=gap,
            )
        )
    return out


def _variance(
    method: str, rows: list[Mapping[str, Any]], everyone: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ctx = _context(row)
        peers = [
            _points(o)
            for o in everyone
            if o["row_key"]["method"] not in (method, "pn_oracle")
            and all(
                _context(o)[f] == ctx[f]
                for f in (
                    "dataset",
                    "training_path",
                    "run_view",
                    "mechanism",
                    "selection_protocol",
                    "c_token",
                )
            )
        ]
        if not peers:
            continue
        nearest = min(abs(_points(row) - p) for p in peers)
        std = _std_points(row)
        out.append(_obs(ctx, std > nearest, std_points=std, nearest_gap_points=nearest))
    return out


def build_cards(
    report: Mapping[str, Any], *, native_view: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """One card per method; ``native_view`` maps a method to ``os`` / ``ts`` / ``both``."""
    everything = _sorted_rows(list(report.get("rows", [])))
    complete = [r for r in everything if _complete(r)]
    oracle_rows = [r for r in complete if r["row_key"]["mechanism"] == "pn_oracle"]
    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    excluded: dict[str, int] = defaultdict(int)
    for row in everything:
        method = row["row_key"]["method"]
        if method == "pn_oracle":
            continue  # the oracle is a reference line for the others, not a card
        if _complete(row):
            by_method[method].append(row)
        else:
            excluded[method] += 1
    methods = sorted({*by_method, *excluded})
    cards: dict[str, dict[str, Any]] = {}
    for method in methods:
        rows = by_method.get(method, [])
        cards[method] = {
            "method": method,
            "native_view": native_view.get(method),
            "rows_used": len(rows),
            "excluded_rows": excluded.get(method, 0),
            "signals": {
                "mechanism_scar_to_sar": _mechanism(rows),
                "view_os_ts": _views(rows, native_view.get(method)),
                "backbone_path": _paths(rows),
                "c_sensitivity": _c_curves(rows),
                "pn_oracle_alignment": _oracle(rows, oracle_rows),
                "pa_oa_gap": _pa_oa(rows),
                "seed_variance": _variance(
                    method, rows, [r for r in complete if r["row_key"]["mechanism"] != "pn_oracle"]
                ),
            },
            "not_computed": [{"signal": name, "reason": why} for name, why in NOT_COMPUTED.items()],
        }
    return cards


def _describe(signal: str, item: Mapping[str, Any]) -> str:
    ctx = item["context"]
    where = (
        " / ".join(
            (ctx["dataset"], ctx["training_path"], ctx["run_view"], ctx["selection_protocol"])
        )
        + f" / c={ctx['c_token']}"
    )
    detail = ", ".join(
        f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in item.items()
        if key not in ("context", "flag")
    )
    return f"{where}: {detail}"


def render_cards(cards: Mapping[str, Mapping[str, Any]]) -> str:
    """The cards as Markdown, one section per method."""
    lines = [
        "# 方法诊断卡",
        "",
        "异常标记只是提示，不判定实现有错；标记后要回看该方法的实现、选模与数据视图。",
        "每条观测标为「本实验观测」，待查方向标为「推断」；本卡不引用文献，不含「文献事实」。",
        "单 seed 或缺 seed 的行不进任何信号，只计入「已排除行」。",
        "",
    ]
    for method in sorted(cards):
        card = cards[method]
        lines += [
            f"## {method}",
            "",
            f"原生视图：{card['native_view'] or '台账未给'}；"
            f"已用行 {card['rows_used']}，已排除行 {card['excluded_rows']}。",
            "",
        ]
        for signal in SIGNALS:
            items = card["signals"][signal]
            flagged = [item for item in items if item["flag"]]
            lines.append(f"### {signal}（观测 {len(items)} 条，标记 {len(flagged)} 条）")
            if not items:
                lines.append("- 本实验观测：报告里没有可比较的行")
            for item in items:
                mark = "⚠ " if item["flag"] else ""
                lines.append(f"- 本实验观测：{mark}{_describe(signal, item)}")
            if flagged:
                lines.append(f"- 推断（待查方向）：{_DIRECTIONS[signal]}")
            lines.append("")
        lines.append("### 未计算的信号")
        for item in card["not_computed"]:
            name = (
                "外部锚点偏差" if item["signal"] == "external_anchor_deviation" else item["signal"]
            )
            lines.append(f"- {name}：{item['reason']}")
        lines.append("")
    return "\n".join(lines)
