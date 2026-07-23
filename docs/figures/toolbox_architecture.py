"""Generate the PU Learning Toolbox architecture diagram (academic style)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


# ── visual constants ────────────────────────────────────────────────────
IMPL_COLOR = "#2B2B2B"
IMPL_TEXT = "white"
PLAN_COLOR = "white"
PLAN_EDGE = "#888888"
PLAN_TEXT = "#777777"
LAYER_BG = "#F3F3F3"
LAYER_EDGE = "#BBBBBB"
LAYER_LABEL_COLOR = "#333333"

FIG_W = 14
LAYER_X = 0.6
LAYER_W = 12.8
LAYER_H = 1.20
LAYER_GAP = 0.12
BOX_H = 0.58
FONT_LAYER = 12.5
FONT_BOX = 10.5
FONT_SUB = 8.5

# ── layer definitions (bottom → top) ───────────────────────────────────
LAYERS = [
    {
        "label": "Core Infrastructure",
        "modules": [
            {"name": "core/", "sub": "Base classes & validation", "impl": True, "w": 3.5},
            {"name": "registry/", "sub": "Algorithm metadata & routing", "impl": True, "w": 3.0},
            {"name": "utils/", "sub": "Shared basis functions", "impl": True, "w": 1.8},
            {"name": "preprocessing/", "sub": "PU label generation & profiling", "impl": True, "w": 3.8},
        ],
    },
    {
        "label": "Estimation",
        "modules": [
            {"name": "prior/", "sub": "Class-prior estimation", "impl": "partial", "w": 4.2},
            {"name": "losses/", "sub": "PU risk functions", "impl": True, "w": 4.2},
        ],
    },
    {
        "label": "Algorithms  (estimators/)",
        "modules": [
            {"name": "classic/", "sub": "Calibration-based PU", "impl": True, "w": 2.8},
            {"name": "risk/", "sub": "Risk-based PU classifiers", "impl": True, "w": 3.4},
            {"name": "bias_aware/", "sub": "Selection-bias PU", "impl": True, "w": 2.6},
            {"name": "deep/", "sub": "Deep PU methods (planned)", "impl": False, "w": 2.6},
        ],
    },
    {
        "label": "Source Integration",
        "modules": [
            {"name": "source_adapters/", "sub": "Third-party code wrappers", "impl": "partial", "w": 4.2},
        ],
    },
    {
        "label": "Evaluation",
        "modules": [
            {"name": "metrics/", "sub": "PU classification metrics", "impl": True, "w": 2.8},
            {"name": "model_selection/", "sub": "PU cross-validation", "impl": True, "w": 3.4},
            {"name": "benchmarks/", "sub": "(planned)", "impl": False, "w": 2.8},
        ],
    },
    {
        "label": "User Layer",
        "modules": [
            {"name": "advisor/", "sub": "Algorithm recommender (planned)", "impl": False, "w": 3.2},
            {"name": "examples/", "sub": "Minimal usage demos", "impl": True, "w": 2.8},
        ],
    },
]


def _draw_module_box(ax, x, y, w, h, name, sub, impl):
    if impl is True:
        fc, ec, lw, ls = IMPL_COLOR, IMPL_COLOR, 1.4, "-"
        name_color, sub_color = IMPL_TEXT, "#CCCCCC"
    elif impl == "partial":
        fc, ec, lw, ls = "#666666", "#444444", 1.4, "-"
        name_color, sub_color = "white", "#D0D0D0"
    else:
        fc, ec, lw, ls = PLAN_COLOR, PLAN_EDGE, 1.2, (0, (4, 3))
        name_color, sub_color = "#444444", PLAN_TEXT

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, name,
            ha="center", va="center", fontsize=FONT_BOX,
            fontweight="bold", color=name_color, family="monospace")
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub,
                ha="center", va="center", fontsize=FONT_SUB,
                color=sub_color)


def _draw_layer(ax, idx, layer_def):
    y_base = 0.5
    y = y_base + idx * (LAYER_H + LAYER_GAP)
    layer_rect = FancyBboxPatch(
        (LAYER_X, y), LAYER_W, LAYER_H,
        boxstyle="round,pad=0.08",
        facecolor=LAYER_BG, edgecolor=LAYER_EDGE, linewidth=0.9,
    )
    ax.add_patch(layer_rect)

    ax.text(LAYER_X + 0.3, y + LAYER_H - 0.12, layer_def["label"],
            ha="left", va="top", fontsize=FONT_LAYER,
            fontweight="bold", color=LAYER_LABEL_COLOR)

    modules = layer_def["modules"]
    total_w = sum(m["w"] for m in modules)
    gap = (LAYER_W - 0.8 - total_w) / max(len(modules) - 1, 1) if len(modules) > 1 else 0
    gap = min(gap, 0.45)
    total_with_gaps = total_w + gap * (len(modules) - 1)
    start_x = LAYER_X + (LAYER_W - total_with_gaps) / 2

    cx = start_x
    box_y = y + 0.18
    for m in modules:
        _draw_module_box(ax, cx, box_y, m["w"], BOX_H, m["name"], m["sub"], m["impl"])
        cx += m["w"] + gap


def _draw_dependency_arrow(ax):
    y_base = 0.5
    arrow_x = LAYER_X + LAYER_W + 0.6
    y_bottom = y_base + 0.1
    y_top = y_base + len(LAYERS) * LAYER_H + (len(LAYERS) - 1) * LAYER_GAP - 0.1
    ax.annotate(
        "", xy=(arrow_x, y_top), xytext=(arrow_x, y_bottom),
        arrowprops=dict(
            arrowstyle="->,head_width=0.5,head_length=0.3",
            color="#666666", lw=3.0,
        ),
    )
    mid_y = (y_bottom + y_top) / 2
    ax.text(arrow_x + 0.35, mid_y, "Dependency",
            ha="left", va="center", fontsize=13, color="#666666",
            fontweight="bold", rotation=90)


def _draw_legend(ax):
    impl_patch = mpatches.Patch(facecolor=IMPL_COLOR, edgecolor=IMPL_COLOR,
                                label="Implemented")
    partial_patch = mpatches.Patch(facecolor="#666666", edgecolor="#444444",
                                  label="Partially implemented")
    plan_patch = mpatches.Patch(facecolor=PLAN_COLOR, edgecolor=PLAN_EDGE,
                                linestyle="--", linewidth=1.2, label="Planned")
    ax.legend(
        handles=[impl_patch, partial_patch, plan_patch],
        loc="lower center", ncol=3, frameon=True,
        fontsize=10, edgecolor="#BBBBBB", fancybox=True,
        bbox_to_anchor=(0.5, -0.01),
    )


def main():
    total_h = 0.5 + len(LAYERS) * LAYER_H + (len(LAYERS) - 1) * LAYER_GAP + 0.3
    fig_w = FIG_W + 1.6
    fig, ax = plt.subplots(figsize=(fig_w, total_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, total_h)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, layer in enumerate(LAYERS):
        _draw_layer(ax, i, layer)
    _draw_dependency_arrow(ax)
    _draw_legend(ax)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    fig.savefig("docs/figures/toolbox_architecture.png", dpi=300,
                bbox_inches="tight", facecolor="white", pad_inches=0.15)
    fig.savefig("docs/figures/toolbox_architecture.svg",
                bbox_inches="tight", facecolor="white", pad_inches=0.15)
    print("Saved: docs/figures/toolbox_architecture.png")
    print("Saved: docs/figures/toolbox_architecture.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
