#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""Cross-lingual alignment overview figure (small, for page-1 column).

Grouped vertical bars: AI lemmas vs Baseline vs Permutation null,
at K=50 and K=200. Designed to fit in a single column (~3.3 in wide).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style (Clean, Academic) ──────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

# ── Data ─────────────────────────────────────────────────────────────────

DATA = {
    "K = 50": {
        "AI lemmas":  0.658,
        "Baseline":   0.606,
        "Null":       0.611,
    },
    "K = 200": {
        "AI lemmas":  0.730,
        "Baseline":   0.657,
        "Null":       0.668,
    },
}

# Standard errors (SE = std / sqrt(n))
# Null: std from permutation distribution (B=1000)
# AI & Baseline: SE from 20 per-seed mean-max-cosine values
STDERR = {
    "K = 50": {
        "AI lemmas": 0.019,
        "Baseline":  0.017,
        "Null":      0.019,
    },
    "K = 200": {
        "AI lemmas": 0.019,
        "Baseline":  0.016,
        "Null":      0.019,
    },
}

# Significance: AI vs Null (permutation test)
SIG_AI_NULL = {
    "K = 50":  "*",    # p = 0.017
    "K = 200": "**",   # p = 0.006
}

# Significance: AI vs Baseline (Mann-Whitney U, one-sided)
SIG_AI_BL = {
    "K = 50":  "*",    # p = 0.027
    "K = 200": "**",   # p = 0.002
}

COLORS = {
    "AI lemmas": "#3E6FB0",
    "Baseline":  "#4C9A5F",
    "Null":      "#C7C7C7",
}

# ── Plot ─────────────────────────────────────────────────────────────────

def plot():
    groups = list(DATA.keys())          
    # Ordered logically: Primary Method -> Baseline -> Null distribution
    series = ["AI lemmas", "Baseline", "Null"]
    n_groups = len(groups)
    n_series = len(series)

    bar_width = 0.25
    x = np.arange(n_groups)

    # Made the figure slightly taller to accommodate the significance brackets
    fig, ax = plt.subplots(figsize=(3.3, 3.2))

    # Store coordinates for the significance brackets later
    bar_coords = {}

    for i, s in enumerate(series):
        vals = [DATA[g][s] for g in groups]
        offset = (i - (n_series - 1) / 2) * bar_width
        
        bars = ax.bar(
            x + offset, vals, bar_width,
            color=COLORS[s], edgecolor="white", linewidth=0.8,
            label=s, zorder=3, alpha=0.95
        )
        bar_coords[s] = x + offset
        
        # Error bars (SE for all series)
        errs = [STDERR[g][s] for g in groups]
        ax.errorbar(
            x + offset, vals, yerr=errs,
            fmt="none", ecolor="#666666", elinewidth=1.0, capsize=2.5,
            zorder=4,
        )

    # Significance brackets
    for j, g in enumerate(groups):
        max_h = max(
            DATA[g][s] + STDERR[g][s] for s in ["AI lemmas", "Baseline", "Null"]
        )

        # ── Bracket 1: AI vs Baseline (lower bracket) ──
        x_ai = bar_coords["AI lemmas"][j]
        x_bl = bar_coords["Baseline"][j]

        y_base1 = max_h + 0.012
        y_top1 = y_base1 + 0.008

        ax.plot([x_ai, x_ai, x_bl, x_bl], [y_base1, y_top1, y_top1, y_base1],
                lw=1.0, c="#666666")
        ax.text(
            (x_ai + x_bl) / 2, y_top1 + 0.002,
            SIG_AI_BL[g],
            ha="center", va="bottom", fontsize=10, color="#666666", fontweight="bold"
        )

        # ── Bracket 2: AI vs Null (upper bracket) ──
        x_null = bar_coords["Null"][j]

        y_base2 = y_top1 + 0.022
        y_top2 = y_base2 + 0.008

        ax.plot([x_ai, x_ai, x_null, x_null], [y_base2, y_top2, y_top2, y_base2],
                lw=1.0, c="#666666")
        ax.text(
            (x_ai + x_null) / 2, y_top2 + 0.002,
            SIG_AI_NULL[g],
            ha="center", va="bottom", fontsize=10, color="#666666", fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Mean max-cosine similarity")
    ax.set_ylim(0.50, 0.88)  # Raised ceiling for two bracket tiers

    # Cleaner Grid
    ax.grid(axis="y", color="#e0e0e0", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Clean up Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(colors="#333333")

    # Legend
    ax.legend(loc="upper left", frameon=False, ncol=1)

    fig.tight_layout()

    outdir = Path(__file__).resolve().parent
    for ext in ("pdf", "png"):
        out = outdir / f"fig_crosslingual_alignment.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}", file=sys.stderr)

    plt.close(fig)

if __name__ == "__main__":
    plot()
