#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Generate the main marketshare shift figure (Figure 2 in the paper).
Shows AI-associated word growth vs baseline word stagnation across 34 languages.
Uses content-only LPR-20 as primary analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

LANG_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "cs": "Czech",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "hi": "Hindi", "hr": "Croatian", "id": "Indonesian", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "kk": "Kazakh", "ko": "Korean",
    "ky": "Kyrgyz", "lt": "Lithuanian", "lv": "Latvian", "mr": "Marathi",
    "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sr": "Serbian", "ta": "Tamil", "tr": "Turkish",
    "uk": "Ukrainian", "zh": "Chinese",
}


def make_figure(tsv_path, out_path, title_suffix=""):
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["language"] != "hu"]  # Exclude Hungarian

    df["ai_growth"] = df["ai_growth_pct"] * 100
    df["base_growth"] = df["base_growth_pct"] * 100
    df["lang_name"] = df["language"].map(LANG_NAMES)
    df = df.sort_values("ai_growth")
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(4.2, 6.8))

    y = np.arange(len(df))

    # Horizontal lines connecting AI and baseline for each language
    for i, row in df.iterrows():
        ax.plot(
            [row["base_growth"], row["ai_growth"]],
            [y[i], y[i]],
            color="#cccccc", linewidth=0.5, zorder=1,
        )

    # Baseline words (gray open circles)
    ax.scatter(
        df["base_growth"], y,
        color="white", edgecolors="#888888", linewidths=0.8,
        marker="o", s=22, label="Baseline words", zorder=3,
    )

    # AI words: blue if significant, red outline if not
    sig_mask = df["ai_is_sig"] == True
    ax.scatter(
        df.loc[sig_mask, "ai_growth"], y[sig_mask],
        color="#2166ac", edgecolors="#2166ac", linewidths=0.5,
        marker="s", s=22, label="AI-associated (sig.)", zorder=4,
    )
    if (~sig_mask).any():
        ax.scatter(
            df.loc[~sig_mask, "ai_growth"], y[~sig_mask],
            color="white", edgecolors="#d6604d", linewidths=1.0,
            marker="s", s=22, label="AI-associated (n.s.)", zorder=4,
        )

    # Zero line
    ax.axvline(x=0, color="black", linewidth=0.4, linestyle="-", zorder=0)

    # Mean lines
    ai_mean = df["ai_growth"].mean()
    base_mean = df["base_growth"].mean()
    ax.axvline(x=ai_mean, color="#2166ac", linewidth=0.8, linestyle="--",
               alpha=0.6, zorder=2)
    ax.axvline(x=base_mean, color="#888888", linewidth=0.8, linestyle="--",
               alpha=0.6, zorder=2)

    # Labels
    ax.set_yticks(y)
    ax.set_yticklabels(df["lang_name"], fontsize=8)
    ax.set_xlabel("Prevalence change, pre- vs.\u2009post-GPT release (%)")
    ax.set_xlim(-100, 100)

    # Legend
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="none", fontsize=8)

    # Light grid
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Primary: content-only
    make_figure(
        os.path.join(base, "out/ig/marketshare_analysis-content-lpr-20/marketshare_summary.tsv"),
        os.path.join(base, "figures/fig_marketshare_content.pdf"),
    )
    # Secondary: all words
    make_figure(
        os.path.join(base, "out/ig/marketshare_analysis-lpr-20/marketshare_summary.tsv"),
        os.path.join(base, "figures/fig_marketshare_all.pdf"),
    )
