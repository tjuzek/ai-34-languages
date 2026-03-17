#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""Diachronic visualisation of AI-overused word concepts across languages.

Three panels: importance (NOUN), emphasise (VERB), care/rigour (ADJ).
Y-axis: % change from pre-GPT mean (2012–2021).
X-axis: years 2012–2024.

Caches the raw OPM data to aiwl/out/ig/diachronic_opm/fig_diachronic_cache.json
so that subsequent figure iterations don't require re-scanning POS files.
Delete the cache file to force a re-scan.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ── Word lists per concept ──────────────────────────────────────────────

CONCEPTS = {
    r"$\it{care/rigour}$" + "  (ADJ)": [
        ("cs", "pečlivý", "ADJ"),
        ("cs", "precizní", "ADJ"),
        ("de", "präzisen", "ADJ"),
        ("de", "präzis", "ADJ"),
        ("de", "sorgfältig", "ADJ"),
        ("en", "thorough", "ADJ"),
        ("es", "impecable", "ADJ"),
        ("fr", "rigide", "ADJ"),
        ("hi", "समग्र", "ADJ"),
        ("it", "mirato", "ADJ"),
        ("it", "impeccabile", "ADJ"),
        ("it", "approfondito", "ADJ"),
        ("pt", "rigoroso", "ADJ"),
        ("ru", "внимательный", "ADJ"),
        ("zh", "精湛", "ADJ"),
    ],
    r"$\it{emphasise}$" + "  (VERB)": [
        ("cs", "zdůraznit", "VERB"),
        ("cs", "zdůrazňovat", "VERB"),
        ("cs", "vyzdvihnout", "VERB"),
        ("de", "hervorheben", "VERB"),
        ("en", "emphasize", "VERB"),
        ("en", "highlight", "VERB"),
        ("es", "realzar", "VERB"),
        ("fr", "marquer", "VERB"),
        ("it", "evidenziare", "VERB"),
        ("pt", "enfatizar", "VERB"),
        ("ru", "подчеркивать", "VERB"),
    ],
    r"$\it{importance}$" + "  (NOUN)": [
        ("cs", "důležitost", "NOUN"),
        ("de", "bedeutung", "NOUN"),
        ("en", "importance", "NOUN"),
        ("fr", "importance", "NOUN"),
        ("it", "importanza", "NOUN"),
        ("pt", "importância", "NOUN"),
    ],
}

YEARS = list(range(2012, 2025))
PRE_GPT_YEARS = list(range(2012, 2022))  # 2012–2021
POS_DIR = Path("data/diachronic-rawdata-filtered-sampled-pos")
CACHE_PATH = Path("out/ig/diachronic_opm/fig_diachronic_cache.json")

# ── Language display labels ─────────────────────────────────────────────

LANG_NAMES = {
    "cs": "Czech", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "hi": "Hindi", "it": "Italian", "pt": "Portuguese",
    "ru": "Russian", "zh": "Chinese",
}

# ── Colour palette (one per language) ───────────────────────────────────

LANG_COLORS = {
    "cs": "#e41a1c", "de": "#377eb8", "en": "#4daf4a", "es": "#984ea3",
    "fr": "#ff7f00", "hi": "#a65628", "it": "#f781bf", "pt": "#808080",
    "ru": "#17becf", "zh": "#bcbd22",
}

# ── Scanning ────────────────────────────────────────────────────────────

def scan_pos_file(pos_path, target_keys):
    """Single-pass scan. Returns {key: count} and total_tokens."""
    counts = {k: 0 for k in target_keys}
    total = 0
    with open(pos_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line or line[0] == "#" or line[0] == "\n":
                continue
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            parts = stripped.split("\t")
            if len(parts) < 4:
                continue
            lemma = parts[2].lower().replace("ё", "е")
            upos = parts[3]
            key = f"{lemma}_{upos}"
            if key in counts:
                counts[key] += 1
    return counts, total


def collect_opm():
    """Collect OPM for every (lang, key, year). Uses cache if available."""
    # Try loading from cache
    if CACHE_PATH.exists():
        print(f"Loading cached data from {CACHE_PATH}", file=sys.stderr)
        with open(CACHE_PATH) as f:
            cached = json.load(f)
        # Convert keys back from "lang|key|year" strings
        opm = {}
        for k, v in cached["opm"].items():
            lang, key, year = k.rsplit("|", 2)
            opm[(lang, key, int(year))] = v
        print(f"  {len(opm)} data points loaded.", file=sys.stderr)
        return opm

    # No cache — scan POS files
    lang_keys = defaultdict(set)
    for items in CONCEPTS.values():
        for lang, lemma, upos in items:
            key = f"{lemma}_{upos}"
            lang_keys[lang].add(key)

    opm = {}  # (lang, key, year) -> opm
    totals = {}  # (lang, year) -> total_tokens
    for lang, keys in sorted(lang_keys.items()):
        for year in YEARS:
            pos_path = POS_DIR / lang / f"{year}.pos"
            if not pos_path.exists():
                print(f"  [skip] {pos_path}", file=sys.stderr)
                continue
            print(f"  {lang}/{year} ...", end="", flush=True, file=sys.stderr)
            counts, total = scan_pos_file(pos_path, keys)
            totals[(lang, year)] = total
            for key, cnt in counts.items():
                opm[(lang, key, year)] = (cnt / total) * 1_000_000 if total > 0 else 0.0
            print(f" {total:,} tok", file=sys.stderr)

    # Save cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "opm": {f"{lang}|{key}|{year}": val for (lang, key, year), val in opm.items()},
        "totals": {f"{lang}|{year}": val for (lang, year), val in totals.items()},
    }
    with open(CACHE_PATH, "w") as f:
        json.dump(cache_data, f)
    print(f"\nCache saved to {CACHE_PATH} ({len(opm)} data points)", file=sys.stderr)

    return opm


def compute_pct_change(opm):
    """For each (lang, key), compute % change from pre-GPT mean per year."""
    pairs = set()
    for (lang, key, year) in opm:
        pairs.add((lang, key))

    result = {}  # (lang, key) -> {year: pct_change}
    for lang, key in pairs:
        pre_vals = [opm.get((lang, key, y), 0.0) for y in PRE_GPT_YEARS]
        pre_mean = np.mean(pre_vals)
        if pre_mean == 0:
            continue
        yearly = {}
        for y in YEARS:
            val = opm.get((lang, key, y), None)
            if val is not None:
                yearly[y] = ((val - pre_mean) / pre_mean) * 100
        result[(lang, key)] = yearly

    return result


# ── Plotting ────────────────────────────────────────────────────────────

def plot(pct_data):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Per-panel y-limits: care/rigour, emphasise, importance
    PANEL_YLIMS = [(-50, 250), (-50, 150), (-50, 100)]

    for panel_i, (ax, (concept_name, items)) in enumerate(
        zip(axes, CONCEPTS.items())
    ):
        ymin, ymax = PANEL_YLIMS[panel_i]
        # Background shading: white (pre-GPT), light grey (2022), grey (post-GPT)
        ax.axvspan(2021.5, 2022.5, color="#f0f0f0", zorder=0)   # 2022: transition year
        ax.axvspan(2022.5, YEARS[-1] + 0.5, color="#e0e0e0", zorder=0)  # 2023–2024: post-GPT

        # Individual language lines (thin, transparent)
        legend_langs = {}
        for lang, lemma, upos in items:
            key = f"{lemma}_{upos}"
            series = pct_data.get((lang, key))
            if series is None:
                print(f"  [warn] no data for {lang}/{key}", file=sys.stderr)
                continue

            years = sorted(series.keys())
            vals = [series[y] for y in years]
            color = LANG_COLORS.get(lang, "#333333")

            line, = ax.plot(years, vals, color=color, linewidth=0.8,
                           alpha=0.45, marker=".", markersize=2, zorder=2)

            if lang not in legend_langs:
                legend_langs[lang] = line

        # Cross-lingual mean (bold)
        all_years_vals = defaultdict(list)
        for lang, lemma, upos in items:
            key = f"{lemma}_{upos}"
            series = pct_data.get((lang, key))
            if series is None:
                continue
            for y, v in series.items():
                all_years_vals[y].append(v)

        mean_years = sorted(all_years_vals.keys())
        mean_vals = [np.mean(all_years_vals[y]) for y in mean_years]
        mean_line, = ax.plot(mean_years, mean_vals, color="black",
                            linewidth=2.5, zorder=4, marker="o",
                            markersize=4)

        # Reference line at 0%
        ax.axhline(0, color="#aaaaaa", linewidth=0.6, linestyle="-", zorder=1)

        ax.set_title(concept_name, fontsize=11, pad=8)
        ax.set_xlim(YEARS[0] - 0.3, YEARS[-1] + 0.3)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(YEARS)
        ax.set_xticklabels(
            [f"'{str(y)[2:]}" for y in YEARS], rotation=45, fontsize=8
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.tick_params(axis="y", labelsize=8)

        # Legend: languages + mean
        handles = list(legend_langs.values()) + [mean_line]
        labels = [LANG_NAMES.get(l, l) for l in legend_langs] + ["mean"]
        ax.legend(handles, labels, fontsize=9, loc="upper left",
                  framealpha=0.9, edgecolor="none", ncol=2,
                  handlelength=1.5, columnspacing=1.0)

    axes[0].set_ylabel("% change from\npre-GPT mean", fontsize=9,
                       rotation=90, labelpad=3)

    # Annotate shading zones in the importance panel (rightmost)
    ax_ann = axes[2]
    ann_y = -25
    for x, label, color in [
        (2021,  "pre-ChatGPT",      "#666666"),
        (2022,  "ChatGPT release",  "#555555"),
        (2023,   "post-ChatGPT",    "#555555"),
    ]:
        ax_ann.text(x, ann_y, label, rotation=90, fontsize=7.5,
                    color=color, ha="center", va="center",
                    fontstyle="italic")

    fig.tight_layout(w_pad=1.5)

    outdir = Path(__file__).resolve().parent
    for ext in ("pdf", "png"):
        out = outdir / f"fig_diachronic.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}", file=sys.stderr)

    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Collecting OPM ...", file=sys.stderr)
    opm = collect_opm()
    print("Computing % change ...", file=sys.stderr)
    pct = compute_pct_change(opm)
    print("Plotting ...", file=sys.stderr)
    plot(pct)
    print("Done.", file=sys.stderr)
