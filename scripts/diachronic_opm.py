#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""Diachronic OPM (Occurrences Per Million) analysis.

Computes per-word OPM across WMT News Crawl years (2020, 2021, 2023, 2024)
for AI-overused words and baseline words in each language.  Produces
per-language CSVs and a cross-language summary for Section 3.6 of the paper.

Input
-----
POS-tagged data at {pos-dir}/{lang}/{year}.pos (CoNLL format).
LPR inventories at {data-dir}/{lang}/las_word_{lang}.csv.

Output
------
{out-dir}/{lang}/opm_{lang}.csv  – per-word counts and OPM for each year
{out-dir}/diachronic_summary.csv – one row per language with aggregate stats
{out-dir}/summary.txt            – human-readable overview

Example
-------
    python scripts/diachronic_opm.py \\
        --data-dir out/ig/las-gpt4.1-mini-content-only-lpr/policy=quantile_trim=no_M=1_K=12_jitter=no \\
        --pos-dir data/ig/pos-basic-sample \\
        --out-dir out/ig/diachronic_opm \\
        --n-seeds 20 --seed 42
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV"}
YEARS = [2020, 2021, 2023, 2024]


# ---------------------------------------------------------------------------
# Data loading – word selection
# ---------------------------------------------------------------------------

def load_seeds_for_language(
    csv_path: Path, n_seeds: int, rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select AI seeds and UPOS-matched baseline seeds for one language.

    AI seeds: top-n content words by lpr_MH_guarded desc, lpr_guard_ok=1.
    Baseline seeds: n words with smallest |LAS|, UPOS-matched to AI seeds,
                    tie-broken by highest c_M.

    Returns (ai_df, baseline_df) each with at least columns:
        key, lemma, upos, lpr_MH_guarded
    """
    df = pd.read_csv(csv_path)

    # Normalise Russian ё → е (Stanza may produce ё in lemmas but POS files use е)
    df["key"] = df["key"].str.replace("ё", "е", regex=False)
    df["lemma"] = df["lemma"].str.replace("ё", "е", regex=False)

    # Content words that passed the guard
    content = df[(df["lpr_guard_ok"] == 1) & df["upos"].isin(CONTENT_UPOS)].copy()

    # AI seeds: highest lpr_MH_guarded
    ai_seeds = (
        content.sort_values("lpr_MH_guarded", ascending=False)
        .head(n_seeds)
        .reset_index(drop=True)
    )

    # Target UPOS distribution of AI seeds
    upos_counts = Counter(ai_seeds["upos"])

    # Baseline: smallest |LAS|, tie-broken by highest c_M, UPOS-matched
    ai_keys = set(ai_seeds["key"])
    candidates = content[~content["key"].isin(ai_keys)].copy()
    candidates["abs_las"] = candidates["LAS"].abs()
    candidates = candidates.sort_values(
        ["abs_las", "c_M"], ascending=[True, False]
    )

    baseline_parts = []
    for upos, count in upos_counts.items():
        upos_cands = candidates[candidates["upos"] == upos].head(count)
        baseline_parts.append(upos_cands)

    baseline_seeds = pd.concat(baseline_parts).reset_index(drop=True)

    return ai_seeds, baseline_seeds


# ---------------------------------------------------------------------------
# POS file scanning
# ---------------------------------------------------------------------------

def scan_pos_file(
    pos_path: Path, target_keys: set[str],
) -> tuple[dict[str, int], int]:
    """Single-pass scan of a CoNLL POS file.

    Returns (key_counts, total_tokens) where key_counts maps
    matched keys to their raw counts and total_tokens is the
    denominator for OPM.
    """
    counts: dict[str, int] = {k: 0 for k in target_keys}
    total = 0

    with open(pos_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line or line[0] == "#" or line[0] == "\n":
                continue
            # Blank lines (sentence boundary) – strip check
            stripped = line.strip()
            if not stripped:
                continue

            total += 1

            # CoNLL format: index \t form \t lemma \t upos
            # Use split with maxsplit for speed
            parts = stripped.split("\t")
            if len(parts) < 4:
                continue

            lemma = parts[2].lower().replace('ё', 'е')
            upos = parts[3]
            key = f"{lemma}_{upos}"

            if key in counts:
                counts[key] += 1

    return counts, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Diachronic OPM analysis of AI-overused vs baseline words.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    ap.add_argument(
        "--data-dir", type=Path, required=True,
        help="LPR output directory with per-language las_word_{lang}.csv files.",
    )
    ap.add_argument(
        "--pos-dir", type=Path, required=True,
        help="Directory with POS-tagged data: {lang}/{year}.pos.",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("out/ig/diachronic_opm"),
        help="Output directory.",
    )
    ap.add_argument(
        "--n-seeds", type=int, default=20,
        help="Number of AI seeds (and baseline seeds) per language.",
    )
    ap.add_argument(
        "--langs", type=str, nargs="+", default=None,
        help="Restrict to these language codes (default: all available).",
    )
    ap.add_argument(
        "--no-baseline", action="store_true",
        help="Skip baseline words, output only AI seeds.",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Discover languages (intersection of LPR dir and POS dir)
    lpr_langs = {d.name for d in args.data_dir.iterdir() if d.is_dir()}
    pos_langs = {d.name for d in args.pos_dir.iterdir() if d.is_dir()}
    langs = sorted(lpr_langs & pos_langs)
    if args.langs:
        langs = sorted(set(langs) & set(args.langs))

    print(f"Languages: {len(langs)}", file=sys.stderr)
    print(f"Years: {YEARS}", file=sys.stderr)
    print(f"Seeds per language: {args.n_seeds} AI + matched baseline", file=sys.stderr)
    print(file=sys.stderr)

    summary_rows = []

    for lang_i, lang in enumerate(langs, 1):
        print(
            f"[{lang_i}/{len(langs)}] {lang} ...",
            file=sys.stderr, end=" ", flush=True,
        )

        # -- Load word lists for this language --
        csv_path = args.data_dir / lang / f"las_word_{lang}.csv"
        if not csv_path.exists():
            print("SKIP (no CSV)", file=sys.stderr)
            continue

        ai_seeds, baseline_seeds = load_seeds_for_language(
            csv_path, args.n_seeds, rng,
        )

        # Build combined target set
        ai_key_set = set(ai_seeds["key"])
        bl_key_set = set() if args.no_baseline else set(baseline_seeds["key"])
        all_keys = ai_key_set | bl_key_set

        # -- Scan POS files for each year --
        year_counts: dict[int, dict[str, int]] = {}
        year_totals: dict[int, int] = {}

        for year in YEARS:
            pos_path = args.pos_dir / lang / f"{year}.pos"
            if not pos_path.exists():
                print(f"(missing {year})", file=sys.stderr, end=" ")
                year_counts[year] = {k: 0 for k in all_keys}
                year_totals[year] = 0
                continue

            counts, total = scan_pos_file(pos_path, all_keys)
            year_counts[year] = counts
            year_totals[year] = total

        print(
            " | ".join(f"{y}: {year_totals[y]:,} tok" for y in YEARS),
            file=sys.stderr,
        )

        # -- Build per-word rows --
        rows = []
        seed_groups = [("ai", ai_seeds)]
        if not args.no_baseline:
            seed_groups.append(("baseline", baseline_seeds))
        for seed_type, seed_df in seed_groups:
            for _, srow in seed_df.iterrows():
                key = srow["key"]
                lemma = srow["lemma"]
                upos = srow["upos"]
                lpr_rank = int(srow.get("rank_LPR", 0))
                lpr_value = srow["lpr_MH_guarded"]

                row = {
                    "key": key,
                    "lemma": lemma,
                    "upos": upos,
                    "seed_type": seed_type,
                    "lpr_rank": lpr_rank,
                    "lpr_value": lpr_value,
                }

                for year in YEARS:
                    row[f"count_{year}"] = year_counts[year].get(key, 0)
                    row[f"total_{year}"] = year_totals[year]

                    if year_totals[year] > 0:
                        row[f"opm_{year}"] = (
                            year_counts[year].get(key, 0)
                            / year_totals[year]
                            * 1_000_000
                        )
                    else:
                        row[f"opm_{year}"] = np.nan

                # Pre/post means
                opm_pre_vals = [
                    row[f"opm_{y}"]
                    for y in [2020, 2021]
                    if not (isinstance(row[f"opm_{y}"], float) and np.isnan(row[f"opm_{y}"]))
                ]
                opm_post_vals = [
                    row[f"opm_{y}"]
                    for y in [2023, 2024]
                    if not (isinstance(row[f"opm_{y}"], float) and np.isnan(row[f"opm_{y}"]))
                ]

                opm_pre = np.mean(opm_pre_vals) if opm_pre_vals else np.nan
                opm_post = np.mean(opm_post_vals) if opm_post_vals else np.nan

                row["opm_pre"] = opm_pre
                row["opm_post"] = opm_post

                if opm_pre and opm_pre > 0:
                    row["pct_change"] = (opm_post - opm_pre) / opm_pre * 100
                else:
                    row["pct_change"] = np.nan

                rows.append(row)

        # -- Write per-language CSV --
        lang_df = pd.DataFrame(rows)
        lang_df.to_csv(args.out_dir / f"opm_{lang}.csv", index=False)

        # -- Aggregate for summary --
        ai_pct = lang_df.loc[lang_df["seed_type"] == "ai", "pct_change"].dropna()
        bl_pct = lang_df.loc[lang_df["seed_type"] == "baseline", "pct_change"].dropna()

        summary_rows.append({
            "lang": lang,
            "n_ai_words": int((lang_df["seed_type"] == "ai").sum()),
            "n_baseline_words": int((lang_df["seed_type"] == "baseline").sum()),
            "mean_pct_change_ai": ai_pct.mean() if len(ai_pct) else np.nan,
            "median_pct_change_ai": ai_pct.median() if len(ai_pct) else np.nan,
            "mean_pct_change_baseline": bl_pct.mean() if len(bl_pct) else np.nan,
            "median_pct_change_baseline": bl_pct.median() if len(bl_pct) else np.nan,
        })

    # -- Write summary CSV --
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_dir / "diachronic_summary.csv", index=False)

    # -- Write summary.txt --
    lines = []
    lines.append("=" * 70)
    lines.append("DIACHRONIC OPM ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Languages analysed: {len(summary_df)}")
    lines.append(f"Years: {YEARS}")
    lines.append(f"Seeds per language: {args.n_seeds} AI + UPOS-matched baseline")
    lines.append(f"Pre period: mean(2020, 2021)  |  Post period: mean(2023, 2024)")
    lines.append("")

    lines.append("--- Aggregate across languages ---")
    lines.append("")

    mean_ai = summary_df["mean_pct_change_ai"].mean()
    median_ai = summary_df["median_pct_change_ai"].median()
    mean_bl = summary_df["mean_pct_change_baseline"].mean()
    median_bl = summary_df["median_pct_change_baseline"].median()

    lines.append(f"  AI words:       mean-of-means = {mean_ai:+.1f}%,  median-of-medians = {median_ai:+.1f}%")
    lines.append(f"  Baseline words: mean-of-means = {mean_bl:+.1f}%,  median-of-medians = {median_bl:+.1f}%")
    lines.append("")

    n_ai_higher = int((summary_df["mean_pct_change_ai"] > summary_df["mean_pct_change_baseline"]).sum())
    lines.append(f"  Languages where AI mean > baseline mean: {n_ai_higher}/{len(summary_df)}")
    lines.append("")

    lines.append("--- Per-language summary ---")
    lines.append("")
    lines.append(f"  {'lang':<6} {'AI mean%':>10} {'AI med%':>10} {'BL mean%':>10} {'BL med%':>10}")
    lines.append(f"  {'----':<6} {'--------':>10} {'-------':>10} {'--------':>10} {'-------':>10}")
    for _, r in summary_df.iterrows():
        lines.append(
            f"  {r['lang']:<6} {r['mean_pct_change_ai']:>+10.1f} {r['median_pct_change_ai']:>+10.1f} "
            f"{r['mean_pct_change_baseline']:>+10.1f} {r['median_pct_change_baseline']:>+10.1f}"
        )
    lines.append("")
    lines.append("=" * 70)

    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    print(f"\nResults written to {args.out_dir}/", file=sys.stderr)
    print(f"  AI words:       mean-of-means = {mean_ai:+.1f}%", file=sys.stderr)
    print(f"  Baseline words: mean-of-means = {mean_bl:+.1f}%", file=sys.stderr)
    print(f"  AI > baseline in {n_ai_higher}/{len(summary_df)} languages", file=sys.stderr)


if __name__ == "__main__":
    main()
