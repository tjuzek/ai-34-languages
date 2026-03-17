#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Calculate "Market Share" shift of AI-overused words vs. a Baseline.

This script:
1. Identifies the Top-N AI-overused words (by LAS or by LPR).
2. Identifies a Baseline set of N neutral words (metric ~ 0) with similar frequency.
   - Strategy: Selects the top 2*N most frequent neutral words,
     and randomly samples N from that pool (no trimming).
3. Calculates the aggregate market share (sum of relative frequencies) for both groups
   in Pre-LLM vs Post-LLM WMT data (years configurable).
4. Outputs a summary TSV.

NEW:
- --top_metric {las,lpr}
  * las: Top-N by rank_LAS (fallback LAS desc)
  * lpr: Top-N by lpr_MH desc (fallback rank_LPR if desired), default filtered to lpr_guard_ok==1
- --lpr_ignore_guard to disable default guard filtering in LPR mode.

Usage (LAS):
python scripts/calculate_marketshare_shift.py \
  --ai_input_dir out/ig/las-gpt4.1-mini-all/policy=quantile_trim=no_M=1_K=12_jitter=no \
  --wmt_counts_dir data/ig/wmt-counts \
  --output_dir out/ig/marketshare_analysis-all-10 \
  --top_n 10 \
  --baseline_metric avg_wmt \
  --seed 42 \
  --pre_years 2020 2021 \
  --post_years 2023 2024 \
  --top_metric las

Usage (LPR):
python scripts/calculate_marketshare_shift.py \
  --ai_input_dir out/ig/las-gpt4.1-mini-content-only-lpr/policy=quantile_trim=no_M=1_K=12_jitter=no \
  --wmt_counts_dir data/ig/wmt-counts \
  --output_dir out/ig/marketshare_analysis-content-lpr-20-2023 \
  --top_n 20 \
  --baseline_metric avg_wmt \
  --seed 42 \
  --pre_years 2020 2021 \
  --post_years 2023 \
  --top_metric lpr
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import scipy.stats as stats


def load_wmt_counts_for_period(lang_dir: Path, years):
    """
    Loads and aggregates counts for a specific set of years.
    Returns:
       - combined_counts (dict): {lemma_UPOS: total_count}
       - total_tokens (int): Sum of all tokens in these years
    """
    combined_counts = {}
    total_tokens = 0

    for year in years:
        filename = f"{year}_counts.tsv"
        filepath = lang_dir / filename

        if not filepath.exists():
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 2:
                        continue
                    key = parts[0]
                    try:
                        count = int(parts[1])
                    except ValueError:
                        continue

                    combined_counts[key] = combined_counts.get(key, 0) + count
                    total_tokens += count
        except Exception as e:
            print(f"   [ERROR] Reading {filename}: {e}")

    return combined_counts, total_tokens


def _ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def get_baseline_words(
    df: pd.DataFrame,
    n: int,
    seed: int,
    baseline_metric: str = "c_H",
    neutral_metric: str = "las",
):
    """
    Selects N baseline words that are:
    1. Neutral w.r.t. the chosen neutral metric (LAS ~ 0 or LPR ~ 0)
    2. Sampled from high frequency words.
       Logic: Take top 2*N freq, sample N.

    neutral_metric:
      - 'las': use abs(LAS) closeness to 0
      - 'lpr': use abs(lpr_MH) closeness to 0
    """
    df_neutral = df.copy()

    # Choose neutrality measure
    if neutral_metric == "las":
        if "LAS" not in df_neutral.columns:
            return []
        df_neutral["abs_neutral"] = _ensure_numeric(df_neutral["LAS"]).abs()
    elif neutral_metric == "lpr":
        if "lpr_MH" not in df_neutral.columns:
            return []
        df_neutral["abs_neutral"] = _ensure_numeric(df_neutral["lpr_MH"]).abs()
    else:
        raise ValueError("neutral_metric must be 'las' or 'lpr'")

    # Take the 2000 closest-to-neutral candidates
    candidates = df_neutral.sort_values("abs_neutral", ascending=True).head(2000)

    # Decide which frequency proxy to sort by (baseline_metric)
    if baseline_metric == "avg_wmt":
        if "wmt_rel_pre" in candidates.columns and "wmt_rel_post" in candidates.columns:
            candidates["wmt_avg"] = (
                _ensure_numeric(candidates["wmt_rel_pre"]) + _ensure_numeric(candidates["wmt_rel_post"])
            ) / 2.0
            sort_col = "wmt_avg"
        else:
            sort_col = "c_H" if "c_H" in candidates.columns else None
    else:
        sort_col = "c_H" if "c_H" in candidates.columns else None

    if sort_col is None:
        # If we can't compute any reasonable frequency proxy, return empty baseline
        return []

    candidates[sort_col] = _ensure_numeric(candidates[sort_col])

    pool_size = n * 2
    pool = candidates.sort_values(sort_col, ascending=False).head(pool_size)

    if len(pool) < n:
        print(f"   [WARN] Not enough neutral candidates found ({len(pool)}). Using all.")
        return pool["key"].astype(str).tolist()

    baseline_words = pool.sample(n=n, random_state=seed)
    return baseline_words["key"].astype(str).tolist()


def select_top_ai_words(
    df: pd.DataFrame,
    top_n: int,
    top_metric: str,
    lpr_ignore_guard: bool,
):
    """
    Returns (ai_words, note_dict) where ai_words is a list[str] and note_dict has diagnostics.
    """
    note = {
        "top_metric": top_metric,
        "lpr_guard_applied": False,
        "n_rows_before": int(len(df)),
        "n_rows_after_filter": int(len(df)),
        "selection_method": "",
    }

    if top_metric == "las":
        if "rank_LAS" in df.columns:
            # rank_LAS: 1 = most overused
            d2 = df.copy()
            d2["rank_LAS"] = pd.to_numeric(d2["rank_LAS"], errors="coerce").fillna(1e18)
            ai_words = d2.sort_values("rank_LAS", ascending=True).head(top_n)["key"].astype(str).tolist()
            note["selection_method"] = "rank_LAS_asc"
            return ai_words, note

        if "LAS" in df.columns:
            d2 = df.copy()
            d2["LAS"] = _ensure_numeric(d2["LAS"])
            ai_words = d2.sort_values("LAS", ascending=False).head(top_n)["key"].astype(str).tolist()
            note["selection_method"] = "LAS_desc"
            return ai_words, note

        return [], {**note, "selection_method": "missing_rank_LAS_and_LAS"}

    if top_metric == "lpr":
        d2 = df.copy()
        if (not lpr_ignore_guard) and ("lpr_guard_ok" in d2.columns):
            d2["lpr_guard_ok"] = pd.to_numeric(d2["lpr_guard_ok"], errors="coerce").fillna(0).astype(int)
            before = len(d2)
            d2 = d2[d2["lpr_guard_ok"] == 1]
            note["lpr_guard_applied"] = True
            note["n_rows_after_filter"] = int(len(d2))
            note["n_rows_before"] = int(before)
        elif not lpr_ignore_guard:
            # Guard requested but missing column
            note["lpr_guard_applied"] = False

        if "lpr_MH" in d2.columns:
            d2["lpr_MH"] = _ensure_numeric(d2["lpr_MH"])
            ai_words = d2.sort_values("lpr_MH", ascending=False).head(top_n)["key"].astype(str).tolist()
            note["selection_method"] = "lpr_MH_desc"
            return ai_words, note

        if "rank_LPR" in d2.columns:
            d2["rank_LPR"] = pd.to_numeric(d2["rank_LPR"], errors="coerce").fillna(1e18)
            ai_words = d2.sort_values("rank_LPR", ascending=True).head(top_n)["key"].astype(str).tolist()
            note["selection_method"] = "rank_LPR_asc"
            return ai_words, note

        return [], {**note, "selection_method": "missing_lpr_MH_and_rank_LPR"}

    raise ValueError("top_metric must be 'las' or 'lpr'")


def process_language(
    lang_code: str,
    ai_file: Path,
    wmt_lang_dir: Path,
    top_n: int,
    baseline_metric: str,
    seed: int,
    pre_years,
    post_years,
    top_metric: str,
    lpr_ignore_guard: bool,
):
    """
    Process a single language. Returns a dict of results.
    """
    # 1. Load CSV
    try:
        df = pd.read_csv(ai_file)
    except Exception as e:
        print(f"[{lang_code}] Error reading CSV: {e}")
        return None

    if "key" not in df.columns:
        print(f"[{lang_code}] No 'key' column found in {ai_file.name}")
        return None

    # Normalise Russian ё → е (Stanza may produce ё in lemmas but WMT POS files use е)
    df["key"] = df["key"].str.replace("ё", "е", regex=False)
    if "lemma" in df.columns:
        df["lemma"] = df["lemma"].str.replace("ё", "е", regex=False)

    # 2. Select AI group by chosen metric
    ai_words, note = select_top_ai_words(df, top_n, top_metric, lpr_ignore_guard)
    if not ai_words:
        print(f"[{lang_code}] No AI words selected (metric={top_metric}).")
        return None

    # 3. Baseline group (neutral relative to same metric family)
    neutral_metric = "las" if top_metric == "las" else "lpr"
    baseline_words = get_baseline_words(df, top_n, seed, baseline_metric=baseline_metric, neutral_metric=neutral_metric)
    if len(baseline_words) < top_n:
        print(f"[{lang_code}] Warning: Only found {len(baseline_words)} baseline words.")

    # 4. Load WMT counts (Pre vs Post)
    counts_pre, total_pre = load_wmt_counts_for_period(wmt_lang_dir, pre_years)
    counts_post, total_post = load_wmt_counts_for_period(wmt_lang_dir, post_years)

    if total_pre == 0 or total_post == 0:
        print(f"[{lang_code}] Skipping (WMT data missing)")
        return None

    # 5. Aggregate market shares
    def get_group_stats(word_list, counts, total):
        agg_count = 0
        for w in word_list:
            agg_count += counts.get(w, 0)
        return agg_count, (agg_count / total) if total > 0 else 0.0

    ai_count_pre, ai_share_pre = get_group_stats(ai_words, counts_pre, total_pre)
    ai_count_post, ai_share_post = get_group_stats(ai_words, counts_post, total_post)

    base_count_pre, base_share_pre = get_group_stats(baseline_words, counts_pre, total_pre)
    base_count_post, base_share_post = get_group_stats(baseline_words, counts_post, total_post)

    # 6. Significance test (Chi-square) for AI group shift
    table = [
        [ai_count_pre, total_pre - ai_count_pre],
        [ai_count_post, total_post - ai_count_post],
    ]
    try:
        chi2, p_val, _, _ = stats.chi2_contingency(table)
        is_sig = p_val < 0.001
    except Exception:
        p_val = 1.0
        is_sig = False

    # 7. Growth rates
    ai_growth = (ai_share_post - ai_share_pre) / ai_share_pre if ai_share_pre > 0 else 0.0
    base_growth = (base_share_post - base_share_pre) / base_share_pre if base_share_pre > 0 else 0.0

    return {
        "language": lang_code,
        "top_metric": top_metric,
        "ai_share_pre": ai_share_pre,
        "ai_share_post": ai_share_post,
        "ai_growth_pct": ai_growth,
        "ai_p_value": p_val,
        "ai_is_sig": is_sig,
        "base_share_pre": base_share_pre,
        "base_share_post": base_share_post,
        "base_growth_pct": base_growth,
        "top_n_requested": top_n,
        "top_n_ai_used": len(ai_words),
        "n_baseline_used": len(baseline_words),
        "ai_selection_method": note.get("selection_method", ""),
        "lpr_guard_applied": bool(note.get("lpr_guard_applied", False)),
        "n_rows_before_filter": int(note.get("n_rows_before", len(df))),
        "n_rows_after_filter": int(note.get("n_rows_after_filter", len(df))),
        "ai_file": str(ai_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate Market Share Shift (AI vs Baseline)")

    parser.add_argument("--ai_input_dir", type=str, required=True, help="Base dir with language folders containing CSVs")
    parser.add_argument("--wmt_counts_dir", type=str, required=True, help="Base dir containing WMT count folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir to save summary TSV")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top AI words to analyse")

    parser.add_argument(
        "--top_metric",
        type=str,
        default="las",
        choices=["las", "lpr"],
        help="How to select Top-N AI words: 'las' (rank_LAS/LAS) or 'lpr' (lpr_MH/rank_LPR). Default: las",
    )
    parser.add_argument(
        "--lpr_ignore_guard",
        action="store_true",
        help="(top_metric=lpr) Ignore lpr_guard_ok==1 filter and allow all rows for Top-N selection.",
    )

    parser.add_argument(
        "--baseline_metric",
        type=str,
        default="avg_wmt",
        choices=["c_H", "avg_wmt"],
        help="Metric to sort baseline words by frequency (c_H or avg_wmt)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for baseline sampling")

    parser.add_argument(
        "--pre_years",
        nargs="+",
        default=["2020", "2021"],
        help="Years to include in Pre-LLM period (default: 2020 2021)",
    )
    parser.add_argument(
        "--post_years",
        nargs="+",
        default=["2023", "2024"],
        help="Years to include in Post-LLM period (default: 2023 2024)",
    )

    args = parser.parse_args()

    ai_base = Path(args.ai_input_dir)
    wmt_base = Path(args.wmt_counts_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    lang_dirs = [d for d in ai_base.iterdir() if d.is_dir()]
    lang_dirs.sort()

    print(f"Scanning {len(lang_dirs)} languages...")
    print(f"Analysis Config: Pre={args.pre_years}, Post={args.post_years}, top_metric={args.top_metric}")

    for lang_dir in lang_dirs:
        lang_code = lang_dir.name
        wmt_lang_dir = wmt_base / lang_code

        # Prefer standard LAS/LPR table(s)
        csv_files = list(lang_dir.glob("las_word_*.csv"))
        if not csv_files:
            # Backwards compatibility: old annotated naming
            csv_files = list(lang_dir.glob("*_annotated.csv"))

        if not csv_files or not wmt_lang_dir.exists():
            continue

        csv_files.sort()
        ai_file = csv_files[0]

        print(f"Processing {lang_code} ({ai_file.name})...")
        res = process_language(
            lang_code=lang_code,
            ai_file=ai_file,
            wmt_lang_dir=wmt_lang_dir,
            top_n=args.top_n,
            baseline_metric=args.baseline_metric,
            seed=args.seed,
            pre_years=args.pre_years,
            post_years=args.post_years,
            top_metric=args.top_metric,
            lpr_ignore_guard=args.lpr_ignore_guard,
        )
        if res:
            results.append(res)

    if results:
        df_res = pd.DataFrame(results)

        cols = [
            "language",
            "top_metric",
            "ai_share_pre",
            "ai_share_post",
            "ai_growth_pct",
            "ai_is_sig",
            "ai_p_value",
            "base_share_pre",
            "base_share_post",
            "base_growth_pct",
            "top_n_requested",
            "top_n_ai_used",
            "n_baseline_used",
            "ai_selection_method",
            "lpr_guard_applied",
            "n_rows_before_filter",
            "n_rows_after_filter",
            "ai_file",
        ]
        cols = [c for c in cols if c in df_res.columns]
        df_res = df_res[cols]

        out_path = out_dir / "marketshare_summary.tsv"
        df_res.to_csv(out_path, sep="\t", index=False)
        print(f"\nSuccess! Summary saved to: {out_path}")
        print(df_res[["language", "top_metric", "ai_growth_pct", "base_growth_pct", "ai_is_sig"]].head())
    else:
        print("No valid data processed.")


if __name__ == "__main__":
    main()
