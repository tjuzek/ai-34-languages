#!/usr/bin/env python
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Compare relative-frequency profiles (human & model) between two LAS CSV files.

The CSVs must have at least the columns: key, c_H, c_M

Example:
    python tools/compare_relfreq.py out/las-gpt4.1-mini-window-sizes/policy=quantile_trim=no_M=1_K=10_jitter=no/en/las_word_en.csv out/las-gpt4.1-mini-window-sizes/policy=quantile_trim=no_M=1_K=10_jitter=no/en/las_word_en.csv \
        --min-count 20

This will print lines like:
    relfreq_M (count-based) ~ relfreq_M (count-based), min_count=20: n=9673, Pearson r=0.9856
    relfreq_H (count-based) ~ relfreq_H (count-based), min_count=20: n=13204, Pearson r=0.9920
"""

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class CorrResult:
    name: str
    n: int
    r: float


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation for finite pairs in x, y.
    Returns np.nan if variance is zero or no valid pairs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    denom = np.sqrt((x_centered ** 2).sum()) * np.sqrt((y_centered ** 2).sum())
    if denom == 0.0:
        return float("nan")
    return float((x_centered * y_centered).sum() / denom)


def compute_relfreq_correlations(
    merged: pd.DataFrame,
    min_count: int,
) -> Tuple[CorrResult, CorrResult]:
    """
    Compute Pearson correlations of relative frequencies from counts for
    model (c_M) and human (c_H).

    merged: DataFrame with columns c_M_1, c_M_2, c_H_1, c_H_2
    min_count: minimum total count (file1 + file2) to include an item
    """

    # --- Model side ---
    mask_M = (merged["c_M_1"] + merged["c_M_2"]) >= min_count

    total_c_M_1 = merged["c_M_1"].sum()
    total_c_M_2 = merged["c_M_2"].sum()

    freq_M_1 = merged["c_M_1"] / total_c_M_1 if total_c_M_1 > 0 else np.nan
    freq_M_2 = merged["c_M_2"] / total_c_M_2 if total_c_M_2 > 0 else np.nan

    r_M = pearson_corr(freq_M_1[mask_M], freq_M_2[mask_M])
    res_M = CorrResult(
        name=f"relfreq_M (count-based) ~ relfreq_M (count-based), min_count={min_count}",
        n=int(mask_M.sum()),
        r=r_M,
    )

    # --- Human side ---
    mask_H = (merged["c_H_1"] + merged["c_H_2"]) >= min_count

    total_c_H_1 = merged["c_H_1"].sum()
    total_c_H_2 = merged["c_H_2"].sum()

    freq_H_1 = merged["c_H_1"] / total_c_H_1 if total_c_H_1 > 0 else np.nan
    freq_H_2 = merged["c_H_2"] / total_c_H_2 if total_c_H_2 > 0 else np.nan

    r_H = pearson_corr(freq_H_1[mask_H], freq_H_2[mask_H])
    res_H = CorrResult(
        name=f"relfreq_H (count-based) ~ relfreq_H (count-based), min_count={min_count}",
        n=int(mask_H.sum()),
        r=r_H,
    )

    return res_M, res_H


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare relative-frequency profiles (human & model) between two LAS CSV files."
    )
    parser.add_argument("csv1", help="First LAS CSV file (e.g. las_word_en_15.csv)")
    parser.add_argument("csv2", help="Second LAS CSV file (e.g. las_word_en_35.csv)")
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Minimum total count (file1 + file2) to include in correlations "
             "(default: 20).",
    )
    args = parser.parse_args()

    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    required_cols = {"key", "c_H", "c_M"}
    missing1 = required_cols - set(df1.columns)
    missing2 = required_cols - set(df2.columns)
    if missing1:
        raise ValueError(f"{args.csv1} is missing columns: {missing1}")
    if missing2:
        raise ValueError(f"{args.csv2} is missing columns: {missing2}")

    merged = pd.merge(df1, df2, on="key", suffixes=("_1", "_2"), how="inner")

    res_M, res_H = compute_relfreq_correlations(merged, min_count=args.min_count)

    # Only output the two correlation lines, in the format you liked
    print(f"{res_M.name}: n={res_M.n}, Pearson r={res_M.r:.4f}")
    print(f"{res_H.name}: n={res_H.n}, Pearson r={res_H.r:.4f}")


if __name__ == "__main__":
    main()

