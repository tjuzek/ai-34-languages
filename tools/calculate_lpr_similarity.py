#!/usr/bin/env python
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Compare LPR profiles between two LAS CSV files.

Focus: lpr_MH (log prevalence ratio, model vs human).
Key idea: treat guarded-out LPR values as missing (NaN), not as zeros,
so correlation is not artificially inflated by shared forced zeros.

Writes a timestamped .txt report to --out_dir.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    name: str
    n: int
    value: float


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
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
    denom = np.sqrt((x_centered**2).sum()) * np.sqrt((y_centered**2).sum())
    if denom == 0.0:
        return float("nan")
    return float((x_centered * y_centered).sum() / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman = Pearson correlation of ranks.
    Uses average ranks for ties (pandas rank default is fine here).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    xs = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    ys = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    return pearson_corr(xs, ys)


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find a column in df matching any candidate (case-insensitive).
    Returns the actual df column name or None.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _standardize_columns(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """
    Ensure df has:
      - key
      - lpr_MH
      - lpr_guard_ok
    by renaming from plausible alternatives if needed.
    """
    if "key" not in df.columns:
        raise ValueError(f"[{filepath}] Missing required column: 'key'")

    lpr_col = _find_col(df, ["lpr_MH", "lpr_mh", "lpr"])
    if lpr_col is None:
        raise ValueError(f"[{filepath}] Missing required LPR column (expected 'lpr_MH').")

    guard_col = _find_col(df, ["lpr_guard_ok", "lpr_guard_ok_mh", "guard_ok", "lpr_ok"])
    if guard_col is None:
        raise ValueError(f"[{filepath}] Missing required guard column (expected 'lpr_guard_ok').")

    out = df.copy()
    if lpr_col != "lpr_MH":
        out = out.rename(columns={lpr_col: "lpr_MH"})
    if guard_col != "lpr_guard_ok":
        out = out.rename(columns={guard_col: "lpr_guard_ok"})
    return out


def compute_lpr_similarity(
    merged: pd.DataFrame,
    *,
    require_guard_both: bool = True,
    min_abs_lpr: float = 0.0,
    topk_abs_lpr: int = 0,
    clip_abs_lpr: Optional[float] = None,
) -> Tuple[List[SimResult], dict]:
    """
    Returns:
      - list of similarity results (Pearson, Spearman, MAD, MAE, RMSE)
      - stats dict (n_total, n_guard_both, coverage, etc.)
    """
    lpr1 = merged["lpr_MH_1"].astype(float)
    lpr2 = merged["lpr_MH_2"].astype(float)

    g1 = merged["lpr_guard_ok_1"]
    g2 = merged["lpr_guard_ok_2"]

    # treat any positive/nonzero as "ok"
    g1_ok = pd.to_numeric(g1, errors="coerce").fillna(0).astype(int) > 0
    g2_ok = pd.to_numeric(g2, errors="coerce").fillna(0).astype(int) > 0

    if require_guard_both:
        ok = g1_ok & g2_ok
    else:
        ok = g1_ok | g2_ok

    # IMPORTANT: guarded-out values become NaN (not 0)
    lpr1 = lpr1.where(ok, np.nan)
    lpr2 = lpr2.where(ok, np.nan)

    # Optional clipping to reduce impact of extreme ratios (rare but possible)
    if clip_abs_lpr is not None and clip_abs_lpr > 0:
        lpr1 = lpr1.clip(lower=-clip_abs_lpr, upper=clip_abs_lpr)
        lpr2 = lpr2.clip(lower=-clip_abs_lpr, upper=clip_abs_lpr)

    mask = np.isfinite(lpr1.to_numpy()) & np.isfinite(lpr2.to_numpy())

    # Optional: focus on “signal” region rather than huge pile near 0
    if min_abs_lpr > 0:
        a = np.abs(lpr1.to_numpy())
        b = np.abs(lpr2.to_numpy())
        mask = mask & (np.maximum(a, b) >= min_abs_lpr)

    # Optional: restrict to top-K by max(|lpr|) across the two files
    if topk_abs_lpr and topk_abs_lpr > 0:
        idx = np.where(mask)[0]
        if idx.size > 0:
            a = np.abs(lpr1.to_numpy()[idx])
            b = np.abs(lpr2.to_numpy()[idx])
            score = np.maximum(a, b)
            order = np.argsort(-score)  # descending
            keep = idx[order[: min(topk_abs_lpr, order.size)]]
            new_mask = np.zeros_like(mask, dtype=bool)
            new_mask[keep] = True
            mask = new_mask

    x = lpr1.to_numpy()[mask]
    y = lpr2.to_numpy()[mask]
    n = int(mask.sum())

    results: List[SimResult] = []
    results.append(SimResult(name="Pearson r (lpr_MH)", n=n, value=pearson_corr(x, y)))
    results.append(SimResult(name="Spearman ρ (lpr_MH)", n=n, value=spearman_corr(x, y)))

    if n >= 1:
        diffs = x - y
        results.append(SimResult(name="Median |Δlpr| (MAD)", n=n, value=float(np.median(np.abs(diffs)))))
        results.append(SimResult(name="Mean |Δlpr| (MAE)", n=n, value=float(np.mean(np.abs(diffs)))))
        results.append(SimResult(name="RMSE(Δlpr)", n=n, value=float(np.sqrt(np.mean(diffs**2)))))
    else:
        results.append(SimResult(name="Median |Δlpr| (MAD)", n=n, value=float("nan")))
        results.append(SimResult(name="Mean |Δlpr| (MAE)", n=n, value=float("nan")))
        results.append(SimResult(name="RMSE(Δlpr)", n=n, value=float("nan")))

    stats = {
        "n_total_keys_inner_join": int(len(merged)),
        "n_guard_both": int((g1_ok & g2_ok).sum()),
        "coverage_guard_both": float(((g1_ok & g2_ok).sum() / len(merged)) if len(merged) else float("nan")),
        "require_guard_both": bool(require_guard_both),
        "min_abs_lpr": float(min_abs_lpr),
        "topk_abs_lpr": int(topk_abs_lpr),
        "clip_abs_lpr": None if clip_abs_lpr is None else float(clip_abs_lpr),
    }
    return results, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lpr_MH similarity between two LAS CSV files.")
    parser.add_argument("--f1", required=True, help="First LAS CSV file")
    parser.add_argument("--f2", required=True, help="Second LAS CSV file")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save output")

    # Guard handling
    parser.add_argument(
        "--guard-mode",
        choices=["both", "either"],
        default="both",
        help="Use only items where guard ok in BOTH files (both) or in EITHER file (either). Default: both.",
    )

    # Optional focusing controls
    parser.add_argument(
        "--min-abs-lpr",
        type=float,
        default=0.0,
        help="Only include items with max(|lpr1|,|lpr2|) >= threshold. Default: 0.0 (no filter).",
    )
    parser.add_argument(
        "--topk-abs-lpr",
        type=int,
        default=0,
        help="If >0, restrict to top-K items by max(|lpr1|,|lpr2|) after filtering. Default: 0 (no top-K).",
    )
    parser.add_argument(
        "--clip-abs-lpr",
        type=float,
        default=None,
        help="If set, clip lpr values to [-clip, +clip] before computing metrics (robustness to extremes).",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filepath = os.path.join(args.out_dir, f"lpr_similarity_{timestamp}.txt")

    output_lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        output_lines.append(msg)

    log("--- LPR Similarity Analysis ---")
    log(f"File 1: {args.f1}")
    log(f"File 2: {args.f2}")
    log(f"Guard mode: {args.guard_mode}")
    log(f"min_abs_lpr: {args.min_abs_lpr}")
    log(f"topk_abs_lpr: {args.topk_abs_lpr}")
    log(f"clip_abs_lpr: {args.clip_abs_lpr}")
    log("-" * 30)

    try:
        df1 = pd.read_csv(args.f1)
        df2 = pd.read_csv(args.f2)
        df1 = _standardize_columns(df1, args.f1)
        df2 = _standardize_columns(df2, args.f2)
    except Exception as e:
        log(f"Error loading/validating files: {e}")
        sys.exit(1)

    merged = pd.merge(df1, df2, on="key", suffixes=("_1", "_2"), how="inner")

    require_guard_both = args.guard_mode == "both"
    results, stats = compute_lpr_similarity(
        merged,
        require_guard_both=require_guard_both,
        min_abs_lpr=args.min_abs_lpr,
        topk_abs_lpr=args.topk_abs_lpr,
        clip_abs_lpr=args.clip_abs_lpr,
    )

    log(f"Inner-join keys: {stats['n_total_keys_inner_join']}")
    log(f"Guard ok in BOTH: {stats['n_guard_both']} (coverage={stats['coverage_guard_both']:.4f})")
    log("")

    for res in results:
        if np.isfinite(res.value):
            log(f"{res.name}: n={res.n}, value={res.value:.6f}")
        else:
            log(f"{res.name}: n={res.n}, value=nan")

    log("\nAnalysis Complete.")

    with open(out_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[INFO] Saved to: {out_filepath}")


if __name__ == "__main__":
    main()