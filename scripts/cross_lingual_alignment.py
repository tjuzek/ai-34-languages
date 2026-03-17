#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""Cross-lingual alignment quantification.

Measures how well AI-overused English words align with AI-overused words
in target languages, using multilingual embeddings.

Input
-----
Per-language CSV files produced by the LPR pipeline, located at:
    {data-dir}/{lang}/las_word_{lang}.csv

Each CSV has columns including: lemma, upos, lpr_MH_guarded, lpr_guard_ok.

Output
------
Written to {out-dir}/results/:
    match_details.csv   – one row per seed × target language
    seed_summary.csv    – one row per seed with cross-lingual hit counts
    summary.txt         – human-readable aggregate for the paper

Embeddings are cached to {out-dir}/embeddings/{lang}.npy so that
re-runs skip the (slow) encoding step.

Two matching modes:
    restricted   – for each seed, search only among target top-K words;
                   primary metric is mean-max-cosine (floor-free);
                   also reports binary hit rates at --cosine-floor (default)
    unrestricted – search the full target vocabulary for the closest word,
                   then check whether that word falls inside top-K

Example
-------
    python scripts/cross_lingual_alignment.py \\
        --data-dir out/ig/las-gpt4.1-mini-content-only-lpr/policy=quantile_trim=no_M=1_K=12_jitter=no \\
        --out-dir out/ig/cross_lingual_alignment \\
        --n-seeds 20 --k-values 50 200 --n-permutations 1000 --seed 42
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
EXCLUDE_LANGS = {"en", "hu"}  # en = source; hu = excluded per spec
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_language_csv(csv_path: Path) -> pd.DataFrame:
    """Load a single language CSV and return a DataFrame."""
    return pd.read_csv(csv_path)


def load_english_seeds(
    data_dir: Path, n_seeds: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ai_seeds, baseline_seeds) DataFrames from English data.

    AI seeds: top-n by lpr_MH_guarded descending, lpr_guard_ok=1,
              content words only.
    Baseline seeds: n words with smallest |LAS|, UPOS-matched to AI seeds,
                    tie-broken by highest model count (c_M) so that the
                    baseline consists of frequent but non-shifting words.
    """
    en_path = data_dir / "en" / "las_word_en.csv"
    df = load_language_csv(en_path)

    # Filter to guarded content words
    content = df[(df["lpr_guard_ok"] == 1) & df["upos"].isin(CONTENT_UPOS)].copy()

    # AI seeds: highest lpr_MH_guarded
    ai_seeds = (
        content.sort_values("lpr_MH_guarded", ascending=False)
        .head(n_seeds)
        .reset_index(drop=True)
    )

    # Target UPOS distribution of AI seeds
    upos_counts = Counter(ai_seeds["upos"])

    # Baseline seeds: smallest |LAS|, tie-broken by highest c_M,
    # UPOS-matched to AI seeds
    ai_keys = set(ai_seeds["key"])
    candidates = content[~content["key"].isin(ai_keys)].copy()
    candidates["abs_las"] = candidates["LAS"].abs()
    candidates = candidates.sort_values(["abs_las", "c_M"], ascending=[True, False])

    baseline_parts = []
    for upos, count in upos_counts.items():
        upos_cands = candidates[candidates["upos"] == upos].head(count)
        baseline_parts.append(upos_cands)

    baseline_seeds = pd.concat(baseline_parts).reset_index(drop=True)

    return ai_seeds, baseline_seeds


def load_target_languages(
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load all target-language CSVs (excluding en, hu).

    Returns {lang_code: DataFrame} with an added 'lpr_rank' column
    (1-indexed, descending by lpr_MH_guarded).
    """
    targets = {}
    for lang_dir in sorted(data_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        if lang in EXCLUDE_LANGS:
            continue
        csv_path = lang_dir / f"las_word_{lang}.csv"
        if not csv_path.exists():
            continue
        df = load_language_csv(csv_path)
        # Keep only guarded rows
        df = df[df["lpr_guard_ok"] == 1].copy()
        # Assign LPR rank (1-indexed, descending by lpr_MH_guarded)
        df = df.sort_values("lpr_MH_guarded", ascending=False).reset_index(drop=True)
        df["lpr_rank"] = df.index + 1
        targets[lang] = df
    return targets


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embeddings(
    lemmas: list[str],
    model,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Embed lemmas using the sentence-transformers model.

    If cache_path exists and row count matches, load from cache.
    Otherwise encode and save.
    """
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(lemmas):
            return cached

    embeddings = model.encode(lemmas, show_progress_bar=False, convert_to_numpy=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)

    return embeddings


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of a and each row of b."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def match_seeds_to_language_restricted(
    seed_embs: np.ndarray,
    seed_df: pd.DataFrame,
    target_embs: np.ndarray,
    target_df: pd.DataFrame,
    lang: str,
    cosine_floor: float,
    k_values: list[int],
    seed_type: str,
) -> list[dict]:
    """Restricted matching: for each seed, search only among top-K words.

    For each K, find the best cosine match among target words with
    lpr_rank <= K. Reports both the max cosine and a binary hit (>= floor).
    Match details report the match at the largest K.
    """
    k_max = max(k_values)
    # Embeddings and df are sorted by lpr_rank (= row index + 1),
    # so top-K words are rows 0..K-1.
    n_avail = min(k_max, target_embs.shape[0])
    sim_topk = cosine_similarity_matrix(seed_embs, target_embs[:n_avail])

    rows = []
    for i in range(len(seed_df)):
        sims_i = sim_topk[i]  # shape (n_avail,)
        best_j = int(sims_i.argmax())
        best_cos = float(sims_i[best_j])

        trow = target_df.iloc[best_j]
        row = {
            "seed_type": seed_type,
            "en_lemma": seed_df.iloc[i]["lemma"],
            "en_upos": seed_df.iloc[i]["upos"],
            "en_lpr": seed_df.iloc[i]["lpr_MH_guarded"],
            "lang": lang,
            "matched_lemma": trow["lemma"],
            "matched_key": trow["key"],
            "cosine": best_cos,
            "lpr_rank": int(trow["lpr_rank"]),
            "lpr_value": trow["lpr_MH_guarded"],
        }
        # For each K, find best match within that slice
        for k in k_values:
            k_eff = min(k, n_avail)
            sub_best = float(sims_i[:k_eff].max())
            row[f"max_cos_{k}"] = sub_best
            row[f"hit_at_{k}"] = int(sub_best >= cosine_floor)

        rows.append(row)
    return rows


def match_seeds_to_language_unrestricted(
    seed_embs: np.ndarray,
    seed_df: pd.DataFrame,
    target_embs: np.ndarray,
    target_df: pd.DataFrame,
    lang: str,
    cosine_floor: float,
    k_values: list[int],
    seed_type: str,
) -> list[dict]:
    """Unrestricted matching: best cosine match in full vocabulary.

    A hit is counted when the globally best match has lpr_rank <= K.
    """
    sim = cosine_similarity_matrix(seed_embs, target_embs)
    best_idx = sim.argmax(axis=1)
    best_cos = sim[np.arange(len(best_idx)), best_idx]

    rows = []
    for i in range(len(seed_df)):
        cos = float(best_cos[i])
        row = {
            "seed_type": seed_type,
            "en_lemma": seed_df.iloc[i]["lemma"],
            "en_upos": seed_df.iloc[i]["upos"],
            "en_lpr": seed_df.iloc[i]["lpr_MH_guarded"],
            "lang": lang,
        }
        if cos < cosine_floor:
            row["matched_lemma"] = ""
            row["matched_key"] = ""
            row["cosine"] = cos
            row["lpr_rank"] = np.nan
            row["lpr_value"] = np.nan
            for k in k_values:
                row[f"max_cos_{k}"] = cos
                row[f"hit_at_{k}"] = 0
        else:
            j = int(best_idx[i])
            trow = target_df.iloc[j]
            row["matched_lemma"] = trow["lemma"]
            row["matched_key"] = trow["key"]
            row["cosine"] = cos
            row["lpr_rank"] = int(trow["lpr_rank"])
            row["lpr_value"] = trow["lpr_MH_guarded"]
            for k in k_values:
                row[f"max_cos_{k}"] = cos
                row[f"hit_at_{k}"] = int(trow["lpr_rank"] <= k)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def run_permutation_test(
    en_df: pd.DataFrame,
    ai_upos_counts: Counter,
    target_langs: dict[str, pd.DataFrame],
    target_embeddings: dict[str, np.ndarray],
    model,
    embed_cache_dir: Path,
    cosine_floor: float,
    k_values: list[int],
    n_perms: int,
    rng: np.random.Generator,
    match_mode: str,
) -> dict[str, list[float]]:
    """Run permutation null: sample random English content words UPOS-matched.

    Returns dict with keys like "hit_at_{k}" and "mean_cos_{k}",
    each mapping to a list of per-permutation values.
    """
    content = en_df[
        (en_df["lpr_guard_ok"] == 1) & en_df["upos"].isin(CONTENT_UPOS)
    ].copy()

    upos_groups = {
        upos: grp.index.to_numpy()
        for upos, grp in content.groupby("upos")
    }

    # Pre-embed all English content lemmas
    en_lemmas = content["lemma"].tolist()
    en_cache = embed_cache_dir / "en_content.npy"
    en_embs = get_embeddings(en_lemmas, model, en_cache)
    content_idx_to_row = {idx: row for row, idx in enumerate(content.index)}

    k_max = max(k_values)
    results = {}
    for k in k_values:
        results[f"hit_at_{k}"] = []
        results[f"mean_cos_{k}"] = []

    for perm_i in range(n_perms):
        sampled_indices = []
        for upos, count in ai_upos_counts.items():
            pool = upos_groups.get(upos, np.array([], dtype=int))
            if len(pool) < count:
                chosen = pool
            else:
                chosen = rng.choice(pool, size=count, replace=False)
            sampled_indices.extend(chosen)

        if not sampled_indices:
            for k in k_values:
                results[f"hit_at_{k}"].append(0.0)
                results[f"mean_cos_{k}"].append(0.0)
            continue

        emb_rows = [content_idx_to_row[i] for i in sampled_indices]
        perm_embs = en_embs[emb_rows]

        total_hits = {k: 0 for k in k_values}
        total_cos = {k: 0.0 for k in k_values}

        for lang in target_langs:
            t_embs = target_embeddings[lang]

            if match_mode == "restricted":
                n_avail = min(k_max, t_embs.shape[0])
                sim = cosine_similarity_matrix(perm_embs, t_embs[:n_avail])
                for i in range(len(sampled_indices)):
                    for k in k_values:
                        k_eff = min(k, n_avail)
                        best_cos = float(sim[i, :k_eff].max())
                        total_cos[k] += best_cos
                        if best_cos >= cosine_floor:
                            total_hits[k] += 1
            else:  # unrestricted
                sim = cosine_similarity_matrix(perm_embs, t_embs)
                best_idx = sim.argmax(axis=1)
                best_cos_arr = sim[np.arange(len(best_idx)), best_idx]
                for i in range(len(sampled_indices)):
                    cos = float(best_cos_arr[i])
                    for k in k_values:
                        total_cos[k] += cos
                    if cos < cosine_floor:
                        continue
                    j = int(best_idx[i])
                    rank = int(target_langs[lang].iloc[j]["lpr_rank"])
                    for k in k_values:
                        if rank <= k:
                            total_hits[k] += 1

        n_possible = len(sampled_indices) * len(target_langs)
        for k in k_values:
            results[f"hit_at_{k}"].append(
                total_hits[k] / n_possible if n_possible > 0 else 0.0
            )
            results[f"mean_cos_{k}"].append(
                total_cos[k] / n_possible if n_possible > 0 else 0.0
            )

        if (perm_i + 1) % 100 == 0:
            print(f"  Permutation {perm_i + 1}/{n_perms}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_summary(
    out_path: Path,
    ai_details: pd.DataFrame,
    baseline_details: pd.DataFrame,
    perm_results: dict[str, list[float]],
    k_values: list[int],
    target_langs: dict[str, pd.DataFrame],
    n_seeds: int,
    n_perms: int,
    match_mode: str,
    cosine_floor: float,
):
    """Write human-readable summary.txt."""
    lines = []
    lines.append("=" * 70)
    lines.append("CROSS-LINGUAL ALIGNMENT QUANTIFICATION")
    lines.append("=" * 70)
    lines.append("")

    n_langs = len(target_langs)
    lines.append(f"Seeds: {n_seeds} AI + {len(baseline_details['en_lemma'].unique())} baseline")
    lines.append(f"Target languages: {n_langs}")
    lines.append(f"Permutations: {n_perms}")
    lines.append(f"Match mode: {match_mode}")
    lines.append(f"Cosine floor: {cosine_floor}")
    lines.append("")

    for k in k_values:
        lines.append(f"--- K = {k} ---")
        lines.append("")

        # Mean max-cosine (primary metric for restricted mode)
        cos_col = f"max_cos_{k}"
        if cos_col in ai_details.columns:
            ai_mcos = ai_details[cos_col].mean()
            bl_mcos = baseline_details[cos_col].mean()
            perm_key = f"mean_cos_{k}"
            if perm_key in perm_results:
                perm_mcos = np.array(perm_results[perm_key])
                perm_mean_c = perm_mcos.mean()
                perm_std_c = perm_mcos.std()
                z_c = (ai_mcos - perm_mean_c) / perm_std_c if perm_std_c > 0 else float("inf")
                p_c = (np.sum(perm_mcos >= ai_mcos) + 1) / (len(perm_mcos) + 1)
            else:
                perm_mean_c = perm_std_c = z_c = p_c = float("nan")

            lines.append(f"  Mean max-cosine (primary metric):")
            lines.append(f"    AI seeds:            {ai_mcos:.4f}")
            lines.append(f"    Baseline seeds:      {bl_mcos:.4f}")
            lines.append(f"    Permutation null:    {perm_mean_c:.4f}  (std={perm_std_c:.4f})")
            lines.append(f"    z-score (AI vs null): {z_c:.2f}")
            lines.append(f"    Empirical p-value:   {p_c:.6f}")
            lines.append("")

        # Binary hit rate
        hit_col = f"hit_at_{k}"
        ai_rate = ai_details[hit_col].mean()
        bl_rate = baseline_details[hit_col].mean()
        perm_rates = np.array(perm_results.get(f"hit_at_{k}", [0.0]))
        perm_mean = perm_rates.mean()
        perm_std = perm_rates.std()
        z = (ai_rate - perm_mean) / perm_std if perm_std > 0 else float("inf")
        p = (np.sum(perm_rates >= ai_rate) + 1) / (len(perm_rates) + 1)

        lines.append(f"  Hit rate (cosine >= {cosine_floor}):")
        lines.append(f"    AI seeds:            {ai_rate:.4f}  ({ai_rate*100:.1f}%)")
        lines.append(f"    Baseline seeds:      {bl_rate:.4f}  ({bl_rate*100:.1f}%)")
        lines.append(f"    Permutation null:    {perm_mean:.4f}  (std={perm_std:.4f})")
        lines.append(f"    z-score (AI vs null): {z:.2f}")
        lines.append(f"    Empirical p-value:   {p:.6f}")
        lines.append("")

    # Per-language mean-max-cosine
    k_first = k_values[0]
    cos_col = f"max_cos_{k_first}"
    if cos_col in ai_details.columns:
        lines.append(f"--- Per-language mean max-cosine (AI seeds, K={k_first}) ---")
        lang_cos = (
            ai_details.groupby("lang")[cos_col]
            .mean()
            .sort_values(ascending=False)
        )
        for lang, mcos in lang_cos.items():
            lines.append(f"  {lang}: {mcos:.3f}")
        lines.append("")

    # Top AI seed example matches
    lines.append("--- Top AI seed example matches ---")
    top_matches = ai_details[ai_details["matched_lemma"] != ""].nlargest(15, "cosine")
    for _, row in top_matches.iterrows():
        lines.append(
            f"  {row['en_lemma']} -> {row['lang']}: "
            f"{row['matched_lemma']} (cos={row['cosine']:.3f}, "
            f"rank={int(row['lpr_rank'])})"
        )
    lines.append("")
    lines.append("=" * 70)

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Cross-lingual alignment quantification using multilingual embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    ap.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing per-language subdirectories with las_word_{lang}.csv files.",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("out/ig/cross_lingual_alignment"),
        help="Output directory for results and cached embeddings.",
    )
    ap.add_argument(
        "--n-seeds", type=int, default=20,
        help="Number of AI seeds (and baseline seeds) to select.",
    )
    ap.add_argument(
        "--k-values", type=int, nargs="+", default=[50, 200],
        help="LPR rank thresholds for hit rate calculation.",
    )
    ap.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for the null distribution.",
    )
    ap.add_argument(
        "--cosine-floor", type=float, default=0.5,
        help="Minimum cosine similarity to count as a binary hit.",
    )
    ap.add_argument(
        "--match-mode", choices=["restricted", "unrestricted"], default="restricted",
        help="restricted = search only top-K target words (default); "
             "unrestricted = search full vocabulary, check if best match is in top-K.",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    ap.add_argument(
        "--device", type=str, default="cpu",
        help="Device for sentence-transformers (cpu, cuda, mps).",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    match_fn = (
        match_seeds_to_language_restricted
        if args.match_mode == "restricted"
        else match_seeds_to_language_unrestricted
    )

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("Loading English seeds ...", file=sys.stderr)
    ai_seeds, baseline_seeds = load_english_seeds(args.data_dir, args.n_seeds)
    print(f"  AI seeds: {len(ai_seeds)} words", file=sys.stderr)
    print(f"  Baseline seeds: {len(baseline_seeds)} words", file=sys.stderr)

    ai_upos_counts = Counter(ai_seeds["upos"])
    print(f"  AI UPOS distribution: {dict(ai_upos_counts)}", file=sys.stderr)

    print("Loading target languages ...", file=sys.stderr)
    target_langs = load_target_languages(args.data_dir)
    print(f"  Loaded {len(target_langs)} target languages", file=sys.stderr)

    # ── 2. Load embedding model ───────────────────────────────────────────
    print(f"Loading embedding model ({MODEL_NAME}) ...", file=sys.stderr)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=args.device)

    # ── 3. Embed ──────────────────────────────────────────────────────────
    embed_dir = args.out_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    print("Embedding English seeds ...", file=sys.stderr)
    ai_embs = get_embeddings(
        ai_seeds["lemma"].tolist(), model, embed_dir / "en_ai_seeds.npy"
    )
    bl_embs = get_embeddings(
        baseline_seeds["lemma"].tolist(), model, embed_dir / "en_baseline_seeds.npy"
    )

    print("Embedding target languages ...", file=sys.stderr)
    target_embeddings = {}
    for lang, tdf in sorted(target_langs.items()):
        lemmas = tdf["lemma"].tolist()
        cache_path = embed_dir / f"{lang}.npy"
        target_embeddings[lang] = get_embeddings(lemmas, model, cache_path)
        print(f"  {lang}: {len(lemmas)} lemmas embedded", file=sys.stderr)

    # ── 4. Match ──────────────────────────────────────────────────────────
    print(f"Matching seeds ({args.match_mode} mode) ...", file=sys.stderr)
    all_ai_rows = []
    all_bl_rows = []
    for lang in sorted(target_langs):
        ai_rows = match_fn(
            ai_embs, ai_seeds,
            target_embeddings[lang], target_langs[lang],
            lang, args.cosine_floor, args.k_values, "ai",
        )
        all_ai_rows.extend(ai_rows)

        bl_rows = match_fn(
            bl_embs, baseline_seeds,
            target_embeddings[lang], target_langs[lang],
            lang, args.cosine_floor, args.k_values, "baseline",
        )
        all_bl_rows.extend(bl_rows)

    ai_details = pd.DataFrame(all_ai_rows)
    bl_details = pd.DataFrame(all_bl_rows)
    match_details = pd.concat([ai_details, bl_details], ignore_index=True)

    # ── 5. Seed summary ──────────────────────────────────────────────────
    def make_seed_summary(details: pd.DataFrame, seed_type: str) -> pd.DataFrame:
        rows = []
        for lemma in details["en_lemma"].unique():
            sub = details[details["en_lemma"] == lemma]
            first = sub.iloc[0]
            row = {
                "seed_type": seed_type,
                "lemma": lemma,
                "upos": first["en_upos"],
                "lpr": first["en_lpr"],
            }
            for k in args.k_values:
                cos_col = f"max_cos_{k}"
                hit_col = f"hit_at_{k}"
                if cos_col in sub.columns:
                    row[f"mean_max_cos_{k}"] = sub[cos_col].mean()
                row[f"n_langs_hit_{k}"] = int(sub[hit_col].sum())
            # Example matches (top-3 by cosine, excluding empty)
            matched = sub[sub["matched_lemma"] != ""]
            if not matched.empty:
                top3 = matched.nlargest(3, "cosine")
                examples = "; ".join(
                    f"{r['lang']}:{r['matched_lemma']}"
                    for _, r in top3.iterrows()
                )
            else:
                examples = ""
            row["example_matches"] = examples
            rows.append(row)
        return pd.DataFrame(rows)

    ai_summary = make_seed_summary(ai_details, "ai")
    bl_summary = make_seed_summary(bl_details, "baseline")
    seed_summary = pd.concat([ai_summary, bl_summary], ignore_index=True)

    # ── 6. Permutation test ───────────────────────────────────────────────
    print(f"Running {args.n_permutations} permutations ...", file=sys.stderr)
    en_path = args.data_dir / "en" / "las_word_en.csv"
    en_df = load_language_csv(en_path)

    perm_results = run_permutation_test(
        en_df=en_df,
        ai_upos_counts=ai_upos_counts,
        target_langs=target_langs,
        target_embeddings=target_embeddings,
        model=model,
        embed_cache_dir=embed_dir,
        cosine_floor=args.cosine_floor,
        k_values=args.k_values,
        n_perms=args.n_permutations,
        rng=rng,
        match_mode=args.match_mode,
    )

    # ── 7. Write output ──────────────────────────────────────────────────
    results_dir = args.out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    match_details.to_csv(results_dir / "match_details.csv", index=False)
    seed_summary.to_csv(results_dir / "seed_summary.csv", index=False)
    write_summary(
        results_dir / "summary.txt",
        ai_details, bl_details,
        perm_results, args.k_values,
        target_langs, args.n_seeds, args.n_permutations,
        args.match_mode, args.cosine_floor,
    )

    print(f"\nResults written to {results_dir}/", file=sys.stderr)

    # Print quick summary to stdout
    for k in args.k_values:
        cos_col = f"max_cos_{k}"
        hit_col = f"hit_at_{k}"
        if cos_col in ai_details.columns:
            ai_mcos = ai_details[cos_col].mean()
            bl_mcos = bl_details[cos_col].mean()
            perm_mcos = np.mean(perm_results.get(f"mean_cos_{k}", [0]))
            print(f"K={k}: mean_cos  AI={ai_mcos:.3f}  baseline={bl_mcos:.3f}  null={perm_mcos:.3f}")
        ai_rate = ai_details[hit_col].mean()
        bl_rate = bl_details[hit_col].mean()
        perm_rate = np.mean(perm_results.get(f"hit_at_{k}", [0]))
        print(f"K={k}: hit_rate  AI={ai_rate:.3f}  baseline={bl_rate:.3f}  null={perm_rate:.3f}")


if __name__ == "__main__":
    main()
