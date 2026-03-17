#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
LAS analysis for AIWL continuations in JSONL format.

This script pairs human second halves with GPT continuations (matched by sent_id),
builds per-language Doc lists, and applies the LAS discovery pipeline from
analyse_las.py using lemma+UPOS keys and small window sizes (e.g. K=20).

NEW (2026-01): In discover mode, the output wLAS CSV is additionally annotated with:
  - ell_ratio_MH    = ell_M / ell_H (as before; ell_* come from LAS pipeline)
  - count_ratio_MH  = (c_M + alpha) / (c_H + alpha)   [count-smoothed ratio]
  - lpr_MH          = log(count_ratio_MH)             [log prevalence ratio]
  - lpr_guard_ok    = 1 iff c_M >= lpr_min_cm else 0
  - lpr_MH_guarded  = lpr_MH if guard ok else 0.0     [no NaNs]

Rationale:
- LPR is meant as a "spike detector": if humans never use w (c_H=0) but the model
  uses it frequently (c_M large), LPR becomes large but finite (due to smoothing).
- Guardrail prevents elevating low-support model items (default: c_M < 20).

Modes:
  - discover: compute per-language wLAS tables directly from JSONL inputs.
  - eval:     score JSONL continuations (human or model) using a precomputed
              wLAS table at sequence, document, or dataset level.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Import LAS toolkit pieces from analyse_las.py
from analyse_las import (
    STOPWORDS_EN,
    Token, Sentence, Doc,
    discover_wlas,
    select_vstar, corpus_plas,
    slas_per_sentence, dlas_per_document,
    delta_from_wtable, parse_lp_arg,
)

# --------------------------------------------------------------------
# Helpers: discover language pairs & build Doc lists from JSONL
# --------------------------------------------------------------------

# We assume two-letter language codes as filename prefix: de_, nl_, en_, zh_, ...
LANG_RE = re.compile(r"^([a-z]{2})_pregenhalves_.*\.jsonl$")


def discover_language_pairs(human_dir: Path, gpt_dir: Path) -> List[Tuple[str, Path, Path]]:
    """
    Find (lang, human_path, gpt_path) triples by matching filename prefixes.

    - Human JSONL:  <lang>_pregenhalves_*.jsonl
    - GPT JSONL:    <lang>_pregenhalves_*_generations_pos.jsonl
    """
    human_files: Dict[str, Path] = {}
    for p in human_dir.glob("*_pregenhalves_*.jsonl"):
        m = LANG_RE.match(p.name)
        if not m:
            continue
        lang = m.group(1)
        human_files[lang] = p

    pairs: List[Tuple[str, Path, Path]] = []
    for lang, hpath in sorted(human_files.items()):
        pattern = f"{lang}_pregenhalves_*_generations_pos.jsonl"
        gpaths = sorted(gpt_dir.glob(pattern))
        if not gpaths:
            print(f"[warn] no GPT generations_pos file found for lang={lang}")
            continue
        if len(gpaths) > 1:
            print(f"[warn] multiple GPT files for lang={lang}, using first: {gpaths[0].name}")
        gpath = gpaths[0]
        pairs.append((lang, hpath, gpath))

    return pairs


def build_docs_for_pair(human_path: Path, gpt_path: Path) -> Tuple[List[Doc], List[Doc]]:
    """
    Build human_docs and model_docs for a single language.

    - human_docs: one Doc per human 'second_half' continuation.
    - model_docs: one Doc per GPT 'parse_ai' continuation.

    Docs are aligned by sent_id / item_id:

      human.id.sent_id  ==  gpt.item_id

    Any human items that never went to GPT are ignored.
    """
    # 1) Index human second_half tokens by sent_id
    human_tokens_by_id: Dict[str, List[Token]] = {}

    with human_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sent_id = obj["id"]["sent_id"]
            toks: List[Token] = []
            for t in obj["parse"]["second_half"]:
                form = t.get("form", t.get("lemma", ""))
                lemma = t.get("lemma", "")
                upos = t.get("pos", "")
                toks.append(Token(form=form, lemma=lemma, upos=upos))
            human_tokens_by_id[sent_id] = toks

    # 2) Iterate GPT JSONL in its own order; build Docs for matched pairs
    human_docs: List[Doc] = []
    model_docs: List[Doc] = []

    with gpt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sent_id = obj["item_id"]

            if sent_id not in human_tokens_by_id:
                # Should be rare; skip if no matching human entry
                continue

            h_tokens = human_tokens_by_id[sent_id]
            m_tokens: List[Token] = []
            for t in obj["parse_ai"]:
                lemma = t.get("lemma", "")
                upos = t.get("pos", "")
                # No surface form stored for GPT → use lemma as form
                m_tokens.append(Token(form=lemma, lemma=lemma, upos=upos))

            h_sent = Sentence(sent_id=sent_id, tokens=h_tokens)
            m_sent = Sentence(sent_id=sent_id, tokens=m_tokens)

            human_docs.append(Doc(doc_id=sent_id, sentences=[h_sent]))
            model_docs.append(Doc(doc_id=sent_id, sentences=[m_sent]))

    return human_docs, model_docs


# --------------------------------------------------------------------
# Helpers: docs from JSONL for eval mode
# --------------------------------------------------------------------

def docs_from_jsonl(jsonl_path: Path, role: str, half: str = "second") -> List[Doc]:
    """
    Build Docs from JSONL for eval mode.

    role:
      - 'model': use GPT parse_ai tokens.
      - 'human': use parse.<half> (typically half='second').

    Each continuation is treated as a single-sentence document.
    """
    docs: List[Doc] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if role == "model":
                sent_id = obj["item_id"]
                toks: List[Token] = []
                for t in obj["parse_ai"]:
                    lemma = t.get("lemma", "")
                    upos = t.get("pos", "")
                    toks.append(Token(form=lemma, lemma=lemma, upos=upos))
            elif role == "human":
                sent_id = obj["id"]["sent_id"]
                key_half = f"{half}_half"
                toklist = obj["parse"][key_half]
                toks: List[Token] = []
                for t in toklist:
                    form = t.get("form", t.get("lemma", ""))
                    lemma = t.get("lemma", "")
                    upos = t.get("pos", "")
                    toks.append(Token(form=form, lemma=lemma, upos=upos))
            else:
                raise ValueError("role must be 'human' or 'model'.")

            sent = Sentence(sent_id=sent_id, tokens=toks)
            docs.append(Doc(doc_id=sent_id, sentences=[sent]))
    return docs


# --------------------------------------------------------------------
# LPR annotation + CSV writing (extended wLAS table)
# --------------------------------------------------------------------

def _is_stopword_en(row: Dict) -> bool:
    lem = (row.get("lemma") or "").lower()
    frm = (row.get("form") or "").lower()
    token = lem if lem else frm
    return (token in STOPWORDS_EN) if token else False


def annotate_lpr_rows(
    rows: List[Dict],
    alpha: float = 0.5,
    min_cM: int = 20,
) -> None:
    """
    In-place annotation for log prevalence ratio (LPR), based on *counts* with add-alpha smoothing:

      count_ratio_MH = (c_M + alpha) / (c_H + alpha)
      lpr_MH         = log(count_ratio_MH)

    Guardrail:
      lpr_guard_ok    = 1 iff c_M >= min_cM else 0
      lpr_MH_guarded  = lpr_MH if guard ok else 0.0  (no NaNs)

    Notes:
    - This behaves nicely when c_H = 0 (still finite and large if c_M is large).
    - If c_M = 0, guard_ok is always 0 (certainly if min_cM >= 1), which matches your preference.
    """
    a = float(alpha)
    if a <= 0.0:
        raise ValueError("alpha must be > 0 for add-alpha smoothing.")

    # 1) compute ratios + guarded variants
    for r in rows:
        c_h = float(r.get("c_H", 0.0))
        c_m = float(r.get("c_M", 0.0))

        # Count-smoothed ratio for LPR
        count_ratio = (c_m + a) / (c_h + a)
        lpr = math.log(count_ratio)

        guard_ok = 1 if c_m >= float(min_cM) else 0

        # Keep the ell-based ratio too (useful for debugging/consistency with LAS pipeline)
        ell_h = float(r.get("ell_H", 0.0))
        ell_m = float(r.get("ell_M", 0.0))
        ell_ratio = ell_m / max(ell_h, 1e-15)

        r["ell_ratio_MH"] = ell_ratio
        r["count_ratio_MH"] = count_ratio
        r["lpr_MH"] = lpr
        r["lpr_guard_ok"] = guard_ok
        r["lpr_MH_guarded"] = lpr if guard_ok else 0.0

    # 2) compute rank_LPR (descending), using unguarded lpr_MH
    idx_sorted = sorted(
        range(len(rows)),
        key=lambda i: float(rows[i].get("lpr_MH", float("-inf"))),
        reverse=True,
    )
    rank_map = {i: rank for rank, i in enumerate(idx_sorted, start=1)}
    for i, r in enumerate(rows):
        r["rank_LPR"] = rank_map[i]


def write_wlas_csv_with_lpr(
    path: Path,
    rows: List[Dict],
    topk: int,
    exc_upos_outp: set,
    exc_stpw_outp: bool,
) -> None:
    """
    Like analyse_las.write_wlas_csv(), but includes LPR-related columns.
    """
    out_rows = rows
    if exc_upos_outp:
        out_rows = [r for r in out_rows if r.get("upos", "") not in exc_upos_outp]
    if exc_stpw_outp:
        out_rows = [r for r in out_rows if not _is_stopword_en(r)]
    if topk and topk > 0:
        out_rows = out_rows[:topk]

    fieldnames = [
        # ranks
        "rank_LAS",
        "rank_LPR",
        # identity
        "key", "lemma", "form", "upos",
        # support + prevalences
        "n_pairs", "c_H", "c_M", "ell_H", "ell_M",
        # LAS
        "Delta_MH", "LAS", "pp_MH", "relpct_MH",
        # LPR
        "ell_ratio_MH",
        "count_ratio_MH",
        "lpr_MH",
        "lpr_guard_ok",
        "lpr_MH_guarded",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)


# --------------------------------------------------------------------
# Discover mode: wLAS per language (JSONL)
# --------------------------------------------------------------------

def discover_main(args: argparse.Namespace) -> None:
    human_dir: Path = args.human_dir
    gpt_dir: Path = args.gpt_dir
    out_root: Path = args.out_root

    if not human_dir.is_dir():
        raise FileNotFoundError(f"--human-dir not found: {human_dir}")
    if not gpt_dir.is_dir():
        raise FileNotFoundError(f"--gpt-dir not found: {gpt_dir}")

    pairs = discover_language_pairs(human_dir, gpt_dir)
    if not pairs:
        print("[error] no language pairs discovered.")
        return

    # For now, lemma+UPOS only, always lowercased keys unless --no-lowercase
    key_type = "lemma_pos"
    lowercase = not args.no_lowercase

    exc_calc = set(args.exc_upos_calc)
    exc_outp = set(args.exc_upos_outp)

    variant_folder = (
        f"policy={args.window_policy}"
        f"_trim={'yes' if args.trim_common else 'no'}"
        f"_M={args.num_windows}"
        f"_K={args.windowk}"
        f"_jitter={'yes' if args.jitter else 'no'}"
    )
    out_variant_root = out_root / variant_folder
    out_variant_root.mkdir(parents=True, exist_ok=True)

    for lang, human_path, gpt_path in pairs:
        print(f"[lang={lang}] {human_path.name}  <->  {gpt_path.name}")
        human_docs, model_docs = build_docs_for_pair(human_path, gpt_path)
        if not human_docs or not model_docs:
            print(f"[warn] no docs for lang={lang}, skipping.")
            continue

        rows, qc = discover_wlas(
            human_docs,
            model_docs,
            K=args.windowk,
            key_type=key_type,
            exc_upos_calc=exc_calc,
            lowercase=lowercase,
            seed0=args.seed,
            policy=args.window_policy,
            trim_common=args.trim_common,
            M=max(1, args.num_windows),
            jitter=args.jitter,
        )

        # ---- annotate LPR columns in-place ----
        annotate_lpr_rows(
            rows,
            alpha=args.lpr_alpha,
            min_cM=args.lpr_min_cm,
        )

        lang_out = out_variant_root / lang
        lang_out.mkdir(parents=True, exist_ok=True)

        csv_path = lang_out / f"las_word_{lang}.csv"
        write_wlas_csv_with_lpr(
            csv_path,
            rows,
            topk=args.topk,
            exc_upos_outp=exc_outp,
            exc_stpw_outp=args.exc_stpw_outp,
        )

        # pLAS summary over V*
        vstar_rows = select_vstar(rows, mode=args.vstar, min_hprev=args.min_hprev)
        p_summary = corpus_plas(vstar_rows)

        summary = {
            "mode": "discover_jsonl",
            "lang": lang,
            "human_file": str(human_path),
            "gpt_file": str(gpt_path),
            "params": {
                "key": key_type,
                "windowk": args.windowk,
                "num_windows": args.num_windows,
                "window_policy": args.window_policy,
                "trim_common": args.trim_common,
                "jitter": args.jitter,
                "seed": args.seed,
                "lowercase_keys": lowercase,
                "exc_upos_calc": sorted(exc_calc),
                "exc_upos_outp": sorted(exc_outp),
                "exc_stpw_outp": args.exc_stpw_outp,
                "vstar": args.vstar,
                "min_hprev": args.min_hprev,
                "lpr_alpha": args.lpr_alpha,
                "lpr_min_cm": args.lpr_min_cm,
            },
            "qc": qc,
            "pLAS": p_summary,
            "rows_written": args.topk if args.topk > 0 else len(rows),
            "csv_path": str(csv_path),
            "columns_note": (
                "CSV includes LAS columns plus ell_ratio_MH (ell-based), "
                "count_ratio_MH (count+alpha), lpr_MH=log(count_ratio_MH), "
                "lpr_guard_ok=(c_M>=min_cm), lpr_MH_guarded (0.0 if guard fails), rank_LPR."
            ),
        }

        with (lang_out / f"summary_{lang}.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[ok] discover_jsonl: {lang} → {csv_path} (n_pairs={qc.get('n_eligible', 0)})")


# --------------------------------------------------------------------
# Eval mode: sLAS / dLAS / dataset LAS on JSONL
# --------------------------------------------------------------------

def eval_main(args: argparse.Namespace) -> None:
    if args.wtable is None or args.eval_jsonl is None:
        raise ValueError("--wtable and --eval-jsonl are required for --mode eval")

    wtable = args.wtable
    eval_jsonl = args.eval_jsonl
    out_root: Path = args.out_root

    if not wtable.is_file():
        raise FileNotFoundError(f"--wtable not found: {wtable}")
    if not eval_jsonl.is_file():
        raise FileNotFoundError(f"--eval-jsonl not found: {eval_jsonl}")

    # Parse --lp: numeric p, or NONE/signed for signed mean
    lp = parse_lp_arg(args.lp)
    if lp is not None and lp <= 0:
        raise ValueError("--lp must be > 0 (or NONE/signed).")

    delta = delta_from_wtable(wtable)
    docs = docs_from_jsonl(eval_jsonl, role=args.role, half=args.half)

    exc_calc = set(args.exc_upos_calc)
    lowercase = not args.no_lowercase
    key_type = "lemma_pos"
    source_tag = args.source_tag or args.role

    eval_dir = out_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    seq_rows, m_seq = slas_per_sentence(
        docs,
        delta,
        key_type=key_type,
        lowercase=lowercase,
        exc_upos_calc=exc_calc,
        source_tag=source_tag,
        lp=lp,
    )

    stem = eval_jsonl.stem
    scoring_mode = "signed" if lp is None else "lp"

    if args.level == "sequences":
        seq_csv = eval_dir / f"las_seq_{stem}.csv"
        with seq_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "source",
                "doc_id",
                "sent_index",
                "sent_id",
                "len_tokens",
                "len_used",
                "sLAS",
                "sLAS_pos",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in seq_rows:
                w.writerow(r)

        with (eval_dir / f"summary_seq_{stem}.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": "eval_jsonl",
                    "level": "sequences",
                    "source": source_tag,
                    "scoring_mode": scoring_mode,
                    "norm_p": lp,
                    "mLAS": m_seq,
                    "rows": len(seq_rows),
                    "wtable": str(wtable),
                    "eval_jsonl": str(eval_jsonl),
                    "role": args.role,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[ok] eval_jsonl/sequences: {seq_csv}")
        return

    # Documents: mean over sequence-level sLAS per doc (here: each doc has 1 sentence)
    doc_rows, mean_doc = dlas_per_document(seq_rows)
    if args.level == "documents":
        doc_csv = eval_dir / f"las_doc_{stem}.csv"
        with doc_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["source", "doc_id", "dLAS", "dLAS_pos"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in doc_rows:
                w.writerow(r)

        with (eval_dir / f"summary_doc_{stem}.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": "eval_jsonl",
                    "level": "documents",
                    "source": source_tag,
                    "scoring_mode": scoring_mode,
                    "norm_p": lp,
                    "mLAS_from_sentences": m_seq,
                    "mean_doc_dLAS": mean_doc,
                    "docs": len(doc_rows),
                    "wtable": str(wtable),
                    "eval_jsonl": str(eval_jsonl),
                    "role": args.role,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[ok] eval_jsonl/documents: {doc_csv}")
        return

    # Dataset-level: mean over sequence-level sLAS across all docs
    if args.level == "dataset":
        with (eval_dir / f"summary_dataset_{stem}.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": "eval_jsonl",
                    "level": "dataset",
                    "source": source_tag,
                    "scoring_mode": scoring_mode,
                    "norm_p": lp,
                    "mLAS": m_seq,
                    "n_sequences": len(seq_rows),
                    "n_docs": len({r["doc_id"] for r in seq_rows}),
                    "wtable": str(wtable),
                    "eval_jsonl": str(eval_jsonl),
                    "role": args.role,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[ok] eval_jsonl/dataset: summary_dataset_{stem}.json")
        return

    raise ValueError("--level must be one of: sequences, documents, dataset")


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="LAS analysis on AIWL JSONL continuations (human halves + GPT)."
    )
    ap.add_argument(
        "--mode",
        choices=["discover", "eval"],
        required=True,
        help="discover: compute per-language wLAS (lemma+UPOS) from human vs GPT JSONL. "
             "eval: score JSONL continuations using a precomputed wLAS table.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root directory.",
    )

    # Common options
    ap.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Do NOT lowercase lemmas/forms when building keys (default: lowercase).",
    )
    ap.add_argument(
        "--exc-upos-calc",
        nargs="*",
        default=[],
        help="UPOS tags to exclude from LAS calculations/scoring (e.g. PUNCT SYM).",
    )

    # Discover-specific options
    ap.add_argument("--human-dir", type=Path, help="(discover) Directory with *_pregenhalves_*.jsonl human halves.")
    ap.add_argument("--gpt-dir", type=Path, help="(discover) Directory with *_pregenhalves_*_generations_pos.jsonl GPT outputs.")
    ap.add_argument("--windowk", type=int, default=20, help="(discover) Window size K (tokens). Default: 20.")
    ap.add_argument("--window-policy", choices=["quantile", "first", "last"], default="quantile", help="(discover) Window placement policy.")
    ap.add_argument("--trim-common", action="store_true", help="(discover) Trim starts to common support across H/M per pair (recommended).")
    ap.add_argument("--num-windows", type=int, default=1, help="(discover) Number of stratified windows per pair (M). Default: 1.")
    ap.add_argument("--jitter", action="store_true", help="(discover) Enable quantile jitter for window starts.")
    ap.add_argument("--seed", type=int, default=42, help="(discover) Base seed for window placement.")
    ap.add_argument("--topk", type=int, default=0, help="(discover) Top-k rows to output for wLAS (0 = all).")
    ap.add_argument("--exc-upos-outp", nargs="*", default=[], help="(discover) UPOS tags to exclude from written wLAS CSV (e.g. PUNCT SYM).")
    ap.add_argument("--exc-stpw-outp", action="store_true", help="(discover) Exclude English stopwords from written wLAS CSV (for English only).")
    ap.add_argument("--vstar", choices=["all", "pos"], default="all", help="(discover) V* selection for pLAS.")
    ap.add_argument("--min-hprev", type=float, default=0.0, help="(discover) Optional floor on ℓ_H for V* selection.")

    # LPR params (discover)
    ap.add_argument(
        "--lpr-alpha",
        type=float,
        default=0.5,
        help="(discover) Add-alpha smoothing for count-based LPR. Default: 0.5.",
    )
    ap.add_argument(
        "--lpr-min-cm",
        type=int,
        default=20,
        help="(discover) Guardrail: require c_M >= this for lpr_guard_ok=1. Default: 20.",
    )

    # Eval-specific options
    ap.add_argument("--wtable", type=Path, help="(eval) wLAS CSV from discover mode.")
    ap.add_argument("--eval-jsonl", type=Path, help="(eval) JSONL file with either human halves or GPT continuations.")
    ap.add_argument("--role", choices=["human", "model"], default="model", help="(eval) Which side the JSONL represents.")
    ap.add_argument("--half", choices=["first", "second"], default="second", help="(eval, role=human) Which half to score.")
    ap.add_argument("--level", choices=["sequences", "documents", "dataset"], help="(eval) Output level.")
    ap.add_argument("--source-tag", type=str, default=None, help="(eval) Label to tag the scored source.")
    ap.add_argument(
        "--lp",
        type=str,
        default="2.0",
        help="(eval) L^p exponent p for token contributions, or 'NONE'/'signed' for signed mean.",
    )

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    if args.mode == "discover":
        if args.human_dir is None or args.gpt_dir is None:
            raise ValueError("--human-dir and --gpt-dir are required for --mode discover")
        discover_main(args)
    else:
        if args.level is None:
            raise ValueError("--level is required for --mode eval")
        eval_main(args)


if __name__ == "__main__":
    main()
