#!/usr/bin/env python
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Build Continuation Dataset (JSONL) - Memory Optimized

Constructs a continuation benchmark for GPT experiments by splitting 
POS-tagged sentences into (Prompt, Reference) pairs.

Features:
  - Dynamic Split Logic: Uses Seam Search (+/- 2 radius) to find clean cuts.
  - Morphology Handling: Fixes clitics for Arabic, Marathi, Tamil.
  - Traceability: Preserves original text, split indices, and Gold POS tags.
  - Output: Structured JSONL files ready for evaluation pipelines.
  - OOM Fix: Processes languages sequentially to minimize RAM usage.

Usage example:
python scripts/build_pregen_continuations.py \
  --pos-root data/ig/pos-basic-sample \
  --output-dir data/ig/pre-gen-halves \
  --years 2020 2021 \
  --min-half-tokens 12 \
  --max-items-per-lang 100000 \
  --split-rule floor \
  --seed 42 \
  --debug
"""

import argparse
import json
import random
import re
import gc
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional


DEFAULT_NO_SPACE_LANGS = {"zh", "ja"}

# --- LANGUAGE CONFIGURATION ---

# 1. Prefix Clitics (Arabic): Attach to the NEXT word.
PREFIX_CLITICS_BY_LANG = {
    "ar": {"و", "ف", "ب", "ك", "ل", "س"}
}

# 2. Suffix Clitics (Marathi/Tamil): Attach to the PREVIOUS word.
SUFFIX_CLITICS_BY_LANG = {
    "mr": {
        "च्या", "चे", "चा", "ची", "वर", "तील", "तून", "नी", "शी", "ही", "त", "कडे", "मध्ये"
    },
    "ta": {
        "உம்", "ஆக", "உடன்", "இல்", "இடம்", "ஆல்", "கு", "ஐ", "அது"
    }
}


class AlignmentError(Exception):
    """Raised when token-to-text alignment fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pre-generation continuation dataset (JSONL)."
    )
    parser.add_argument(
        "--pos-root",
        type=str,
        required=True,
        help="Root directory with POS-tagged data (lang subfolders, YEAR.pos files).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pre-gen-continuations",
        help="Directory to write per-language JSONL files. Default: data/pre-gen-continuations",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021],
        help="Years to include. Default: 2020 2021.",
    )
    parser.add_argument(
        "--langs",
        type=str,
        nargs="+",
        default=None,
        help="Languages to include. Default: all subdirectories.",
    )
    parser.add_argument(
        "--min-half-tokens",
        type=int,
        default=20,
        help="Minimum number of tokens required in each half. Default: 20.",
    )
    parser.add_argument(
        "--max-items-per-lang",
        type=int,
        default=None,
        help="Maximum number of items per language. If omitted, keep all.",
    )
    parser.add_argument(
        "--split-rule",
        type=str,
        choices=["floor", "ceil", "random"],
        default="floor",
        help="How to choose split index when length is odd. Default: floor.",
    )
    parser.add_argument(
        "--no-space-langs",
        type=str,
        default="zh,ja",
        help="Comma-separated list of language IDs without spaces. Default: 'zh,ja'.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Seed for random operations (int or 'random'). Defaults to 42.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, log stitching/alignment mismatches to debug log files.",
    )
    parser.add_argument(
        "--debug-log-dir",
        type=str,
        default="out/logs",
        help="Directory for debug logs. Default: out/logs",
    )
    return parser.parse_args()


def determine_seed(seed_arg: Optional[str]) -> int:
    if seed_arg is None:
        print("[WARN] No --seed provided, defaulting to 42 for reproducibility.")
        return 42
    if seed_arg.lower() == "random":
        seed = random.SystemRandom().randint(0, 2**32 - 1)
        print(f"[INFO] Using system-random seed: {seed}")
        return seed
    try:
        seed = int(seed_arg)
    except ValueError:
        raise ValueError(f"Invalid seed argument: {seed_arg!r} (use int or 'random')")
    print(f"[INFO] Using fixed seed: {seed}")
    return seed


def discover_langs(pos_root: Path, langs_arg: Optional[List[str]]) -> List[str]:
    if langs_arg:
        langs = langs_arg
    else:
        langs = [
            p.name
            for p in pos_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]
    langs = sorted(langs)
    print(f"[INFO] Languages: {', '.join(langs)}")
    return langs


def parse_no_space_langs(arg: str) -> set:
    if not arg.strip():
        return set()
    return {x.strip() for x in arg.split(",") if x.strip()}


def iter_sentences_from_pos_file(
    path: Path,
) -> Iterable[Tuple[Optional[str], Optional[str], List[Dict[str, str]]]]:
    """
    Yield (sent_id, text_comment, token_dicts) for each sentence.
    
    token_dicts is a list of dicts: {'form': 'Word', 'lemma': 'word', 'pos': 'NOUN'}
    """
    sent_id = None
    text_comment = None
    tokens: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line:
                if tokens:
                    yield sent_id, text_comment, tokens
                    sent_id, text_comment, tokens = None, None, []
                continue

            if line.startswith("#"):
                if line.startswith("# sent_id ="):
                    sent_id = line[len("# sent_id =") :].strip()
                elif line.startswith("# text ="):
                    text_comment = line[len("# text =") :].strip()
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue
            
            # Parse CoNLL columns (robustly)
            # 0: ID, 1: FORM, 2: LEMMA, 3: UPOS
            form = parts[1]
            lemma = parts[2] if len(parts) > 2 else None
            pos = parts[3] if len(parts) > 3 else None
            
            token_data = {"form": form}
            if lemma and lemma != "_": token_data["lemma"] = lemma
            if pos and pos != "_": token_data["pos"] = pos
            
            tokens.append(token_data)

        if tokens:
            yield sent_id, text_comment, tokens


def reconstruct_text(tokens: List[str], lang: str, no_space_langs: set) -> str:
    """Reconstruct text from token surface strings."""
    if not tokens:
        return ""

    if lang in no_space_langs:
        return "".join(tokens)

    s = " ".join(tokens)

    # 1. Standard Punctuation
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+([)\]\}»])", r"\1", s)
    s = re.sub(r"([(\[«])\s+", r"\1", s)

    # 2. Hyphens
    s = re.sub(r'([a-zA-Z0-9_])\s+-\s+([a-zA-Z0-9_])', r'\1-\2', s)
    s = re.sub(r'(\s|^)-\s+([a-zA-Z0-9_])', r'\1-\2', s)
    s = re.sub(r'([a-zA-Z0-9_])\s+-(?=\s|$)', r'\1-', s)

    # 3. Elisions (Apostrophes)
    s = re.sub(r"([a-zA-Z0-9])'\s+([a-zA-Z0-9])", r"\1'\2", s)

    # 4. Quotes
    s = re.sub(r'(\s|^)"\s+([a-zA-Z0-9_])', r'\1"\2', s)
    s = re.sub(r'([a-zA-Z0-9_])\s+"(?=[\s.,;:!?]|$)', r'\1"', s)
    s = re.sub(r"(\s|^)'\s+([a-zA-Z0-9_])", r"\1'\2", s)
    s = re.sub(r"([a-zA-Z0-9_])\s+'(?=[\s.,;:!?]|$)", r"\1'", s)
    
    # 5. Prefix Clitics (Arabic)
    if lang in PREFIX_CLITICS_BY_LANG:
        clitics = "".join(PREFIX_CLITICS_BY_LANG[lang])
        s = re.sub(f'(\\s|^)([{clitics}])\\s+([^\\s])', r'\1\2\3', s)

    # 6. Suffix Clitics (Marathi/Tamil)
    if lang in SUFFIX_CLITICS_BY_LANG:
        suffixes = "|".join(re.escape(c) for c in SUFFIX_CLITICS_BY_LANG[lang])
        s = re.sub(f'([^\\s])\\s+({suffixes})(?=\\s|$)', r'\1\2', s)

    return s


def find_split_in_original_text(text: str, token_forms: List[str], split_idx: int) -> int:
    """Seam Search: Locate split point in original text via regex."""
    start = max(0, split_idx - 3)
    end = min(len(token_forms), split_idx + 3)
    
    left_tokens = token_forms[start:split_idx]
    right_tokens = token_forms[split_idx:end]
    
    if not left_tokens or not right_tokens:
        return -1

    def build_pattern(toks):
        escaped = [re.escape(t) for t in toks]
        return r"[\W_]*".join(escaped)

    left_pat_str = build_pattern(left_tokens)
    right_pat_str = build_pattern(right_tokens)

    full_pattern = f"({left_pat_str}[\\W_]*)({right_pat_str})"
    matches = list(re.finditer(full_pattern, text))
    
    if len(matches) == 1:
        m = matches[0]
        return m.end(1)
        
    return -1


def parse_sent_id(sent_id: Optional[str]) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[int]]:
    if not sent_id: return None, None, None, None
    parts = sent_id.split("-")
    if len(parts) < 4: return None, None, None, None
    
    lang = parts[0]
    try: year = int(parts[1])
    except ValueError: year = None
    doc_id = parts[2]
    
    sent_idx = None
    if parts[3].startswith("s"):
        try: sent_idx = int(parts[3][1:])
        except ValueError: sent_idx = None
    return lang, year, doc_id, sent_idx


def choose_split_index(n_tokens: int, rule: str, rng: random.Random) -> int:
    if n_tokens < 2:
        raise ValueError("Cannot split sentence with fewer than 2 tokens.")
    half_floor = n_tokens // 2
    half_ceil = (n_tokens + 1) // 2
    
    if rule == "floor": split_idx = half_floor
    elif rule == "ceil": split_idx = half_ceil
    elif rule == "random":
        split_idx = half_floor if half_floor == half_ceil else rng.choice([half_floor, half_ceil])
    else: raise ValueError(f"Unknown split rule: {rule}")
    
    return max(1, min(split_idx, n_tokens - 1))


def align_tokens_to_text(text: str, token_forms: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    pos = 0
    for tok in token_forms:
        if tok == "":
            spans.append((pos, pos))
            continue
        idx = text.find(tok, pos)
        if idx == -1:
            raise AlignmentError(f"Token {tok!r} not found in text starting at {pos}")
        spans.append((idx, idx + len(tok)))
        pos = idx + len(tok)
    return spans


def prepare_items_for_language(
    lang: str,
    pos_root: Path,
    years: List[int],
    min_half_tokens: int,
    split_rule: str,
    base_rng: random.Random,
    no_space_langs: set,
    debug: bool,
    debug_log_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    
    items: List[Dict[str, Any]] = []
    lang_dir = pos_root / lang
    if not lang_dir.is_dir():
        print(f"[WARN] Language directory not found: {lang_dir}")
        return items

    lang_seed = (hash(lang) + base_rng.randint(0, 2**31 - 1)) & 0xFFFFFFFF
    lang_rng = random.Random(lang_seed)
    years_sorted = sorted(set(int(y) for y in years))

    debug_file = None
    if debug and debug_log_dir:
        debug_path = debug_log_dir / f"{lang}-stitching.log"
        debug_file = debug_path.open("w", encoding="utf-8")
        print(f"[INFO] Debug log for {lang}: {debug_path}")

    try:
        for year in years_sorted:
            pos_path = lang_dir / f"{year}.pos"
            if not pos_path.is_file():
                continue

            n_sent_total = 0
            n_sent_kept = 0

            for sent_id, text_comment, token_dicts in iter_sentences_from_pos_file(pos_path):
                n_sent_total += 1
                token_forms = [t["form"] for t in token_dicts]
                n_tokens = len(token_forms)
                
                if n_tokens < 2 * min_half_tokens:
                    continue

                initial_split_idx = choose_split_index(n_tokens, split_rule, lang_rng)
                
                # Variables to populate
                final_split_idx = initial_split_idx
                split_method = "unknown"
                split_shift = 0
                
                first_text = ""
                second_text = ""
                full_text_inferred = ""
                
                orig_text = text_comment
                alignment_success = False
                alignment_fail_reason = None

                # 1. Strategy: Strict Alignment
                if orig_text:
                    try:
                        spans = align_tokens_to_text(orig_text, token_forms)
                        # If we get here, tokens map 1:1 to text
                        split_char = spans[initial_split_idx - 1][1]
                        
                        first_text = orig_text[:split_char]
                        second_text = orig_text[split_char:]
                        full_text_inferred = orig_text # Perfect match
                        
                        split_method = "strict_alignment"
                        split_shift = 0
                        alignment_success = True
                        
                    except AlignmentError as e:
                        alignment_fail_reason = str(e)

                # 2. Strategy: Seam Search with Radius
                if not alignment_success and orig_text:
                    search_offsets = [0, 1, -1, 2, -2]
                    
                    for offset in search_offsets:
                        trial_idx = initial_split_idx + offset
                        
                        # Valid split index check
                        if trial_idx < min_half_tokens or trial_idx > (n_tokens - min_half_tokens):
                            continue
                            
                        split_char = find_split_in_original_text(orig_text, token_forms, trial_idx)
                        if split_char != -1:
                            final_split_idx = trial_idx
                            first_text = orig_text[:split_char]
                            second_text = orig_text[split_char:]
                            full_text_inferred = orig_text
                            
                            split_method = "seam_search"
                            split_shift = offset
                            alignment_success = True
                            break
                
                # 3. Strategy: Fallback Reconstruction
                if not alignment_success:
                    # Use initial split index, no shift
                    final_split_idx = initial_split_idx
                    
                    first_forms = token_forms[:final_split_idx]
                    second_forms = token_forms[final_split_idx:]
                    
                    full_text_inferred = reconstruct_text(token_forms, lang, no_space_langs)
                    first_text = reconstruct_text(first_forms, lang, no_space_langs)
                    second_text = reconstruct_text(second_forms, lang, no_space_langs)
                    
                    split_method = "fallback_reconstruction"
                    split_shift = 0
                    
                    # --- JOIN LOGIC (Space Insertion) ---
                    if second_forms and lang not in no_space_langs:
                        needs_space = True
                        
                        # A) Punctuation check
                        left_attach_punct = {",", ".", ";", ":", "!", "?", ")", "]", "}", "»"}
                        if second_forms[0] in left_attach_punct:
                            needs_space = False
                        
                        # B) Prefix Clitic Check (Arabic)
                        if lang in PREFIX_CLITICS_BY_LANG and first_forms:
                            if first_forms[-1] in PREFIX_CLITICS_BY_LANG[lang]:
                                needs_space = False
                                
                        # C) Suffix Clitic Check (Marathi/Tamil)
                        if lang in SUFFIX_CLITICS_BY_LANG:
                            if second_forms[0] in SUFFIX_CLITICS_BY_LANG[lang]:
                                needs_space = False
                        
                        if needs_space:
                            first_text += " "

                # Final verification of length
                first_parsed = token_dicts[:final_split_idx]
                second_parsed = token_dicts[final_split_idx:]
                
                if len(first_parsed) < min_half_tokens or len(second_parsed) < min_half_tokens:
                    continue

                # Build Metadata
                parsed_lang, parsed_year, doc_id, sent_idx = parse_sent_id(sent_id)
                if parsed_year is None: parsed_year = year
                if parsed_lang is None: parsed_lang = lang

                is_perfect = False
                if orig_text and orig_text == full_text_inferred:
                    is_perfect = True

                # Construct Hierarchical Item
                item = {
                    "id": {
                        "lang": parsed_lang,
                        "year": parsed_year,
                        "file": f"{lang}/{year}.pos",
                        "sent_id": sent_id or "",
                        "doc_id": doc_id or "",
                        "sent_idx": sent_idx if sent_idx is not None else -1
                    },
                    "metrics": {
                        "n_tokens_full": n_tokens,
                        "n_tokens_first": len(first_parsed),
                        "n_tokens_second": len(second_parsed),
                        "char_len_full": len(full_text_inferred),
                        "char_len_first": len(first_text),
                        "char_len_second": len(second_text)
                    },
                    "split_meta": {
                        "index": final_split_idx,
                        "method": split_method,
                        "shift": split_shift,
                        "is_perfect_reconstruction": is_perfect
                    },
                    "text": {
                        "full_original": orig_text or "",
                        "full_inferred": full_text_inferred,
                        "first_half": first_text,
                        "second_half": second_text
                    },
                    "parse": {
                        "first_half": first_parsed,
                        "second_half": second_parsed
                    }
                }

                # Debug Logging
                if debug_file is not None and orig_text:
                    concat_halves = first_text + second_text
                    log_this = False
                    if split_method == "strict_alignment" and not is_perfect: log_this = True
                    if split_method == "seam_search" and not is_perfect: log_this = True
                    if split_method == "fallback_reconstruction" and not is_perfect: log_this = True

                    if log_this:
                        print("=== SENTENCE ===", file=debug_file)
                        print(f"sent_id: {sent_id}", file=debug_file)
                        print(f"method: {split_method} (shift {split_shift})", file=debug_file)
                        if alignment_fail_reason:
                            print(f"strict_align_failed: {alignment_fail_reason}", file=debug_file)
                        print(f"orig_text: {orig_text}", file=debug_file)
                        print(f"full_inf:  {full_text_inferred}", file=debug_file)
                        print(f"concat:    {concat_halves}", file=debug_file)
                        print("", file=debug_file)

                items.append(item)
                n_sent_kept += 1

            print(f"[INFO] {lang} {year}: eligible/kept={n_sent_kept} (total={n_sent_total})")
    finally:
        if debug_file: debug_file.close()

    return items


def sample_items_for_single_language(
    items: List[Dict[str, Any]],
    lang: str,
    max_items_per_lang: Optional[int],
    base_seed: int,
) -> List[Dict[str, Any]]:
    """Samples items for a single language if max limit is exceeded."""
    if max_items_per_lang is None:
        return items

    n = len(items)
    if n <= max_items_per_lang:
        return items

    # Deterministic seed per language, mixed with base seed
    lang_seed = (base_seed * 31 + sum(ord(c) for c in lang)) & 0xFFFFFFFF
    rng = random.Random(lang_seed)
    
    print(f"[INFO] {lang}: sampling {max_items_per_lang} from {n} items.")
    return rng.sample(items, max_items_per_lang)


def write_jsonl_for_single_language(
    output_dir: Path,
    lang: str,
    items: List[Dict[str, Any]],
    years: List[int],
) -> None:
    """Writes the items for a single language to its JSONL file."""
    if not items:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    min_year = min(years)
    max_year = max(years)

    fname = f"{lang}_pregenhalves_{min_year}-{max_year}.jsonl"
    out_path = output_dir / fname

    # Sort by year, doc_id, sent_idx for determinism
    items_sorted = sorted(
        items,
        key=lambda x: (x["id"]["year"], x["id"]["file"], x["id"]["sent_idx"])
    )

    with out_path.open("w", encoding="utf-8") as f:
        for item in items_sorted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(items_sorted)} items to {out_path}")


def main() -> None:
    args = parse_args()
    pos_root = Path(args.pos_root)
    output_dir = Path(args.output_dir)
    base_seed = determine_seed(args.seed)
    base_rng = random.Random(base_seed)
    
    langs = discover_langs(pos_root, args.langs)
    years = [int(y) for y in args.years]
    no_space_langs = parse_no_space_langs(args.no_space_langs)

    print(f"[INFO] Years: {years}")
    if args.max_items_per_lang:
        print(f"[INFO] max_items_per_lang = {args.max_items_per_lang}")

    debug_log_dir = None
    if args.debug:
        debug_log_dir = Path(args.debug_log_dir)
        debug_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Debug logs: {debug_log_dir}")

    # --- MAIN LOOP: Process One Language at a Time ---
    for lang in langs:
        print(f"--- Processing {lang} ---")
        
        # 1. Build
        items = prepare_items_for_language(
            lang=lang,
            pos_root=pos_root,
            years=years,
            min_half_tokens=args.min_half_tokens,
            split_rule=args.split_rule,
            base_rng=base_rng,
            no_space_langs=no_space_langs,
            debug=args.debug,
            debug_log_dir=debug_log_dir,
        )
        
        # 2. Sample (in place, effectively)
        items = sample_items_for_single_language(
            items,
            lang=lang,
            max_items_per_lang=args.max_items_per_lang,
            base_seed=base_seed,
        )

        # 3. Write
        write_jsonl_for_single_language(output_dir, lang, items, years)
        
        # 4. Cleanup (Explicitly free memory before next lang)
        del items
        gc.collect()

    print("[INFO] Done.")


if __name__ == "__main__":
    main()