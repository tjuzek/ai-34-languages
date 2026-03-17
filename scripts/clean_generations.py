#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Clean and flag GPT continuation JSONL files.

- Input: directory with *_generations.jsonl (one JSON object per line).
- Output: mirror files in an output directory with:
    - Cleaned `output.ai_continuation`.
    - New field `output.ai_continuation_pre_cleaning`.
    - Flags:
        * output.cleaned (0/1)
        * output.review_markup (0/1)
        * output.review_repetition (0/1)
        * output.review_punct_dominated (0/1)
        * output.review_any (0/1)

Example usage:
python scripts/clean_generations.py \
  --input-dir data/ig/validation-window-sizes/gpt4.1-continuations-mini-en-allyrs-35-100k \
  --output-dir data/ig/validation-window-sizes/gpt4.1-continuations-mini-en-allyrs-35-100k-cleaned
"""

import argparse
import json
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path


# -----------------------
# Config / thresholds
# -----------------------

DIGIT_DOMINANCE_THRESHOLD = 0.5  # fraction of alnum chars that are digits

# repetition / looping
REPETITION_MIN_TOKENS = 20
REPETITION_MAX_TOKEN_FRAC = 0.4
REPETITION_TTR_THRESH = 0.3

# punctuation dominance
PUNCT_FRAC_THRESHOLD = 0.4
PUNCT_SET = set(string.punctuation) | set("“”„«»—–…")

# Markup detection patterns
CODE_PAT = re.compile(
    r"```|<code>|</code>|def\s+\w+\s*\(|class\s+\w+|function\s+\w+\s*\(|var\s+\w+\s*=",
    re.IGNORECASE,
)
HTML_PAT = re.compile(r"</?[a-zA-Z][^>]*>")


# -----------------------
# Basic cleaning helpers
# -----------------------

def strip_control_chars(text: str) -> str:
    """Remove Unicode control characters, but keep newlines/tabs (if present)."""
    out_chars = []
    for ch in text:
        if ch in ("\n", "\t"):
            out_chars.append(ch)
        else:
            if not unicodedata.category(ch).startswith("C"):
                out_chars.append(ch)
    return "".join(out_chars)


def normalize_whitespace(text: str) -> str:
    """Replace tabs/newlines/nbsp with spaces, collapse multiple spaces."""
    text = text.replace("\u00A0", " ")  # non-breaking space
    text = text.replace("\t", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_quotes_ellipsis(text: str) -> str:
    """Normalise some common fancy quotes and ellipsis."""
    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "’": "'",
        "‘": "'",
        "…": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def basic_clean(text: str) -> str:
    """Pipeline of simple cleaning steps."""
    text = strip_control_chars(text)
    text = normalize_whitespace(text)
    text = normalize_quotes_ellipsis(text)
    return text


# -----------------------
# Detection helpers
# -----------------------

def is_digit_dominated(text: str, threshold: float = DIGIT_DOMINANCE_THRESHOLD) -> bool:
    """Return True if > threshold of alphanumeric chars are digits."""
    alnum_chars = [ch for ch in text if ch.isalnum()]
    if not alnum_chars:
        return False
    digit_chars = [ch for ch in alnum_chars if ch.isdigit()]
    return (len(digit_chars) / len(alnum_chars)) > threshold


def has_code_like(text: str) -> bool:
    return bool(CODE_PAT.search(text))


def has_html_like(text: str) -> bool:
    return bool(HTML_PAT.search(text))


def repetition_stats(text: str):
    """Return (n_tokens, type_token_ratio, max_token_frac)."""
    tokens = text.split()
    if not tokens:
        return 0, 0.0, 0.0
    counts = Counter(tokens)
    n = len(tokens)
    ttr = len(counts) / n
    max_frac = max(counts.values()) / n
    return n, ttr, max_frac


def is_too_repetitive(
    text: str,
    min_tokens: int = REPETITION_MIN_TOKENS,
    max_token_frac: float = REPETITION_MAX_TOKEN_FRAC,
    ttr_thresh: float = REPETITION_TTR_THRESH,
) -> bool:
    n, ttr, max_frac = repetition_stats(text)
    if n < min_tokens:
        return False
    return (max_frac > max_token_frac) and (ttr < ttr_thresh)


def frac_punct_chars(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    punct = sum(ch in PUNCT_SET for ch in chars)
    return punct / len(chars)


def is_punct_dominated(text: str, thresh: float = PUNCT_FRAC_THRESHOLD) -> bool:
    return frac_punct_chars(text) > thresh


# -----------------------
# Core processing
# -----------------------

def process_record(obj: dict) -> dict:
    """
    Clean and flag a single JSON record in-place, returning the modified object.
    """
    output_obj = obj.get("output")
    if not isinstance(output_obj, dict):
        # Nothing to do; just return the object unchanged.
        return obj

    orig = output_obj.get("ai_continuation", "")
    if not isinstance(orig, str):
        orig = "" if orig is None else str(orig)

    # Basic cleaned version for most checks
    cleaned_text = basic_clean(orig)

    # Digit dominance check
    digit_dom = is_digit_dominated(cleaned_text)

    # Markup / repetition / punctuation flags – based on cleaned text,
    # markup detection uses original to catch tags/code more reliably.
    review_markup = 1 if (has_code_like(orig) or has_html_like(orig)) else 0
    review_repetition = 1 if is_too_repetitive(cleaned_text) else 0
    review_punct = 1 if is_punct_dominated(cleaned_text) else 0
    review_any = 1 if (review_markup or review_repetition or review_punct) else 0

    # Initialise pre-cleaning field to empty; will fill if we actually change.
    output_obj["ai_continuation_pre_cleaning"] = ""

    if digit_dom:
        # Too many digits: blank the continuation, store original.
        output_obj["ai_continuation_pre_cleaning"] = orig
        output_obj["ai_continuation"] = ""
        cleaned_flag = 1
    else:
        # Apply cleaning only if it changes the string.
        if cleaned_text != orig:
            output_obj["ai_continuation_pre_cleaning"] = orig
            output_obj["ai_continuation"] = cleaned_text
            cleaned_flag = 1
        else:
            # No change
            cleaned_flag = 0
            # Keep ai_continuation as-is

    # Attach flags (0/1 ints)
    output_obj["cleaned"] = int(cleaned_flag)
    output_obj["review_markup"] = int(review_markup)
    output_obj["review_repetition"] = int(review_repetition)
    output_obj["review_punct_dominated"] = int(review_punct)
    output_obj["review_any"] = int(review_any)

    return obj


def process_file(in_path: Path, out_path: Path) -> None:
    """Process one JSONL file line by line."""
    with in_path.open("r", encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                # Preserve blank lines if any
                fout.write(line)
                continue
            obj = json.loads(line)
            obj = process_record(obj)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------
# CLI
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean and flag GPT continuation JSONL files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with *_generations.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write cleaned JSONL files (will be created if needed).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    for in_path in sorted(input_dir.iterdir()):
        if in_path.is_dir():
            # e.g. "meta" subdir
            continue
        if in_path.suffix != ".jsonl":
            continue
        out_path = output_dir / in_path.name
        print(f"Processing {in_path.name} -> {out_path.name}")
        process_file(in_path, out_path)


if __name__ == "__main__":
    main()

