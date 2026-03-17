#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Sentence-length distribution explorer for AIWL POS-tagged data.

This tool scans 2020.pos and 2021.pos in each language subfolder and
computes, per language:

    - total number of parsed sentences (across the given years)
    - percentage of sentences whose token length exceeds various thresholds
      (>10, >15, ..., >60 tokens)

Token length is defined as the number of token lines in a sentence block
(i.e. non-comment, non-empty lines), including punctuation tokens. The
input is assumed to be in the "basic" Stanza/AIWL format, with sentences
separated by blank lines and optional '# sent_id' / '# text' comment
lines.

Expected directory layout (example):

    data/pos-basic-sample/
        af/
            2020.pos
            2021.pos
        ar/
            2020.pos
            2021.pos
        ...
        zh/
            2020.pos
            2021.pos

Output: a TSV file with one row per language and columns:

    lang    total   %>10-tk    %>15-tk    %>20-tk    ...    %>60-tk

Example usage:

    python tools/parses_len_distribution.py \
        --input-dir data/pos-basic-sample \
        --output out/analysis/parse_len_2020_2021.tsv

By default, the script looks for 2020.pos and 2021.pos.
You can override the years with --years 2019 2020, etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Sentence-length thresholds (strictly "greater than" these values)
THRESHOLDS: List[int] = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def iter_sentence_lengths(path: Path) -> Iterable[int]:
    """
    Yield sentence lengths (in tokens) for a .pos file.

    - Sentences are separated by blank lines.
    - Lines starting with '#' are treated as comments and ignored.
    - A token line is any non-empty line whose first character is a digit;
      for robustness, lines that are non-empty and not comments but do
      not start with a digit are still counted as one token.
    """
    length = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Blank line → sentence boundary
            if not line.strip():
                if length > 0:
                    yield length
                    length = 0
                continue

            # Comment line → ignore
            if line.startswith("#"):
                continue

            # Token line
            first_char = line[0]
            if first_char.isdigit():
                length += 1
            else:
                # Be forgiving if format deviates slightly
                length += 1

        # Last sentence if file does not end with a blank line
        if length > 0:
            yield length


def analyse_language(lang_dir: Path, years: Iterable[int]) -> Tuple[int, Dict[int, int]]:
    """
    Analyse one language directory over the given years.

    Returns:
        total_sents: total number of parsed sentences across the given years
        counts: dict mapping threshold -> #sentences with length > threshold
    """
    total_sents = 0
    counts: Dict[int, int] = {thr: 0 for thr in THRESHOLDS}

    for year in years:
        pos_path = lang_dir / f"{year}.pos"
        if not pos_path.is_file():
            # Silently skip missing files; you can add logging here if desired.
            continue

        for sent_len in iter_sentence_lengths(pos_path):
            total_sents += 1
            for thr in THRESHOLDS:
                if sent_len > thr:
                    counts[thr] += 1

    return total_sents, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute sentence-length distributions from POS-tagged .pos files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing language subfolders (af, ar, ..., zh).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output TSV file (e.g. out/analysis/sent_len_2020_2021.tsv).",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021],
        help="Years to include (expects <year>.pos in each language folder). "
             "Default: 2020 2021",
    )

    args = parser.parse_args()

    input_root = Path(args.input_dir)
    out_path = Path(args.output)
    years: List[int] = args.years

    if not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_root}")

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover language subdirectories
    lang_dirs = sorted(
        [p for p in input_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )

    if not lang_dirs:
        raise SystemExit(f"No language subdirectories found under {input_root}")

    # Write header + rows
    header_cols = ["lang", "total"] + [f"%>{thr}-tk" for thr in THRESHOLDS]

    with out_path.open("w", encoding="utf-8") as out_f:
        out_f.write("\t".join(header_cols) + "\n")

        for lang_dir in lang_dirs:
            lang = lang_dir.name
            total_sents, counts = analyse_language(lang_dir, years)

            if total_sents == 0:
                # No sentences found for this language/years; skip row
                continue

            pct_values: List[str] = []
            for thr in THRESHOLDS:
                pct = 100.0 * counts[thr] / total_sents
                pct_values.append(f"{pct:.3f}")

            row = [lang, str(total_sents)] + pct_values
            out_f.write("\t".join(row) + "\n")

    print(f"Wrote sentence-length distribution to {out_path}")


if __name__ == "__main__":
    main()
