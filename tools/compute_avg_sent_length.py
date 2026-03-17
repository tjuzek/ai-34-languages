#!/usr/bin/env python
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Compute average sentence length (in tokens) from POS-tagged data.

Assumptions
-----------
- Input directory has one subdirectory per language, e.g.:

    data/pos-basic-sample/
        af/
            2020.pos
            2021.pos
            ...
        ar/
            2020.pos
            ...

- Each *.pos file is in a basic CoNLL-U-like format:

      # sent_id = ...
      # text = ...
      1   Token   lemma   UPOS
      2   ...

  * Sentences are separated by blank lines.
  * Comment lines start with '#'.
  * Token lines are tab-separated; we count each non-comment token line
    as one token.

Output
------
Writes a TSV summary to logs/, by default:

    logs/sentence_lengths_pos-basic-sample.tsv

with columns:

    lang    total_sentences    total_tokens    avg_tokens_per_sentence

Usage
-----
From the project root (aiwl/):

    python tools/compute_avg_sent_length.py

You can override paths, e.g.:

    python tools/compute_avg_sent_length.py \
        --pos-dir data/pos-basic-sample \
        --out logs/sentence_lengths_custom.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator


def iter_sentence_lengths(path: Path) -> Iterator[int]:
    """
    Yield sentence lengths (number of token lines) for a single .pos file.

    A "sentence" is defined as a block of non-empty, non-comment lines
    (not starting with '#'), separated by blank lines.
    """
    sent_len = 0

    try:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")

                # Sentence boundary: empty line
                if not line:
                    if sent_len > 0:
                        yield sent_len
                        sent_len = 0
                    continue

                # Skip comments
                if line.startswith("#"):
                    continue

                # Token line: count if it looks like a tab-separated row
                # (conservative: require at least 2 columns)
                if "\t" in line:
                    sent_len += 1
                else:
                    # If there is some weird line, ignore it silently
                    # (you can add logging here if desired)
                    continue

        # Flush last sentence if file does not end with a blank line
        if sent_len > 0:
            yield sent_len

    except UnicodeDecodeError as e:
        raise RuntimeError(f"Unicode error in file {path}: {e}") from e


def compute_lang_stats(lang_dir: Path) -> tuple[int, int]:
    """
    Compute total number of sentences and tokens for a language directory.

    Parameters
    ----------
    lang_dir : Path
        Directory containing *.pos files for one language.

    Returns
    -------
    total_sentences : int
    total_tokens    : int
    """
    total_sentences = 0
    total_tokens = 0

    pos_files = sorted(lang_dir.glob("*.pos"))

    for pos_file in pos_files:
        for sent_len in iter_sentence_lengths(pos_file):
            total_sentences += 1
            total_tokens += sent_len

    return total_sentences, total_tokens


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    # Assume this script lives in aiwl/tools/
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]

    default_pos_dir = project_root / "data" / "pos-basic-sample"
    default_logs_dir = project_root / "logs"
    default_out = default_logs_dir / "sentence_lengths_pos-basic-sample.tsv"

    parser = argparse.ArgumentParser(
        description="Compute average sentence length (tokens) "
                    "per language from POS-tagged files."
    )
    parser.add_argument(
        "--pos-dir",
        type=Path,
        default=default_pos_dir,
        help=f"Root directory with language subfolders (default: {default_pos_dir})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output TSV file (default: {default_out})",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for CLI.
    """
    args = parse_args(argv)

    pos_root: Path = args.pos_dir
    out_path: Path = args.out

    if not pos_root.exists() or not pos_root.is_dir():
        print(f"ERROR: pos-dir does not exist or is not a directory: {pos_root}", file=sys.stderr)
        return 1

    # Ensure logs directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover language subdirectories
    lang_dirs = sorted(
        d for d in pos_root.iterdir() if d.is_dir()
    )

    if not lang_dirs:
        print(f"ERROR: no language subdirectories found under {pos_root}", file=sys.stderr)
        return 1

    # Compute stats
    rows = []
    for lang_dir in lang_dirs:
        lang = lang_dir.name
        total_sentences, total_tokens = compute_lang_stats(lang_dir)

        if total_sentences > 0:
            avg_len = total_tokens / total_sentences
        else:
            avg_len = 0.0

        rows.append((lang, total_sentences, total_tokens, avg_len))

    # Write TSV
    with out_path.open("w", encoding="utf-8") as out_f:
        out_f.write("lang\ttotal_sentences\ttotal_tokens\tavg_tokens_per_sentence\n")
        for lang, total_s, total_t, avg in rows:
            out_f.write(f"{lang}\t{total_s}\t{total_t}\t{avg:.4f}\n")

    print(f"Wrote sentence length summary to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

