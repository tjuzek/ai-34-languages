#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Filter overly long lines in WMT-style language subfolders.

For each language subfolder (e.g. af, ar, bg, ...) under the input root:
- Iterate over all *.txt files (e.g. 2012.txt, 2020.txt).
- For each line:
    * Compute:
        - char_len = number of characters (excluding trailing newline)
        - token_count = number of space-separated elements
    * Keep the line if:
        - token_count <= 200 AND char_len <= 1500
      Otherwise:
        - Exclude it and log (lang, filename, line_number, char_len,
          token_count, line_content).

Kept lines are written to a parallel directory structure under the output root.
Excluded lines are logged to a TSV file in REPO_ROOT/out/logs.

Defaults:
- Input root:   data/raw
- Output root:  data/raw-filtered
- Log dir:      out/logs

Usage:
    python scripts/filter_long_lines.py \
        --input-root data/raw \
        --output-root data/raw-filtered
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO, Tuple


def compute_defaults(script_path: Path) -> Tuple[Path, Path, Path]:
    """
    Given this script's path, compute:
    - repo_root: parent of script directory (the repository root)
    - log_dir: repo_root/out/logs
    """
    script_dir = script_path.parent
    repo_root = script_dir.parent          # repo root
    log_dir = repo_root / "out" / "logs"
    return repo_root, repo_root, log_dir


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root, _repo_root, log_dir = compute_defaults(script_path)

    parser = argparse.ArgumentParser(
        description="Filter overly long lines and log excluded lines."
    )
    parser.add_argument(
        "--input-root",
        "-i",
        type=Path,
        default=repo_root / "data" / "raw",
        help=(
            "Root directory containing language subfolders "
            "(default: data/raw)"
        ),
    )
    parser.add_argument(
        "--output-root",
        "-o",
        type=Path,
        default=repo_root / "data" / "raw-filtered",
        help=(
            "Root directory for filtered output (parallel structure) "
            "(default: data/raw-filtered)"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum allowed number of space-separated tokens per line (default: 200).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1500,
        help="Maximum allowed number of characters per line (default: 1500).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=log_dir,
        help=(
            "Directory for log files (default: REPO_ROOT/out/logs). "
            "You usually don't need to change this."
        ),
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for 'path' if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def open_log_file(log_dir: Path) -> TextIO:
    """Create a timestamped TSV log file and return an open handle."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"filter_long_lines_{timestamp}.tsv"
    f = log_path.open("w", encoding="utf-8", newline="")
    # Header row
    f.write(
        "language\tfilename\tline_number\tchar_len\t"
        "token_count\tline_content\n"
    )
    print(f"[INFO] Logging excluded lines to: {log_path}")
    return f


def process_file(
    lang: str,
    in_path: Path,
    out_path: Path,
    log_fh: TextIO,
    max_tokens: int,
    max_chars: int,
) -> Tuple[int, int]:
    """
    Process a single input file:

    - lang: language code (subfolder name)
    - in_path: input file path
    - out_path: output file path (in filtered folder)
    - log_fh: open log file handle
    - max_tokens, max_chars: thresholds

    Returns (kept_count, excluded_count).
    """
    ensure_parent_dir(out_path)

    kept = 0
    excluded = 0

    with in_path.open("r", encoding="utf-8", errors="replace") as fin, \
            out_path.open("w", encoding="utf-8", errors="replace") as fout:

        for line_number, line in enumerate(fin, start=1):
            # Remove trailing newline for length checks; keep original for writing/logging
            stripped = line.rstrip("\n")
            char_len = len(stripped)
            token_count = len(stripped.split())

            # Keep line if BOTH conditions are met.
            # If you want logical OR instead, change 'and' to 'or'.
            if token_count <= max_tokens and char_len <= max_chars:
                fout.write(line)
                kept += 1
            else:
                excluded += 1
                # Log excluded line
                # Note: we keep the raw content (with internal tabs, if any)
                #       so we replace newlines in the content to keep it single-line in TSV.
                safe_content = stripped.replace("\t", "\\t").replace("\n", " ")
                log_fh.write(
                    f"{lang}\t{in_path.name}\t{line_number}\t"
                    f"{char_len}\t{token_count}\t{safe_content}\n"
                )

    return kept, excluded


def main() -> None:
    args = parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    log_dir: Path = args.log_dir
    max_tokens: int = args.max_tokens
    max_chars: int = args.max_chars

    if not input_root.exists():
        print(f"[ERROR] Input root does not exist: {input_root}", file=sys.stderr)
        sys.exit(1)

    log_fh = open_log_file(log_dir)

    total_kept = 0
    total_excluded = 0
    file_count = 0

    try:
        # Iterate over language subdirectories
        for lang_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
            lang = lang_dir.name
            print(f"[INFO] Processing language: {lang}")

            # Iterate over *.txt files in this language directory
            for in_path in sorted(lang_dir.glob("*.txt")):
                rel_path = in_path.relative_to(input_root)
                out_path = output_root / rel_path

                file_count += 1
                kept, excluded = process_file(
                    lang=lang,
                    in_path=in_path,
                    out_path=out_path,
                    log_fh=log_fh,
                    max_tokens=max_tokens,
                    max_chars=max_chars,
                )
                total_kept += kept
                total_excluded += excluded

                print(
                    f"[INFO] {rel_path}: kept {kept} lines, "
                    f"excluded {excluded} lines."
                )

        print(
            f"[DONE] Processed {file_count} files. "
            f"Total kept: {total_kept}, total excluded: {total_excluded}."
        )

    finally:
        log_fh.close()


if __name__ == "__main__":
    main()

