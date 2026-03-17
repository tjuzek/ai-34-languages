#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
List file sizes and line counts per language in a WMT-style folder structure.

Usage:

    python tools/list_file_stats.py --root data/raw-filtered \
                              --output meta/file_stats.tsv
"""

import argparse
from pathlib import Path


def count_lines(path: Path) -> int:
    """Return the number of lines in a file."""
    # Using binary mode is a bit faster and avoids encoding issues.
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List file sizes and line counts per language from language subfolders."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory containing language subfolders (default: current directory).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="file_stats.tsv",
        help="Output TSV file (default: file_stats.tsv).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.output).resolve()

    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    with out_path.open("w", encoding="utf-8") as out_f:
        # Header
        out_f.write("lang\tfile\tsize_bytes\tline_count\n")

        # Iterate over language subfolders
        for lang_dir in sorted(root.iterdir()):
            if not lang_dir.is_dir():
                continue

            lang = lang_dir.name

            # Iterate over files in the language folder
            for file_path in sorted(lang_dir.iterdir()):
                if not file_path.is_file():
                    continue

                size_bytes = file_path.stat().st_size
                line_count = count_lines(file_path)

                out_f.write(
                    f"{lang}\t{file_path.name}\t{size_bytes}\t{line_count}\n"
                )

    print(f"Wrote file stats to: {out_path}")


if __name__ == "__main__":
    main()
