#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Sample at most 3 million lines per file from a WMT-style folder structure.

Structure assumed:

    INPUT_ROOT/
        af/
            2020.txt
            2021.txt
            ...
        ar/
            2020.txt
            ...
        ...

For each file:
- If line_count < max_lines: copy file to OUTPUT_ROOT/LANG/
- If line_count > max_lines: randomly sample exactly max_lines lines
  (without replacement, reproducible with a fixed seed) and write
  the sampled file to OUTPUT_ROOT/LANG/.

Usage example:

    python scripts/sample_lines_per_file.py \
        --input-root data/raw-filtered \
        --output-root data/raw-filtered-sampled \
        --max-lines 3000000 \
        --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path


def count_lines(path: Path) -> int:
    """Count lines in a text file (fast, in binary mode)."""
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def sample_file_lines(
    in_path: Path,
    out_path: Path,
    total_lines: int,
    max_lines: int,
    seed: int,
    lang: str,
) -> None:
    """
    Sample exactly max_lines line indices from [0, total_lines-1] without replacement,
    then write only those lines to out_path.

    Uses a per-file seed derived from (seed, lang, filename) for reproducibility.
    """
    # Per-file deterministic seed
    file_key = (seed, lang, in_path.name)
    file_seed = hash(file_key) & 0xFFFFFFFF
    rng = random.Random(file_seed)

    # Choose which line indices to keep
    # Note: we index lines from 0 .. total_lines-1
    keep_indices = set(rng.sample(range(total_lines), max_lines))

    # Second pass: write only selected lines
    with in_path.open("r", encoding="utf-8", errors="replace") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for i, line in enumerate(fin):
            if i in keep_indices:
                fout.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cap each file at max_lines lines by random sampling."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Input root directory with language subfolders (e.g. raw-filtered).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root directory for sampled files.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=3_000_000,
        help="Maximum number of lines per file (default: 3,000,000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducible sampling (default: 42).",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Max lines per file: {args.max_lines}")
    print(f"Base seed: {args.seed}")
    print()

    # Iterate over language subfolders
    for lang_dir in sorted(input_root.iterdir()):
        if not lang_dir.is_dir():
            continue

        lang = lang_dir.name
        out_lang_dir = output_root / lang
        out_lang_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing language: {lang}")

        # Iterate over files in the language folder (e.g. 2020.txt, 2021.txt)
        for in_path in sorted(lang_dir.iterdir()):
            if not in_path.is_file():
                continue

            out_path = out_lang_dir / in_path.name

            # Count lines
            total_lines = count_lines(in_path)

            if total_lines < args.max_lines:
                # Just copy the file
                shutil.copy2(in_path, out_path)
                print(
                    f"  {in_path.name}: {total_lines} lines (< {args.max_lines}), copied."
                )
            else:
                # Sample max_lines without replacement
                print(
                    f"  {in_path.name}: {total_lines} lines, sampling {args.max_lines}..."
                )
                sample_file_lines(
                    in_path=in_path,
                    out_path=out_path,
                    total_lines=total_lines,
                    max_lines=args.max_lines,
                    seed=args.seed,
                    lang=lang,
                )
                print("    done.")

    print("\nAll done.")


if __name__ == "__main__":
    main()

