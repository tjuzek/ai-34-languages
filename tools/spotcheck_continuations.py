#!/usr/bin/env python
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
spotcheck_continuations.py

Extract a simple spot-check table from GPT continuation logs.

Given a directory containing JSONL generation files like:

    af_pregenhalves_2020-2021__gpt-4.1-mini_generations.jsonl
    ar_pregenhalves_2020-2021__gpt-4.1-mini_generations.jsonl
    ...
    zh_pregenhalves_2020-2021__gpt-4.1-mini_generations.jsonl
    meta/

each JSONL line is expected to look roughly like:

    {
      "item_id": "af-2020-L53-s1",
      "lang": "af",
      ...
      "output": {
        "ai_continuation": "rigting vlieg, insluitend agteruit. ...",
        ...
      },
      "generation": { ... }
    }

This script collects (item_id, ai_continuation) for all such lines and writes
them to a single TSV file, for quick manual spot checks:

    item_id<TAB>ai_continuation
    af-2020-L53-s1<TAB>rigting vlieg, insluitend agteruit. ...
    ...

Newlines and tab characters in continuations are normalised to spaces to keep
the TSV structure well-formed.

Usage
-----

    python tools/spotcheck_continuations.py \
        --input-dir data/gpt4.1-continuations-test \
        --output out/spotcheck_continuations.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Tuple, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract (item_id, ai_continuation) from JSONL generation logs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *_generations.jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spotcheck_continuations.tsv"),
        help="Output TSV file (default: spotcheck_continuations.tsv).",
    )
    return parser.parse_args()


def iter_generations(jsonl_path: Path, stderr: TextIO) -> Iterator[Tuple[str, str]]:
    """
    Yield (item_id, ai_continuation) pairs from a single JSONL file.

    Lines that cannot be parsed as JSON or that lack the required fields
    are skipped with a warning to stderr.
    """
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"[WARN] {jsonl_path}:{line_no}: JSON decode error: {e}",
                        file=stderr,
                    )
                    continue

                item_id = obj.get("item_id")
                output = obj.get("output") or {}
                ai_cont = output.get("ai_continuation")

                if item_id is None or ai_cont is None:
                    print(
                        f"[WARN] {jsonl_path}:{line_no}: missing item_id or output.ai_continuation",
                        file=stderr,
                    )
                    continue

                # Normalise whitespace to keep TSV structure clean
                ai_cont_clean = ai_cont.replace("\t", " ").replace("\n", " ").strip()

                yield item_id, ai_cont_clean
    except OSError as e:
        print(f"[ERROR] Could not read {jsonl_path}: {e}", file=stderr)


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_path: Path = args.output

    if not input_dir.is_dir():
        print(f"[ERROR] Input dir does not exist or is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all *.jsonl files in the directory (non-recursive).
    jsonl_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix == ".jsonl"
    )

    if not jsonl_files:
        print(f"[WARN] No .jsonl files found in {input_dir}", file=sys.stderr)

    total_rows = 0

    try:
        with output_path.open("w", encoding="utf-8") as out_f:
            # Header
            out_f.write("item_id\tai_continuation\n")

            for jsonl_path in jsonl_files:
                print(f"[INFO] Processing {jsonl_path.name}", file=sys.stderr)
                for item_id, ai_cont in iter_generations(jsonl_path, sys.stderr):
                    out_f.write(f"{item_id}\t{ai_cont}\n")
                    total_rows += 1
    except OSError as e:
        print(f"[ERROR] Could not write to output file {output_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[INFO] Done. Wrote {total_rows} rows to {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

