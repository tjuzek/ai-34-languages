#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
check_window_sizes.py

Check the distribution of line lengths (in tokens) across year files
for multiple languages, and compute medians for long lines.

Intended layout (assuming this script lives in the repo's tools/ directory):

    REPO_ROOT    = parent of the script directory

Default input directory structure (outside the repo):
    PROJECT_ROOT/data/
        af/2012.txt ... 2024.txt
        ar/2012.txt ... 2024.txt
        ...

Default log directory (inside the repo):
    REPO_ROOT/out/logs/

A timestamped log file will be written into REPO_ROOT/out/logs and will
contain the same information that is printed to stdout.

By default, the script checks the following files:
    2020.txt, 2021.txt, 2023.txt, 2024.txt

and uses a token boundary of 200 tokens. These can be overridden via flags.

Example usage (using defaults inferred from script location):

    python3 check_window_sizes.py

Example usage (explicit paths and parameters):

python tools/check_window_sizes.py \
      --data-root data/raw-filtered \
      --files 2020.txt 2021.txt \
      --token-boundary 30

"""

import argparse
import statistics
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


# Default language set (can be overridden via --langs if needed)
DEFAULT_LANGS = {
    "af", "ar", "bg", "cs", "de", "el", "en", "es", "et", "fa", "fi", "fr", "hi",
    "hr", "hu", "id", "is", "it", "ja", "kk", "ko", "ky", "lt", "lv", "mr", "nl",
    "pl", "pt", "ro", "ru", "sr", "ta", "tr", "uk", "zh",
}

DEFAULT_FILES = ["2020.txt", "2021.txt", "2023.txt", "2024.txt"]


def compute_defaults(script_path: Path) -> Tuple[Path, Path, Path]:
    """
    Given this script's path, compute:

    - repo_root: parent of the script directory (the repository root)
    - default_data_root: repo_root / "data"
    - default_log_dir: repo_root / "out" / "logs"
    """
    script_dir = script_path.parent
    repo_root = script_dir.parent          # repo root
    default_data_root = repo_root / "data"
    default_log_dir = repo_root / "out" / "logs"
    return default_data_root, default_log_dir, repo_root


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_data_root, default_log_dir, _repo_root = compute_defaults(script_path)

    parser = argparse.ArgumentParser(
        description="Check distribution of line lengths (tokens, chars) across year files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=(
            "Root directory containing language subfolders (input). "
            "Default: PROJECT_ROOT/data inferred from script location."
        ),
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=sorted(DEFAULT_LANGS),
        help=(
            "Languages to process (subfolder names). "
            "Default: a preset list of WMT-like language codes."
        ),
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help=(
            "Year files to inspect within each language folder. "
            f"Default: {' '.join(DEFAULT_FILES)}"
        ),
    )
    parser.add_argument(
        "--token-boundary",
        type=int,
        default=200,
        help="Threshold in tokens for 'long' lines (default: 200).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=default_log_dir,
        help=(
            "Directory for log files. "
            "Default: REPO_ROOT/out/logs inferred from script location."
        ),
    )
    return parser.parse_args()


def open_log_file(log_dir: Path) -> Path:
    """
    Create a timestamped log file path under log_dir.
    The caller is responsible for opening/closing the file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"check_window_sizes_{timestamp}.log"
    return log_path


def log_and_print(message: str, log_lines: List[str]) -> None:
    """
    Append a message to log_lines and print it to stdout.
    """
    log_lines.append(message)
    print(message)


def process_language(
    lang: str,
    data_root: Path,
    files: Sequence[str],
    token_boundary: int,
    global_char_lengths: List[int],
) -> Tuple[int, int, List[int]]:
    """
    Process a single language folder and return:

      (more, less, char_lengths)

    where:
      - more: number of lines with >= token_boundary tokens
      - less: number of lines with < token_boundary tokens
      - char_lengths: list of character lengths for lines with >= token_boundary
    """
    lang_path = data_root / lang
    more = 0
    less = 0
    char_lengths: List[int] = []

    if not lang_path.exists():
        return more, less, char_lengths  # caller will handle warning

    for fname in files:
        fpath = lang_path / fname
        if not fpath.exists():
            # caller will handle per-file warnings, but we do nothing here
            continue

        with fpath.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.rstrip("\n")
                tokens = stripped.strip().split()
                if len(tokens) >= token_boundary:
                    more += 1
                    length = len(stripped)
                    char_lengths.append(length)
                    global_char_lengths.append(length)
                else:
                    less += 1

    return more, less, char_lengths


def main() -> None:
    args = parse_args()

    data_root: Path = args.data_root
    langs: Sequence[str] = args.langs
    files: Sequence[str] = args.files
    token_boundary: int = args.token_boundary
    log_dir: Path = args.log_dir

    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    log_path = open_log_file(log_dir)
    log_lines: List[str] = []

    log_and_print(f"[INFO] Data root: {data_root}", log_lines)
    log_and_print(f"[INFO] Log file:  {log_path}", log_lines)
    log_and_print(
        f"[INFO] Files: {', '.join(files)} | Token boundary: {token_boundary}",
        log_lines,
    )

    global_more = 0
    global_less = 0
    global_char_lengths: List[int] = []

    for lang in sorted(langs):
        lang_path = data_root / lang
        if not lang_path.exists():
            log_and_print(f"[WARN] Missing folder: {lang_path}", log_lines)
            continue

        # Check for missing files and warn
        missing_files = [fname for fname in files if not (lang_path / fname).exists()]
        for fname in missing_files:
            log_and_print(f"[WARN] Missing file: {lang_path / fname}", log_lines)

        more, less, char_lengths = process_language(
            lang=lang,
            data_root=data_root,
            files=files,
            token_boundary=token_boundary,
            global_char_lengths=global_char_lengths,
        )

        total = more + less
        if total == 0:
            log_and_print(f"[{lang}] No lines found.", log_lines)
            continue

        perc_more = more / total * 100
        perc_less = less / total * 100

        if char_lengths:
            median_chars = statistics.median(char_lengths)
            median_str = f"{median_chars:.1f}"
        else:
            median_str = "n/a"

        msg = (
            f"{lang}: {perc_more:5.3f}% ≥ {token_boundary} tokens, "
            f"{perc_less:5.3f}% < {token_boundary} tokens "
            f"(total: {total}, median chars for ≥{token_boundary}: {median_str})"
        )
        log_and_print(msg, log_lines)

        global_more += more
        global_less += less

    # Global summary
    log_and_print("\n--- GLOBAL SUMMARY ---", log_lines)
    g_total = global_more + global_less
    if g_total > 0:
        global_perc_more = global_more / g_total * 100
        global_perc_less = global_less / g_total * 100
        summary = (
            f"Overall: {global_perc_more:5.3f}% ≥ {token_boundary} tokens, "
            f"{global_perc_less:5.3f}% < {token_boundary} tokens "
            f"(total: {g_total})"
        )
        log_and_print(summary, log_lines)

        if global_char_lengths:
            global_median_chars = statistics.median(global_char_lengths)
            log_and_print(
                f"Global median chars for lines with ≥{token_boundary} tokens: "
                f"{global_median_chars:.1f}",
                log_lines,
            )
        else:
            log_and_print(
                f"No lines with ≥{token_boundary} tokens found globally.",
                log_lines,
            )
    else:
        log_and_print("No lines found globally.", log_lines)

    # Finally, write the accumulated log lines to the log file
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
