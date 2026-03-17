#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
import os
import csv
import json
import re
import math
import argparse

# --- CONFIGURATION DEFAULTS ---
DEFAULT_INPUT_ROOT = "csv_files"  # The root of your experiment outputs
DEFAULT_OUTPUT_DIR = "data"       # Where the website reads from

# Guardrail: words must appear at least this many times in AI text
# to be considered for "High Impact" ranking.
DEFAULT_MIN_AI_COUNT_FOR_IMPACT = 20

# Jeffreys smoothing for ratios (prevents division by zero and "NEW" masking spikes)
DEFAULT_RATIO_SMOOTH = 0.5


def clean_model_name(folder_name: str) -> str:
    """Clean up model names from folder paths."""
    name = folder_name.replace("las-", "")
    name = re.sub(r"-\d{4}-\d{2}-\d{2}.*", "", name)
    return name


# Unicode-aware "has at least one alphanumeric char" check.
# Using str.isalnum() keeps this robust for non-Latin scripts.
def has_any_alnum(token: str) -> bool:
    if token is None:
        return False
    token = str(token).strip()
    if not token:
        return False
    return any(ch.isalnum() for ch in token)


def process_directory(
    input_root: str,
    output_dir: str,
    min_ai_count_for_impact: int,
    mode: str,
    ratio_smooth: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    inventory = []

    print(f"📂 Scanning '{input_root}' for experimental results...")
    print(
        f"🧰 Output mode: {mode} (min_ai_count_for_impact={min_ai_count_for_impact}, ratio_smooth={ratio_smooth})"
    )

    for root, _, files in os.walk(input_root):
        csv_files = [f for f in files if f.startswith("las_word_") and f.endswith(".csv")]

        for csv_file in csv_files:
            lang = csv_file.replace("las_word_", "").replace(".csv", "")
            summary_file = f"summary_{lang}.json"

            if summary_file not in files:
                continue

            # --- 1) EXTRACT METADATA ---
            path_parts = os.path.normpath(root).split(os.sep)
            try:
                try:
                    root_idx = path_parts.index(os.path.basename(input_root))
                    register = path_parts[root_idx + 1]
                    model_raw = path_parts[root_idx + 2]
                except (ValueError, IndexError):
                    continue
                model_clean = clean_model_name(model_raw)
            except Exception:
                continue

            # --- 2) READ SUMMARY JSON ---
            summary_path = os.path.join(root, summary_file)
            k_window = 40
            n_pairs = 0
            total_tokens = 0
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                    k_window = summary.get("params", {}).get("windowk", 40)

                    if "pairing_qc" in summary:
                        n_pairs = summary["pairing_qc"].get("model_lines", 0)
                    if n_pairs == 0 and "qc" in summary:
                        n_pairs = summary["qc"].get("n_pairs", 0)

                    # We assume paired data, so Total Human Tokens ≈ Total AI Tokens
                    total_tokens = n_pairs * k_window if (n_pairs and k_window) else 0
            except Exception as e:
                print(f"❌ Error reading JSON {summary_path}: {e}")

            # --- 3) READ CSV DATA & CALCULATE METRICS ---
            csv_path = os.path.join(root, csv_file)
            rows = []
            n_rows_csv = 0
            n_rows_written = 0
            n_rows_dropped_non_alnum = 0

            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        n_rows_csv += 1

                        # --- NEW: Drop tokens that are purely "special characters" ---
                        form = row.get("form", "")
                        if not has_any_alnum(form):
                            n_rows_dropped_non_alnum += 1
                            continue

                        # Raw counts (critical for LPR + smoothed ratio)
                        raw_count_ai = float(row["c_M"]) if row.get("c_M") else 0.0
                        raw_count_human = float(row["c_H"]) if row.get("c_H") else 0.0

                        # --- SPACE-SAVING FILTER (optional) ---
                        # In compact mode, drop rows where AI count is zero.
                        # (And also drop rows where both are zero.)
                        if mode == "compact":
                            if raw_count_ai == 0.0:
                                continue
                            if raw_count_ai == 0.0 and raw_count_human == 0.0:
                                continue

                        # OPM (Occurrences Per Million) for display
                        if total_tokens > 0:
                            opm_ai = (raw_count_ai / total_tokens) * 1_000_000
                            opm_human = (raw_count_human / total_tokens) * 1_000_000
                        else:
                            opm_ai = 0.0
                            opm_human = 0.0

                        # Smoothed ratio using Jeffreys smoothing on counts.
                        ratio = (raw_count_ai + ratio_smooth) / (raw_count_human + ratio_smooth)

                        # Log Prevalence Ratio (Impact)
                        # Log2( (AI + 1) / (Human + 1) )
                        if raw_count_ai >= min_ai_count_for_impact:
                            smoothed_ai = raw_count_ai + 1.0
                            smoothed_human = raw_count_human + 1.0
                            lpr = math.log2(smoothed_ai / smoothed_human)
                        else:
                            lpr = 0.0

                        las = float(row.get("LAS", 0.0)) if row.get("LAS") else 0.0

                        # Store row (compact keys; ranks assigned later)
                        # Keys:
                        #   w      word (surface form)
                        #   u      UPOS
                        #   las    volume (LAS)
                        #   lpr    impact (LPR)
                        #   a      AI OPM
                        #   h      Human OPM
                        #   r      ratio (smoothed)
                        #   rk_las rank by LAS (desc)
                        #   rk_lpr rank by LPR (desc)
                        rows.append({
                            "w": str(form).strip(),
                            "u": row.get("upos", "UNK"),
                            "las": las,
                            "lpr": lpr,
                            "a": round(opm_ai, 2),
                            "h": round(opm_human, 2),
                            "r": round(ratio, 1),
                            "rk_las": 0,
                            "rk_lpr": 0,
                        })

                # --- 4) MULTI-PASS SORTING & RANKING ---
                # A) LAS ranks (desc)
                rows.sort(key=lambda x: x["las"], reverse=True)
                for i, r in enumerate(rows):
                    r["rk_las"] = i + 1

                # B) LPR ranks (desc)
                rows.sort(key=lambda x: x["lpr"], reverse=True)
                for i, r in enumerate(rows):
                    r["rk_lpr"] = i + 1

                # C) Final sort by LAS rank (default view) & rounding
                rows.sort(key=lambda x: x["rk_las"])
                for r in rows:
                    r["las"] = round(r["las"], 4)
                    r["lpr"] = round(r["lpr"], 4)

                n_rows_written = len(rows)

                # --- 5) SAVE OUTPUT ---
                output_filename = f"{lang}_{register}_{model_clean}.json"
                output_path = os.path.join(output_dir, output_filename)

                # Compact meta keys too (optional, but helps):
                #   np   n_pairs
                #   kw   k_window
                #   tt   total_tokens
                #   src  source_path
                #   md   mode
                #   min  min_ai_count_for_impact
                #   sm   ratio_smooth
                #   n0   n_rows_csv
                #   n1   n_rows_written
                #   nx   rows dropped for non-alnum
                final_data = {
                    "meta": {
                        "np": n_pairs,
                        "kw": k_window,
                        "tt": total_tokens,
                        "src": root,
                        "md": mode,
                        "min": min_ai_count_for_impact,
                        "sm": ratio_smooth,
                        "n0": n_rows_csv,
                        "n1": n_rows_written,
                        "nx": n_rows_dropped_non_alnum,
                    },
                    "data": rows,
                }

                # separators removes whitespace; ensure_ascii=False keeps non-ASCII chars readable
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(final_data, f, ensure_ascii=False, separators=(",", ":"))

                inventory.append({
                    "lang": lang,
                    "register": register,
                    "model": model_clean
                })

                print(
                    f"✅ Generated: {lang.upper()} | {register} | {model_clean} "
                    f"(N={n_pairs}, rows={n_rows_written}/{n_rows_csv}, dropped_non_alnum={n_rows_dropped_non_alnum})"
                )

            except Exception as e:
                print(f"❌ Error processing CSV {csv_path}: {e}")

    # --- 6) INDEX ---
    with open(os.path.join(output_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, separators=(",", ":"))

    print(f"\n🎉 Done! Created {len(inventory)} datasets.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build JSON datasets for LexA-Index website.")
    p.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, help="Root directory containing csv_files/ outputs.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for website JSON files.")
    p.add_argument(
        "--min-ai-count-for-impact",
        type=int,
        default=DEFAULT_MIN_AI_COUNT_FOR_IMPACT,
        help="Minimum AI count (c_M) for a word to get a non-zero impact (LPR).",
    )
    p.add_argument(
        "--mode",
        choices=["full", "compact"],
        default="full",
        help="full = write all rows; compact = drop rows with c_M == 0 to save space.",
    )
    p.add_argument(
        "--ratio-smooth",
        type=float,
        default=DEFAULT_RATIO_SMOOTH,
        help="Additive smoothing constant for ratio=(c_M+smooth)/(c_H+smooth). Default: 0.5 (Jeffreys).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(
        input_root=args.input_root,
        output_dir=args.output_dir,
        min_ai_count_for_impact=args.min_ai_count_for_impact,
        mode=args.mode,
        ratio_smooth=args.ratio_smooth,
    )

