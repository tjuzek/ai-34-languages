#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
POS Token vs Model BPE Token Ratio Analyzer

This tool analyzes .pos files (CoNLL-basic format) to determine the ratio between
linguistic "parsed tokens" (words/punctuation) and GPT-4 "model tokens" (BPE).

Methodology:
1. Parsed Token Count: Counts the number of data rows (tokens) in the .pos entry.
2. Model Token Count: Feeds the `# text = ...` metadata field into tiktoken (cl100k_base).
3. Ratio: BPE_Count / Parsed_Count.

Usage:
    python tools/analyse_pos_model_token_ratio.py --input-dir data/pos-basic-sample
"""

import argparse
import sys
import re
import csv
import statistics
import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import tiktoken
except ImportError:
    print("Error: 'tiktoken' module not found. Please install it via 'pip install tiktoken'.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Parsed Token vs BPE Token ratios.")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="data/pos-basic-sample",
        help="Root directory containing language subfolders (e.g., data/pos-basic-sample/en/2020.pos)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="out/logs",
        help="Directory to save the CSV report."
    )
    return parser.parse_args()

def parse_pos_file(file_path: Path, encoder) -> List[Tuple[int, int]]:
    """
    Parses a single .pos file and returns a list of (parsed_count, bpe_count) tuples.
    
    Expected format blocks:
    # sent_id = ...
    # text = Raw text here
    1   Token   lemma   POS
    2   Token   lemma   POS
    ...
    """
    stats = []
    
    current_text = None
    current_parsed_count = 0
    
    # Regex to identify token rows: starts with a digit followed by a tab
    token_row_pattern = re.compile(r'^\d+\t')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 1. Handle Metadata
            if line.startswith("# text ="):
                # If we have a previous block accumulating, process it now
                # (This handles cases where blocks aren't separated by empty lines)
                if current_text is not None and current_parsed_count > 0:
                    bpe_count = len(encoder.encode(current_text))
                    stats.append((current_parsed_count, bpe_count))
                
                # Start new block
                # Extract text after "# text = " (9 chars)
                current_text = line[8:].strip()
                current_parsed_count = 0
                continue
            
            # 2. Handle Token Rows
            if token_row_pattern.match(line):
                current_parsed_count += 1
                continue
            
            # 3. Handle Block Ends (Empty lines or new IDs)
            if not line or line.startswith("# sent_id"):
                if current_text is not None and current_parsed_count > 0:
                    bpe_count = len(encoder.encode(current_text))
                    stats.append((current_parsed_count, bpe_count))
                    
                    # Reset
                    current_text = None
                    current_parsed_count = 0
                continue

        # End of file flush
        if current_text is not None and current_parsed_count > 0:
            bpe_count = len(encoder.encode(current_text))
            stats.append((current_parsed_count, bpe_count))
            
    return stats

def main():
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    
    if not input_root.exists():
        print(f"Error: Input directory '{input_root}' does not exist.")
        sys.exit(1)
        
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Initialize Tokenizer
    print("Loading tokenizer (cl100k_base)...")
    enc = tiktoken.get_encoding("cl100k_base")
    
    print(f"Scanning {input_root}...")
    
    # Data structure: lang_code -> list of ratios
    lang_ratios: Dict[str, List[float]] = defaultdict(list)
    lang_file_counts: Dict[str, int] = defaultdict(int)
    
    # glob all .pos files in subdirectories
    pos_files = sorted(list(input_root.glob("*/*.pos")))
    
    if not pos_files:
        print("No .pos files found. Check your input directory structure.")
        print(f"Expected: {input_root}/[lang]/[year].pos")
        sys.exit(0)
        
    print(f"Found {len(pos_files)} files. Analyzing...")
    
    for file_path in pos_files:
        # Parent folder name is the language code (e.g., .../cs/2012.pos -> cs)
        lang_code = file_path.parent.name
        lang_file_counts[lang_code] += 1
        
        file_stats = parse_pos_file(file_path, enc)
        
        for parsed, bpe in file_stats:
            if parsed > 0:
                ratio = bpe / parsed
                lang_ratios[lang_code].append(ratio)
    
    # --- Generate Report ---
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_root / f"pos_token_ratios_{timestamp}.csv"
    
    print("\n" + "="*85)
    print(f"{'LANG':<6} | {'FILES':<5} | {'SENTENCES':<9} | {'AVG RATIO':<10} | {'STD DEV':<8} | {'MAX RATIO':<9}")
    print(f"{'':<6} | {'':<5} | {'':<9} | {'(BPE/POS)':<10} | {'':<8} | {'':<9}")
    print("="*85)
    
    csv_data = []
    
    # Sort by Average Ratio (descending) to highlight high-fertility languages
    sorted_langs = sorted(
        lang_ratios.keys(), 
        key=lambda l: statistics.mean(lang_ratios[l]) if lang_ratios[l] else 0, 
        reverse=True
    )
    
    for lang in sorted_langs:
        ratios = lang_ratios[lang]
        count = len(ratios)
        n_files = lang_file_counts[lang]
        
        if count > 0:
            avg_ratio = statistics.mean(ratios)
            try:
                stdev = statistics.stdev(ratios)
            except statistics.StatisticsError:
                stdev = 0.0
            max_ratio = max(ratios)
        else:
            avg_ratio = 0
            stdev = 0
            max_ratio = 0
            
        # Print to console
        print(f"{lang:<6} | {n_files:<5} | {count:<9} | {avg_ratio:<10.3f} | {stdev:<8.3f} | {max_ratio:<9.3f}")
        
        csv_data.append({
            "lang": lang,
            "files_analyzed": n_files,
            "sentences_analyzed": count,
            "avg_bpe_per_pos_token": round(avg_ratio, 4),
            "std_dev": round(stdev, 4),
            "min_ratio": round(min(ratios), 4) if ratios else 0,
            "max_ratio": round(max_ratio, 4) if ratios else 0
        })

    print("="*85)
    
    # Save CSV
    if csv_data:
        keys = csv_data[0].keys()
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"\n[INFO] Detailed report saved to: {log_file}")
    else:
        print("\n[WARN] No data found to report.")

if __name__ == "__main__":
    main()
