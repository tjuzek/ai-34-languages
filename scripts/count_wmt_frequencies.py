# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Create WMT frequency tables (lemma_UPOS) per language and year.
Reads .pos files from a specified input directory structure and generates
frequency tables in TSV format.

Example usage:
python scripts/count_wmt_frequencies.py \
  --input_dir data/ig/pos-basic-sample \
  --output_dir data/ig/wmt-counts
"""

import argparse
import os
import sys
from pathlib import Path
from collections import Counter

def process_pos_file(file_path):
    """
    Reads a .pos file and returns a Counter of 'lemma_UPOS'.
    Skipping lines starting with # and handling malformed lines.
    """
    counts = Counter()
    line_num = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                
                # Robustness check: Ensure we have at least 4 columns 
                # (ID, Form, Lemma, UPOS)
                if len(parts) < 4:
                    continue
                
                # Extract Lemma (col 2) and UPOS (col 3)
                # 0-based indexing: 0=ID, 1=Form, 2=Lemma, 3=UPOS
                lemma = parts[2].lower().replace('ё', 'е')
                upos = parts[3]

                # Create the key
                key = f"{lemma}_{upos}"
                counts[key] += 1
                
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path} at line {line_num}: {e}", file=sys.stderr)
        return None

    return counts

def main():
    parser = argparse.ArgumentParser(description="Create WMT frequency tables (lemma_UPOS) per language and year.")
    
    parser.add_argument('--input_dir', type=str, required=True, 
                        help="Base directory containing language folders (e.g., /aiwl/data/ig/pos-basic-sample)")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Target directory to save .tsv frequency tables")
    
    args = parser.parse_args()
    
    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)
    
    if not input_base.exists():
        print(f"Error: Input directory '{input_base}' does not exist.")
        sys.exit(1)

    print(f"Scanning for languages in: {input_base}")
    
    # Get all subdirectories (assumed to be language codes like 'en', 'de', 'zh')
    lang_dirs = [d for d in input_base.iterdir() if d.is_dir()]
    lang_dirs.sort()
    
    print(f"Found {len(lang_dirs)} languages.")
    
    for lang_dir in lang_dirs:
        lang_code = lang_dir.name
        
        # Setup output folder for this language
        lang_out_dir = output_base / lang_code
        lang_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all .pos files in this language folder
        pos_files = sorted(list(lang_dir.glob("*.pos")))
        
        if not pos_files:
            continue
            
        print(f"[{lang_code}] Processing {len(pos_files)} files...")
        
        for pos_file in pos_files:
            # Generate the output filename
            # e.g., '2021.pos' -> '2021_counts.tsv'
            year_stem = pos_file.stem
            out_filename = f"{year_stem}_counts.tsv"
            out_path = lang_out_dir / out_filename
            
            # Skip if already exists? (Optional, currently overwrites)
            # if out_path.exists(): continue 
            
            # Count
            counts = process_pos_file(pos_file)
            
            if counts is not None:
                # Write to TSV
                # Format: lemma_UPOS \t count
                with open(out_path, 'w', encoding='utf-8') as f:
                    # Sorting by frequency descending is usually best for stats
                    for key, count in counts.most_common():
                        f.write(f"{key}\t{count}\n")
                
                print(f"   -> Wrote {out_path.name} ({len(counts)} unique items)")

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
