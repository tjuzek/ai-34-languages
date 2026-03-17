# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Annotate AI overuse files with WMT shifts (2020/21 vs 2023/24).
Uses Macro-Averaging for relative frequencies (Yearly Average) to account for corpus size imbalances.

python tools/annotate_wmt_shifts.py \
  --ai_input_dir out/ig/las-gpt4.1-mini-most-pos/policy=quantile_trim=no_M=1_K=12_jitter=no \
  --wmt_counts_dir data/ig/wmt-counts \
  --output_dir out/ig/las-gpt4.1-mini-content-only/wmt-annotated
"""

import argparse
import pandas as pd
import scipy.stats as stats
import sys
import os
from pathlib import Path

def load_wmt_counts_by_year(lang_dir, years):
    """
    Loads counts for each year individually.
    Returns a list of tuples: [(counts_dict, total_tokens), (counts_dict, total_tokens), ...]
    """
    data_per_year = []
    
    for year in years:
        filename = f"{year}_counts.tsv"
        filepath = lang_dir / filename
        
        if not filepath.exists():
            print(f"   [WARN] Missing file: {filename} (skipping)")
            continue
            
        counts = {}
        total_tokens = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 2: continue
                    
                    key = parts[0]
                    try:
                        count = int(parts[1])
                    except ValueError:
                        continue
                        
                    counts[key] = counts.get(key, 0) + count
                    total_tokens += count
            
            # Only add if valid data found
            if total_tokens > 0:
                data_per_year.append((counts, total_tokens))
                
        except Exception as e:
            print(f"   [ERROR] Reading {filename}: {e}")

    return data_per_year

def calculate_chi_square(count_pre, total_pre, count_post, total_post):
    """
    Runs a Chi-Square test for independence on POOLED counts.
    (Statistical tests require raw counts, not averages).
    """
    if total_pre == 0 or total_post == 0:
        return 1.0, False

    table = [
        [count_pre, total_pre - count_pre],
        [count_post, total_post - count_post]
    ]
    
    try:
        # result object available in scipy >= 1.10
        result = stats.chi2_contingency(table)
        p = result.pvalue if hasattr(result, 'pvalue') else result[1]
        
        return p, p < 0.001 # Strict significance threshold
    except:
        return 1.0, False

def get_stats_for_key(key, year_datasets):
    """
    Calculates:
      1. Macro-Average Relative Frequency (for Effect Size)
      2. Pooled Raw Counts (for Statistical Significance)
    """
    if not year_datasets:
        return 0.0, 0, 0

    rel_freqs = []
    pooled_count = 0
    pooled_total = 0

    for counts, total in year_datasets:
        # Count for this specific year
        c = counts.get(key, 0)
        
        # Relative Freq for this year
        rel_freqs.append(c / total)
        
        # Add to pools
        pooled_count += c
        pooled_total += total

    # Macro-Average: Average of the relative frequencies
    avg_rel_freq = sum(rel_freqs) / len(rel_freqs)
    
    return avg_rel_freq, pooled_count, pooled_total

def process_language(lang, ai_file, wmt_lang_dir, output_dir):
    print(f"[{lang}] Processing...")
    
    # 1. Load AI Overuse Data
    try:
        df = pd.read_csv(ai_file)
        if 'key' not in df.columns:
            print(f"   [ERROR] 'key' column missing in {ai_file.name}")
            return
    except Exception as e:
        print(f"   [ERROR] Could not read CSV: {e}")
        return

    # 2. Load WMT Data (Separated by Year)
    # Returns list of (dict, total)
    pre_data = load_wmt_counts_by_year(wmt_lang_dir, ['2020', '2021'])
    post_data = load_wmt_counts_by_year(wmt_lang_dir, ['2023', '2024'])
    
    if not pre_data or not post_data:
        print(f"   [SKIP] WMT data missing for {lang} (needs at least one year in both pre/post)")
        return

    # Info print
    pre_tokens = sum(d[1] for d in pre_data)
    post_tokens = sum(d[1] for d in post_data)
    print(f"   Data Loaded: Pre={pre_tokens:,} tokens ({len(pre_data)} yrs), Post={post_tokens:,} tokens ({len(post_data)} yrs)")

    # 3. Annotate Rows
    wmt_rel_pre = []
    wmt_rel_post = []
    wmt_pct_change = []
    wmt_is_sig = []
    wmt_p_value = []

    for index, row in df.iterrows():
        key = row['key'] 
        
        # --- Pre Period Stats ---
        # avg_pre: The averaged relative frequency (User Request)
        # pool_c_pre: The raw count sum (Required for Chi-Square)
        avg_pre, pool_c_pre, pool_t_pre = get_stats_for_key(key, pre_data)
        
        # --- Post Period Stats ---
        avg_post, pool_c_post, pool_t_post = get_stats_for_key(key, post_data)
        
        # --- Percentage Change (based on Averages) ---
        if avg_pre > 0:
            pct_change = (avg_post - avg_pre) / avg_pre
        else:
            pct_change = 0.0 

        # --- Significance Test (based on Pools) ---
        p_val, is_sig = calculate_chi_square(pool_c_pre, pool_t_pre, pool_c_post, pool_t_post)
        
        wmt_rel_pre.append(avg_pre)
        wmt_rel_post.append(avg_post)
        wmt_pct_change.append(pct_change)
        wmt_p_value.append(p_val)
        wmt_is_sig.append(is_sig)

    # 4. Add Columns
    df['wmt_rel_pre'] = wmt_rel_pre
    df['wmt_rel_post'] = wmt_rel_post
    df['wmt_pct_change'] = wmt_pct_change
    df['wmt_p_value'] = wmt_p_value
    df['wmt_is_sig'] = wmt_is_sig

    # 5. Save Output
    out_lang_dir = Path(output_dir) / lang
    out_lang_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_lang_dir / f"{ai_file.stem}_annotated.csv"
    df.to_csv(out_file, index=False)
    print(f"   -> Saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Annotate AI overuse files with WMT shifts (Macro-Average Method)")
    
    parser.add_argument('--ai_input_dir', type=str, required=True, 
                        help="Base dir containing language folders with las_word_[lang].csv")
    parser.add_argument('--wmt_counts_dir', type=str, required=True, 
                        help="Base dir containing WMT count folders (from count_wmt_frequencies.py)")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Dir to save annotated CSVs")
    
    args = parser.parse_args()
    
    ai_base = Path(args.ai_input_dir)
    wmt_base = Path(args.wmt_counts_dir)
    
    # Iterate over language folders in AI Input Dir
    lang_dirs = [d for d in ai_base.iterdir() if d.is_dir()]
    lang_dirs.sort()
    
    print(f"Found {len(lang_dirs)} languages to process.")
    
    for lang_dir in lang_dirs:
        lang_code = lang_dir.name
        
        # Check if corresponding WMT data exists
        wmt_lang_dir = wmt_base / lang_code
        if not wmt_lang_dir.exists():
            print(f"[{lang_code}] Skipping: No WMT counts found.")
            continue
            
        # Find the CSV file (las_word_*.csv)
        csv_files = list(lang_dir.glob("las_word_*.csv"))
        if not csv_files:
            continue
            
        # Process first matching CSV (usually only one)
        process_language(lang_code, csv_files[0], wmt_lang_dir, args.output_dir)

if __name__ == "__main__":
    main()