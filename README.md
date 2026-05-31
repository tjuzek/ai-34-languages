# AI-Associated Lexical Shifts Across 34 Languages

This repository accompanies the paper *AI-Associated Lexical Shifts Across 34 Languages* ([arXiv:2605.25358](https://arxiv.org/abs/2605.25358); the PDF is included here as [`AI-Associated-Lexical-Shifts-34-Languages.pdf`](AI-Associated-Lexical-Shifts-34-Languages.pdf)). Live explorer: [aiwordexplorer.com](https://www.aiwordexplorer.com/).

## Overview

This project quantifies how large language models (LLMs) influence lexical usage across the world's languages. Using a split-halves continuation design on WMT News Crawl data (2012--2024), we compute Log Prevalence Ratio (LPR) metrics to identify words systematically overused by AI relative to human baselines, then track those words diachronically and cross-linguistically.

## Interactive Explorer

The `lexa-index/` directory contains an interactive web tool for exploring AI-overused words across languages. A hosted version is live at [aiwordexplorer.com](https://www.aiwordexplorer.com/).

To run locally:

```bash
cd lexa-index

# Unpack the data
7z x csv_files.7z.001

# Build JSON files from CSVs
python3 build_data.py

# Serve locally
python3 -m http.server

# Open http://localhost:8000/
```

You can also browse overused items directly in the extracted raw `.csv` files under `/lexa-index/csv_files/news/las-gpt4.1-mini/policy=quantile_trim=no_M=1_K=12_jitter=no/`.

## Pipeline Overview

The full pipeline consists of 11 stages, each driven by one or more scripts:

| Stage | Description | Script(s) |
|-------|-------------|-----------|
| 1 | Download WMT News Crawl | `crawl_wmt_news.py` |
| 2 | Filter long lines | `filter_long_lines.py` |
| 3 | Sample (cap at 3M lines/file) | `sample_lines_per_file.py` |
| 4 | POS-tag with Stanza | `tag_with_stanza.py` |
| 5 | Build split-halves pairs | `build_pregen_continuations.py` |
| 6 | Generate AI continuations | `generate_gpt_continuations.py`, `generate_gemini_continuations.py`, `generate_haiku_continuations.py` |
| 7 | Clean generations | `clean_generations.py` |
| 8 | POS-tag AI continuations | `tag_gpt_continuations_stanza.py` |
| 9 | Compute LPR metrics | `analyse_las.py`, `aiwl_las_jsonl.py` |
| 10 | Analysis | `cross_lingual_alignment.py`, `diachronic_opm.py`, `count_wmt_frequencies.py`, `calculate_marketshare_shift.py` |
| 11 | Generate figures | `fig_marketshare.py`, `fig_diachronic.py`, `fig_crosslingual_alignment.py` |

See [COMMANDS.md](COMMANDS.md) for the full set of commands.

## Languages

34 languages are included in the study:

| Code | Language | Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|------|----------|
| af | Afrikaans | fa | Persian | ko | Korean | pt | Portuguese |
| ar | Arabic | fi | Finnish | ky | Kyrgyz | ro | Romanian |
| bg | Bulgarian | fr | French | lt | Lithuanian | ru | Russian |
| cs | Czech | hi | Hindi | lv | Latvian | sr | Serbian |
| de | German | hr | Croatian | mr | Marathi | ta | Tamil |
| el | Greek | id | Indonesian | nl | Dutch | tr | Turkish |
| en | English | is | Icelandic | pl | Polish | uk | Ukrainian |
| es | Spanish | it | Italian | | | zh | Chinese |
| et | Estonian | ja | Japanese | kk | Kazakh | | |

Of these, **10 languages** have diachronic coverage (2012--2024): cs, de, en, es, fr, hi, it, pt, ru, zh.

## Installation

Requires **Python 3.10+**.

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install the package and dependencies
pip install -e .
```

### Additional requirements

- **Stanza models**: Downloaded automatically on first use per language.
- **API keys** (for generation steps only):
  - `OPENAI_API_KEY` -- for GPT-4.1-mini continuations
  - `GEMINI_API_KEY` or `GOOGLE_API_KEY` -- for Gemini continuations (validation)
  - `ANTHROPIC_API_KEY` -- for Haiku continuations (validation)

## Repository Structure

```
aclrepo/
├── README.md                    # This file
├── COMMANDS.md                  # Full pipeline commands
├── pyproject.toml               # Package metadata and dependencies
├── .gitignore
├── scripts/                     # 16 pipeline scripts
│   ├── crawl_wmt_news.py            # Stage 1: Download WMT data
│   ├── filter_long_lines.py         # Stage 2: Remove overly long lines
│   ├── sample_lines_per_file.py     # Stage 3: Cap lines per file
│   ├── tag_with_stanza.py           # Stage 4: POS tagging (Stanza)
│   ├── build_pregen_continuations.py # Stage 5: Build split-halves pairs
│   ├── generate_gpt_continuations.py # Stage 6: GPT continuations
│   ├── generate_gemini_continuations.py # Stage 6: Gemini continuations
│   ├── generate_haiku_continuations.py  # Stage 6: Haiku continuations
│   ├── clean_generations.py         # Stage 7: Clean AI outputs
│   ├── tag_gpt_continuations_stanza.py # Stage 8: POS-tag AI outputs
│   ├── analyse_las.py               # Stage 9: Core LPR engine
│   ├── aiwl_las_jsonl.py            # Stage 9: LPR on JSONL pairs
│   ├── cross_lingual_alignment.py   # Stage 10: Cross-lingual analysis
│   ├── diachronic_opm.py            # Stage 10: Diachronic OPM analysis
│   ├── count_wmt_frequencies.py     # Stage 10: WMT frequency tables
│   └── calculate_marketshare_shift.py # Stage 10: Market share analysis
├── tools/                       # 9 validation/utility scripts
│   ├── calculate_lpr_similarity.py   # Compare LPR rankings (Spearman rho)
│   ├── annotate_wmt_shifts.py        # Annotate WMT frequency shifts
│   ├── analyse_pos_model_token_ratio.py # POS-to-model token ratios
│   ├── check_window_sizes.py        # Line-length distributions
│   ├── compare_relfreq.py           # Compare relative frequencies
│   ├── compute_avg_sent_length.py   # Average sentence length stats
│   ├── list_file_stats.py           # File size/line count inventory
│   ├── parses_len_distribution.py   # Parse length distributions
│   └── spotcheck_continuations.py   # Manual inspection of outputs
├── figures/                     # 3 figure generation scripts
│   ├── fig_crosslingual_alignment.py # Figure 1: cross-lingual bar chart
│   ├── fig_diachronic.py            # Figure 3: diachronic trend lines
│   └── fig_marketshare.py           # Figure 2: market share dot plot
├── config/
│   └── language-specific-prompts.tsv # Language-specific prompt frames
├── meta/
│   ├── file_stats.tsv               # Per-language file size inventory
│   └── language-wmt-stanza/
│       ├── lang_sizes.tsv            # Language data sizes
│       ├── lang_year_pivot.csv       # Year coverage per language
│       ├── wmt_info.txt              # WMT corpus notes
│       └── wmt_languages.txt         # Language list
├── examples/
│   └── en_pregenhalves_2020-2021_example.jsonl  # Small example for testing
├── lexa-index/                  # Interactive AI Word Overuse Explorer
│   ├── index.html                    # Main explorer interface
│   ├── about.html                    # About page
│   ├── build_data.py                 # Build JSON data from CSVs
│   ├── insights.json                 # UI tips metadata
│   ├── csv_files.7z                  # Compressed explorer data (~36MB)
│   ├── favicon.ico
│   ├── social-preview.png
│   ├── README.md
│   └── .gitignore
└── src/aiwl/
    └── __init__.py              # Package marker
```

## Quick Start

Run a minimal example on the included sample data:

```bash
# From the repository root:

# Generate GPT continuations for the example file (requires OPENAI_API_KEY)
python scripts/generate_gpt_continuations.py \
  --input-dir examples \
  --output-dir out/example-continuations \
  --model gpt-4.1-mini \
  --temperature 0.0 --top-p 1.0 \
  --concurrency 4 --rpm 500 -v

# Clean the generations
python scripts/clean_generations.py \
  --input-dir out/example-continuations \
  --output-dir out/example-continuations-cleaned

# POS-tag the cleaned continuations
python scripts/tag_gpt_continuations_stanza.py \
  --input-dir out/example-continuations-cleaned \
  --output-dir out/example-continuations-cleaned-pos
```

## Data

The pipeline processes **WMT News Crawl** data from [https://data.statmt.org/news-crawl/](https://data.statmt.org/news-crawl/).

### Expected directory layout

When running the full pipeline, the following directories are created under the repository root:

```
data/
├── raw/                    # Downloaded WMT text files: {lang}/{year}.txt
├── raw-filtered/           # After filtering long lines
├── raw-filtered-sampled/   # After sampling (cap at 3M lines/file)
├── ig/
│   ├── pos-basic-sample/   # POS-tagged data: {lang}/{year}.pos
│   ├── pre-gen-halves-12/  # Split-halves JSONL pairs
│   ├── gpt4.1-continuations-mini/         # Raw GPT outputs
│   ├── gpt4.1-continuations-mini-cleaned/ # Cleaned GPT outputs
│   └── gpt4.1-continuations-mini-cleaned-pos/ # POS-tagged GPT outputs
```

### Disk space

The full dataset (34 languages, 2012--2024) requires approximately **600+ GB** of storage across all processing stages. The raw WMT download alone is roughly 200 GB.

## Configuration

### Prompt frames

Language-specific prompts for AI continuation generation are defined in `config/language-specific-prompts.tsv`. Each row specifies a language code, prompt template (with `{}` placeholder for the text), a model-to-POS token ratio, and verification status.

### Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | `generate_gpt_continuations.py` | OpenAI API access |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `generate_gemini_continuations.py` | Google Gemini API access |
| `ANTHROPIC_API_KEY` | `generate_haiku_continuations.py` | Anthropic API access |

## Validation

Several robustness checks are provided in `tools/` and documented in COMMANDS.md:

- **Seed robustness**: Re-run with seeds 42--45, compare LPR rankings (Spearman rho > 0.99).
- **Data size sensitivity**: 20k, 50k, 100k, 200k, 500k continuation pairs.
- **Window size sensitivity**: K = 10, 12, 15, 20, 25, 30, 35.
- **Model size**: GPT-4.1-nano, GPT-4.1-mini, GPT-4.1 (full).
- **Model family**: GPT-4.1-mini, Gemini 2.0 Flash, Claude 3 Haiku.
- **LPR similarity**: `tools/calculate_lpr_similarity.py` computes Spearman correlations.

## Figures

All figures can be reproduced from the repository root (after running the relevant analysis steps):

```bash
# Figure 1: Cross-lingual alignment (hardcoded data, no dependencies)
python figures/fig_crosslingual_alignment.py

# Figure 2: Market share shift (requires out/ig/marketshare_analysis-*/marketshare_summary.tsv)
python figures/fig_marketshare.py

# Figure 3: Diachronic trends (requires data/diachronic-rawdata-filtered-sampled-pos/)
python figures/fig_diachronic.py
```

Output: PDF and PNG files in `figures/`.

## Technical Notes

- **`aiwl_las_jsonl.py`** imports from `analyse_las.py` via a bare import. Run from `scripts/` or set `PYTHONPATH=scripts`.
- **`fig_diachronic.py`** expects the current working directory to be the repository root.
- **`fig_marketshare.py`** resolves paths relative to the repository root via `__file__`.
- **`fig_crosslingual_alignment.py`** has hardcoded data (no file reads needed).

## AI Assistance

The code in this repository was developed with AI assistance from GPT (OpenAI), Gemini (Google), and Claude (Anthropic). All AI-generated code was reviewed, tested, and validated by the authors.

Repository polished with Claude Code.

## Citation

If you use this code or data, a citation is appreciated (though not required; see the licence).

```bibtex
@article{juzek-2026-ai-34-languages,
  title   = {AI-Associated Lexical Shifts Across 34 Languages: Cross-Lingual Convergence and Diachronic Uptake in News Writing},
  author  = {Juzek, Thomas Stephan},
  journal = {arXiv preprint arXiv:2605.25358},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.25358},
  url     = {https://arxiv.org/abs/2605.25358}
}
```

## Licence

- **Code:** MIT No Attribution (MIT-0). See [`LICENSE`](LICENSE). Use it freely, no attribution required.
- **Data and word lists:** CC0 1.0 Universal (public domain dedication). See [`LICENSE-DATA`](LICENSE-DATA).

A citation is not required but is appreciated; see the Citation section.

The included paper PDF remains under its own terms (the arXiv preprint, the authors' own), separate from the code and data licences above.
