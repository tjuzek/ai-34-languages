# COMMANDS.md

Full pipeline commands, consolidated from all experiment runs. All paths are relative to the repository root. Run all commands from the repository root unless otherwise noted.

> **Note:** `aiwl_las_jsonl.py` imports from `analyse_las.py` via a bare import. Either run from the `scripts/` directory or set `PYTHONPATH=scripts` before invoking it.

---

## 1. Data Acquisition

### 1.1 Download WMT News Crawl (2020--2024)

```bash
python scripts/crawl_wmt_news.py \
  --outdir data/raw \
  --langs af ar bg cs de el en es et fa fi fr hi hr id is it ja kk ko ky lt lv mr nl pl pt ro ru sr ta tr uk zh \
  --years 2020-2024 \
  --max-workers 4
```

### 1.2 Download Older WMT News (2012--2019, diachronic subset)

```bash
python scripts/crawl_wmt_news.py \
  --outdir data/raw \
  --langs cs de en es fr hi it pt ru zh \
  --years 2012-2019 \
  --max-workers 4
```

---

## 2. Preprocessing

### 2.1 Filter Long Lines

Remove lines exceeding 200 tokens or 1500 characters (which can cause issues with Stanza):

```bash
python scripts/filter_long_lines.py \
  --input-root data/raw \
  --output-root data/raw-filtered
```

### 2.2 Sample Lines

Cap each language/year file at 3 million lines:

```bash
python scripts/sample_lines_per_file.py \
  --input-root data/raw-filtered \
  --output-root data/raw-filtered-sampled \
  --max-lines 3000000 \
  --seed 42
```

---

## 3. POS Tagging (Human Data)

### 3.1 Default (all languages)

```bash
python scripts/tag_with_stanza.py \
  --raw-dir data/raw-filtered-sampled \
  --out-dir data/ig/pos-basic-sample \
  --format basic \
  --chunk-lines 1024 \
  --resume
```

### 3.2 Parallel Execution (2 jobs)

```bash
# JOB 1
python scripts/tag_with_stanza.py --chunk-lines 256 --resume \
  --langs de hi fr es tr bg fa zh pt ta lv uk kk nl mr af &

# JOB 2
python scripts/tag_with_stanza.py --chunk-lines 256 --resume \
  --langs en ru ar cs el sr it ja ko hr id is fi lt ky et &
```

### 3.3 Parallel Execution (4 jobs)

```bash
python scripts/tag_with_stanza.py --chunk-lines 256 --resume --langs de it bg zh hr pl nl lv af et ky &
python scripts/tag_with_stanza.py --chunk-lines 256 --resume --langs en es kk id uk &
python scripts/tag_with_stanza.py --chunk-lines 256 --resume --langs ru fr tr ja ko mr &
python scripts/tag_with_stanza.py --chunk-lines 256 --resume --langs ar cs el sr fa fi is ta &
```

### 3.4 Single Language

```bash
python scripts/tag_with_stanza.py --lang af --input path/to/file.txt
```

---

## 4. Build Split-Halves Pairs

```bash
python scripts/build_pregen_continuations.py \
  --pos-root data/ig/pos-basic-sample \
  --output-dir data/ig/pre-gen-halves-12 \
  --years 2020 2021 \
  --min-half-tokens 12 \
  --max-items-per-lang 100000 \
  --split-rule floor \
  --seed 42 \
  --debug
```

---

## 5. Generate AI Continuations

### 5.1 Primary: GPT-4.1-mini

```bash
export OPENAI_API_KEY="sk-..."

python scripts/generate_gpt_continuations.py \
  --input-dir data/ig/pre-gen-halves-12 \
  --output-dir data/ig/gpt4.1-continuations-mini \
  --model gpt-4.1-mini \
  --temperature 0.0 \
  --top-p 1.0 \
  --concurrency 48 \
  --rpm 4000 \
  -v
```

### 5.2 Validation: Gemini 2.0 Flash

```bash
export GEMINI_API_KEY="..."

python scripts/generate_gemini_continuations.py \
  --input-dir data/ig/validation-models/pre-gen-halves \
  --output-dir data/ig/validation-models/gemini-2.0-flash-continuations \
  --model gemini-2.0-flash-001 \
  --temperature 0.0 \
  --top-p 1.0 \
  --concurrency 6 \
  --rpm 25 \
  -v
```

### 5.3 Validation: Claude 3 Haiku

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python scripts/generate_haiku_continuations.py \
  --input-dir data/ig/validation-models/pre-gen-halves \
  --output-dir data/ig/validation-models/haiku3-continuations \
  --model claude-3-haiku-20240307 \
  --temperature 0.0 \
  --top-p 1.0 \
  --concurrency 48 \
  --rpm 50 \
  -v
```

---

## 6. Clean Generations

```bash
python scripts/clean_generations.py \
  --input-dir data/ig/gpt4.1-continuations-mini \
  --output-dir data/ig/gpt4.1-continuations-mini-cleaned
```

---

## 7. POS-Tag AI Continuations

```bash
python scripts/tag_gpt_continuations_stanza.py \
  --input-dir data/ig/gpt4.1-continuations-mini-cleaned \
  --output-dir data/ig/gpt4.1-continuations-mini-cleaned-pos
```

---

## 8. Compute LPR Metrics

### 8.1 All Words

```bash
PYTHONPATH=scripts python scripts/aiwl_las_jsonl.py \
  --mode discover \
  --human-dir data/ig/pre-gen-halves-12 \
  --gpt-dir data/ig/gpt4.1-continuations-mini-cleaned-pos \
  --out-root out/ig/las-gpt4.1-mini-all-lpr \
  --windowk 12 \
  --exc-upos-calc PUNCT
```

### 8.2 Content Words Only

```bash
PYTHONPATH=scripts python scripts/aiwl_las_jsonl.py \
  --mode discover \
  --human-dir data/ig/pre-gen-halves-12 \
  --gpt-dir data/ig/gpt4.1-continuations-mini-cleaned-pos \
  --out-root out/ig/las-gpt4.1-mini-content-only-lpr \
  --windowk 12 \
  --exc-upos-calc PUNCT \
  --exc-upos-outp ADP AUX CCONJ DET INTJ NUM PART PRON PROPN SCONJ SYM X
```

### 8.3 Function Words Only

```bash
PYTHONPATH=scripts python scripts/aiwl_las_jsonl.py \
  --mode discover \
  --human-dir data/ig/pre-gen-halves-12 \
  --gpt-dir data/ig/gpt4.1-continuations-mini-cleaned-pos \
  --out-root out/ig/las-gpt4.1-mini-no-content-lpr \
  --windowk 12 \
  --exc-upos-calc PUNCT \
  --exc-upos-outp NOUN VERB ADJ ADV
```

### 8.4 No Symbols

```bash
PYTHONPATH=scripts python scripts/aiwl_las_jsonl.py \
  --mode discover \
  --human-dir data/ig/pre-gen-halves-12 \
  --gpt-dir data/ig/gpt4.1-continuations-mini-cleaned-pos \
  --out-root out/ig/las-gpt4.1-mini-nosym-lpr \
  --windowk 12 \
  --exc-upos-calc PUNCT \
  --exc-upos-outp SYM
```

---

## 9. Analysis

### 9.1 WMT Frequency Tables

```bash
python scripts/count_wmt_frequencies.py \
  --input_dir data/ig/pos-basic-sample \
  --output_dir data/ig/wmt-counts
```

### 9.2 Market Share Shift (Content Words, LPR, Top-20)

```bash
python scripts/calculate_marketshare_shift.py \
  --ai_input_dir out/ig/las-gpt4.1-mini-content-only-lpr/policy=quantile_trim=no_M=1_K=12_jitter=no \
  --wmt_counts_dir data/ig/wmt-counts \
  --output_dir out/ig/marketshare_analysis-content-lpr-20 \
  --top_n 20 \
  --baseline_metric avg_wmt \
  --seed 42 \
  --pre_years 2020 2021 \
  --post_years 2023 2024 \
  --top_metric lpr
```

### 9.3 Market Share Shift (All Words, LPR, Top-20)

```bash
python scripts/calculate_marketshare_shift.py \
  --ai_input_dir out/ig/las-gpt4.1-mini-all-lpr/policy=quantile_trim=no_M=1_K=12_jitter=no \
  --wmt_counts_dir data/ig/wmt-counts \
  --output_dir out/ig/marketshare_analysis-lpr-20 \
  --top_n 20 \
  --baseline_metric avg_wmt \
  --seed 42 \
  --pre_years 2020 2021 \
  --post_years 2023 2024 \
  --top_metric lpr
```

### 9.4 Cross-Lingual Alignment

```bash
python scripts/cross_lingual_alignment.py \
  --data-dir out/ig/las-gpt4.1-mini-content-only-lpr/policy=quantile_trim=no_M=1_K=12_jitter=no \
  --out-dir out/ig/cross_lingual_alignment \
  --n-seeds 20 --k-values 50 200 --n-permutations 1000 --seed 42
```

### 9.5 Diachronic OPM Analysis

```bash
python scripts/diachronic_opm.py \
  --data-dir out/ig/las-gpt4.1-mini-content-only-lpr/policy=quantile_trim=no_M=1_K=12_jitter=no \
  --pos-dir data/ig/pos-basic-sample \
  --out-dir out/ig/diachronic_opm \
  --n-seeds 20 --seed 42
```

---

## 10. Validation

### 10.1 Seed Robustness (Seeds 42--45)

Build halves and run the full generate → clean → tag → LPR cycle for each seed:

```bash
for SEED in 43 44 45; do
  python scripts/build_pregen_continuations.py \
    --pos-root data/ig/validation-data-sizes/pos-basic-sample-en \
    --output-dir data/ig/validation-reruns/seed-${SEED} \
    --years 2020 2021 \
    --min-half-tokens 12 \
    --max-items-per-lang 100000 \
    --split-rule floor \
    --seed ${SEED} \
    --debug

  python scripts/generate_gpt_continuations.py \
    --input-dir data/ig/validation-reruns/seed-${SEED} \
    --output-dir data/ig/validation-reruns/gpt4.1-continuations-s${SEED} \
    --model gpt-4.1-mini --temperature 0.0 --top-p 1.0 \
    --concurrency 16 --rpm 1000 -v

  python scripts/clean_generations.py \
    --input-dir data/ig/validation-reruns/gpt4.1-continuations-s${SEED} \
    --output-dir data/ig/validation-reruns/gpt4.1-continuations-s${SEED}-cleaned

  python scripts/tag_gpt_continuations_stanza.py \
    --input-dir data/ig/validation-reruns/gpt4.1-continuations-s${SEED}-cleaned \
    --output-dir data/ig/validation-reruns/gpt4.1-continuations-s${SEED}-cleaned-pos

  PYTHONPATH=scripts python scripts/aiwl_las_jsonl.py \
    --mode discover \
    --human-dir data/ig/validation-reruns/seed-${SEED} \
    --gpt-dir data/ig/validation-reruns/gpt4.1-continuations-s${SEED}-cleaned-pos \
    --out-root out/ig/lpr/validation/seed-reruns/s-${SEED} \
    --windowk 12
done
```

Compare to baseline (seed 42 = primary run at 100k):

```bash
# Identity baseline
python tools/calculate_lpr_similarity.py \
  --f1 out/ig/lpr/validation/data-sizes/n-100k/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
  --f2 out/ig/lpr/validation/data-sizes/n-100k/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
  --out_dir out/ig/validation/seed-reruns/s42-vs-s42/

# Seed 42 vs 43, 44, 45
for SEED in 43 44 45; do
  python tools/calculate_lpr_similarity.py \
    --f1 out/ig/lpr/validation/data-sizes/n-100k/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
    --f2 out/ig/lpr/validation/seed-reruns/s-${SEED}/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
    --out_dir out/ig/validation/seed-reruns/s42-vs-s${SEED}/
done
```

### 10.2 Data Size Sensitivity

Run the full cycle at 20k, 50k, 100k, 200k, 500k pairs:

```bash
for N in 20k 50k 200k 500k; do
  python scripts/build_pregen_continuations.py \
    --pos-root data/ig/validation-data-sizes/pos-basic-sample-en \
    --output-dir data/ig/validation-data-sizes/${N} \
    --years 2020 2021 \
    --min-half-tokens 12 \
    --max-items-per-lang $(echo ${N} | sed 's/k/000/') \
    --split-rule floor --seed 42 --debug

  python scripts/generate_gpt_continuations.py \
    --input-dir data/ig/validation-data-sizes/${N} \
    --output-dir data/ig/validation-data-sizes/gpt4.1-continuations-${N} \
    --model gpt-4.1-mini --temperature 0.0 --top-p 1.0 \
    --concurrency 16 --rpm 1000 -v

  python scripts/clean_generations.py \
    --input-dir data/ig/validation-data-sizes/gpt4.1-continuations-${N} \
    --output-dir data/ig/validation-data-sizes/gpt4.1-continuations-${N}-cleaned

  python scripts/tag_gpt_continuations_stanza.py \
    --input-dir data/ig/validation-data-sizes/gpt4.1-continuations-${N}-cleaned \
    --output-dir data/ig/validation-data-sizes/gpt4.1-continuations-${N}-cleaned-pos

  PYTHONPATH=scripts python scripts/aiwl_las_jsonl.py \
    --mode discover \
    --human-dir data/ig/validation-data-sizes/${N} \
    --gpt-dir data/ig/validation-data-sizes/gpt4.1-continuations-${N}-cleaned-pos \
    --out-root out/ig/lpr/validation/data-sizes/n-${N} \
    --windowk 12
done
```

Compare all sizes to the 100k baseline:

```bash
for N in 20k 50k 200k 500k; do
  python tools/calculate_lpr_similarity.py \
    --f1 out/ig/lpr/validation/data-sizes/n-100k/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
    --f2 out/ig/lpr/validation/data-sizes/n-${N}/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
    --out_dir out/ig/validation/data-sizes/n100k-vs-n${N}/
done
```

### 10.3 Window Size Sensitivity

Run LPR at K = 10, 12, 15, 20, 25, 30, 35:

```bash
for K in 10 12 15 20 25 30 35; do
  PYTHONPATH=scripts python scripts/aiwl_las_jsonl.py \
    --mode discover \
    --human-dir data/ig/validation-window-sizes/pre-gen-halves-en-allyrs-35-100k \
    --gpt-dir data/ig/validation-window-sizes/gpt4.1-continuations-mini-en-allyrs-35-100k-cleaned-pos \
    --out-root out/ig/lpr/validation/window-sizes \
    --windowk ${K}
done
```

Compare all to K=12 baseline:

```bash
for K in 10 15 20 25 30 35; do
  python tools/calculate_lpr_similarity.py \
    --f1 out/ig/lpr/validation/window-sizes/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
    --f2 out/ig/lpr/validation/window-sizes/policy=quantile_trim=no_M=1_K=${K}_jitter=no/en/las_word_en.csv \
    --out_dir out/ig/validation/window-size/K12-vs-K${K}/
done
```

### 10.4 Model Size (GPT-4.1 family: nano, mini, full)

```bash
for MODEL_TAG in nano mini full; do
  MODEL_NAME="gpt-4.1-${MODEL_TAG}"

  python scripts/generate_gpt_continuations.py \
    --input-dir data/ig/validation-model-sizes/pre-gen-halves \
    --output-dir data/ig/validation-model-sizes/gpt4.1-continuations-${MODEL_TAG} \
    --model ${MODEL_NAME} --temperature 0.0 --top-p 1.0 \
    --concurrency 32 --rpm 2500 -v

  python scripts/clean_generations.py \
    --input-dir data/ig/validation-model-sizes/gpt4.1-continuations-${MODEL_TAG} \
    --output-dir data/ig/validation-model-sizes/gpt4.1-continuations-${MODEL_TAG}-cleaned

  python scripts/tag_gpt_continuations_stanza.py \
    --input-dir data/ig/validation-model-sizes/gpt4.1-continuations-${MODEL_TAG}-cleaned \
    --output-dir data/ig/validation-model-sizes/gpt4.1-continuations-${MODEL_TAG}-cleaned-pos

  PYTHONPATH=scripts python scripts/aiwl_las_jsonl.py \
    --mode discover \
    --human-dir data/ig/validation-model-sizes/pre-gen-halves \
    --gpt-dir data/ig/validation-model-sizes/gpt4.1-continuations-${MODEL_TAG}-cleaned-pos \
    --out-root out/ig/lpr/validation/model-size/gpt4.1-${MODEL_TAG} \
    --windowk 12
done
```

Compare sizes to mini baseline:

```bash
for TAG in nano full; do
  python tools/calculate_lpr_similarity.py \
    --f1 out/ig/lpr/validation/model-size/gpt4.1-mini/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
    --f2 out/ig/lpr/validation/model-size/gpt4.1-${TAG}/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
    --out_dir out/ig/validation/model-size/gpt4.1-mini-vs-gpt4.1-${TAG}/
done
```

### 10.5 Model Family (GPT-4.1-mini, Gemini 2.0 Flash, Claude 3 Haiku)

Generate with each model, then clean → tag → LPR:

```bash
# GPT-4.1-mini (primary)
python scripts/generate_gpt_continuations.py \
  --input-dir data/ig/validation-models/pre-gen-halves \
  --output-dir data/ig/validation-models/gpt4.1-continuations-mini \
  --model gpt-4.1-mini --temperature 0.0 --top-p 1.0 \
  --concurrency 48 --rpm 6000 -v

# Gemini 2.0 Flash
python scripts/generate_gemini_continuations.py \
  --input-dir data/ig/validation-models/pre-gen-halves \
  --output-dir data/ig/validation-models/gemini-2.0-flash-continuations \
  --model gemini-2.0-flash-001 --temperature 0.0 --top-p 1.0 \
  --concurrency 6 --rpm 25 -v

# Claude 3 Haiku
python scripts/generate_haiku_continuations.py \
  --input-dir data/ig/validation-models/pre-gen-halves \
  --output-dir data/ig/validation-models/haiku3-continuations \
  --model claude-3-haiku-20240307 --temperature 0.0 --top-p 1.0 \
  --concurrency 48 --rpm 50 -v

# GPT-4.1-mini second run (API non-determinism check)
python scripts/generate_gpt_continuations.py \
  --input-dir data/ig/validation-models/pre-gen-halves \
  --output-dir data/ig/validation-models/gpt4.1-continuations-mini-2 \
  --model gpt-4.1-mini --temperature 0.0 --top-p 1.0 \
  --concurrency 48 --rpm 6000 -v
```

Clean, tag, and compute LPR for each:

```bash
for DIR_PAIR in \
  "gpt4.1-continuations-mini:gpt-4-1-mini" \
  "gpt4.1-continuations-mini-2:gpt-4-1-mini-second-run" \
  "gemini-2.0-flash-continuations:gemini" \
  "haiku3-continuations:haiku"; do

  SRC="${DIR_PAIR%%:*}"
  LABEL="${DIR_PAIR##*:}"

  python scripts/clean_generations.py \
    --input-dir data/ig/validation-models/${SRC} \
    --output-dir data/ig/validation-models/${SRC}-cleaned

  python scripts/tag_gpt_continuations_stanza.py \
    --input-dir data/ig/validation-models/${SRC}-cleaned \
    --output-dir data/ig/validation-models/${SRC}-cleaned-pos

  PYTHONPATH=scripts python scripts/aiwl_las_jsonl.py \
    --mode discover \
    --human-dir data/ig/validation-models/pre-gen-halves \
    --gpt-dir data/ig/validation-models/${SRC}-cleaned-pos \
    --out-root out/ig/lpr/validation/model-family/${LABEL} \
    --windowk 12
done
```

Compare model families:

```bash
python tools/calculate_lpr_similarity.py \
  --f1 out/ig/lpr/validation/model-family/gpt-4-1-mini/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
  --f2 out/ig/lpr/validation/model-family/gemini/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
  --out_dir out/ig/validation/model-family/gpt-4-1-mini-vs-gemini/

python tools/calculate_lpr_similarity.py \
  --f1 out/ig/lpr/validation/model-family/gpt-4-1-mini/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
  --f2 out/ig/lpr/validation/model-family/haiku/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
  --out_dir out/ig/validation/model-family/gpt-4-1-mini-vs-haiku/

# Note: gpt-4-1-mini-second-run measures API-level non-determinism (rho=0.995),
# not a model family difference.
python tools/calculate_lpr_similarity.py \
  --f1 out/ig/lpr/validation/model-family/gpt-4-1-mini/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
  --f2 out/ig/lpr/validation/model-family/gpt-4-1-mini-second-run/policy=quantile_trim=no_M=1_K=12_jitter=no/en/las_word_en.csv \
  --out_dir out/ig/validation/model-family/gpt-4-1-mini-vs-gpt-4-1-mini-second-run/
```

---

## 11. Figures

```bash
# Figure 1: Cross-lingual alignment (hardcoded data, runs instantly)
python figures/fig_crosslingual_alignment.py

# Figure 2: Market share shift
python figures/fig_marketshare.py

# Figure 3: Diachronic trends (run from repo root)
python figures/fig_diachronic.py
```

Output: `figures/fig_*.pdf` and `figures/fig_*.png`.
