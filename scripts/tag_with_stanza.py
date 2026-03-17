#!/usr/bin/env python
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Minimal Stanza-based POS/UD tagger for AIWL.

Modes
-----

1. Single-file / stdin mode (manual use, similar to original):

   Tag an input file (English):

       python scripts/tag_with_stanza.py --lang en --input my_text.txt > my_text.pos

   Tag text from standard input:

       echo "In this study, we formulate a clear hypothesis." \
           | python scripts/tag_with_stanza.py --lang en

   The output format can be chosen via:

       --format basic   # ID, FORM, LEMMA, UPOS (+ sent_id/text comments)
       --format conllu  # full 10-column CoNLL-U

   (In single-file / stdin mode, chunking is ignored; we just process the
    whole text in one go, as before.)

2. Batch mode (no --input and no stdin):

   If invoked without --input and no stdin is piped, the script:

   - Finds language subfolders under /aiwl/data/raw (relative to this script),
   - For each language code L (e.g. af, ky), tags all *.txt files under:
       /aiwl/data/raw/L/
   - Treats each non-empty line in a file as a separate "document"
     (suitable for sentence-per-line corpora),
   - Writes tagged output to:
       /aiwl/data/pos-basic/L/FILE.pos        (for --format basic)
       /aiwl/data/pos-conllu/L/FILE.conllu    (for --format conllu)

   Optional language selection:

       --langs fa id ja ko ru uk

   If --langs is provided, only those language subfolders are processed.
   If omitted, all language subfolders under --raw-dir are processed.

   Optional GPU-oriented chunking:

       --chunk-lines N   (batch mode only)

   If --chunk-lines is not set or <= 1:
       - Behaviour is exactly as before: one line → one nlp() call.

   If --chunk-lines N with N >= 2:
       - Non-empty lines are grouped into batches of up to N lines.
       - Each batch is converted into a list of stanza.Document objects and
         passed into the pipeline in one call (multi-document mode).
       - Output order and sent_id structure stay the same.

   Example:

       cd /aiwl
       python scripts/tag_with_stanza.py             # fresh run, overwrite outputs
       python scripts/tag_with_stanza.py --resume    # continue, skip existing outputs
       python scripts/tag_with_stanza.py --chunk-lines 64  # GPU-friendly batching
       python scripts/tag_with_stanza.py \
         --raw-dir data/raw-sample-filtered \
         --out-dir data/pos-basic-sample \
         --format basic \
         --chunk-lines 512 \
         --langs fa id ja ko ru uk

GPU / CPU
---------

- The script tries to use GPU if available (torch+CUDA), else CPU.
- A small env tweak is applied to work around PyTorch 2.6+ `weights_only`
  changes that otherwise break Stanza model loading.

Logging
-------

- Console (stderr): prints only that another ~1000 sentences have been processed
  (plus final counts per file).
- Log files (under PROJECT_ROOT/out/logs/):

  * throughput.tsv:
      timestamp  lang  file  line_count  processed_mib  elapsed_min
      throughput_mib_per_min  device

  * gpu_usage.tsv (only if GPU is used and nvidia-smi is available):
      timestamp  lang  gpu_index  name  util_gpu_pct  mem_used_mb  mem_total_mb

  * bad_lines.tsv:
      timestamp  lang  file  line_no  text_len  error
    (lines that could not be tagged, e.g. too long or OOM)

Outputs
-------

- In *basic* mode (batch):

    # sent_id = <lang>-<file-stem>-L<line>-s<N>
    # text = <sentence text>
    1   Token   lemma   UPOS
    2   ...
    <blank line between sentences>

- In *basic* mode (single-file / stdin):

    # sent_id = <lang>-<file-stem>-s<N>   (or <lang>-stdin-s<N>)
    # text = <sentence text>
    ...

- In *conllu* mode:

    # sent_id = <lang>-<file-stem>-L<line>-s<N>   (batch)
    # text = <sentence text>
    [optional MWT lines if the language has them]
    ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
"""

import os
import argparse
import sys
import time
import gc
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, TextIO

# ---------------------------------------------------------------------------
# Work around PyTorch 2.6+ weights_only=True default, which breaks Stanza's
# torch.load calls on older model checkpoints.
# This must be set BEFORE torch/stanza are imported.
# ---------------------------------------------------------------------------
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import stanza  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None

# ---------- Paths & constants ----------

# Assume repository structure:
# /aiwl/
#   scripts/tag_with_stanza.py
#   data/raw/<lang>/...
#   data/pos-basic/<lang>/...
#   data/pos-conllu/<lang>/...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_ROOT_BASIC = PROJECT_ROOT / "data" / "pos-basic"
DEFAULT_OUT_ROOT_CONLLU = PROJECT_ROOT / "data" / "pos-conllu"

# Logging paths (for throughput & GPU usage)
LOG_ROOT = PROJECT_ROOT / "out" / "logs"
THROUGHPUT_LOG_PATH = LOG_ROOT / "throughput.tsv"
GPU_LOG_PATH = LOG_ROOT / "gpu_usage.tsv"
BAD_LINES_LOG_PATH = LOG_ROOT / "bad_lines.tsv"
_LOG_DIR_INITIALISED = False

# Use a POS/lemma pipeline that works for more languages (no mwt/depparse).
STANZA_PROCESSORS = "tokenize,pos,lemma"

# Flush output after this many processed sentences (in batch/file mode)
FLUSH_EVERY_SENTENCES = 1000

# Heuristic safeguard: skip absurdly long "documents" (lines).
# You can override via env var AIWL_MAX_DOC_CHARS if you like.
MAX_CHARS_PER_DOC = int(os.environ.get("AIWL_MAX_DOC_CHARS", "20000"))

# Cache of initialized pipelines (used only when cache=True)
_PIPELINES: Dict[str, stanza.Pipeline] = {}


# ---------- GPU / Pipeline handling ----------

def _detect_use_gpu() -> bool:
    """Return True if a GPU is available and torch is installed, else False."""
    if torch is None:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def load_pipeline(lang: str, use_gpu: bool, cache: bool = True) -> stanza.Pipeline:
    """
    Load a Stanza pipeline for the given language.

    If cache=True, keep a cached copy (used for single-file/stdin mode).
    If cache=False, always create a fresh pipeline (used in batch mode
    to avoid accumulating multiple language models in memory).
    """
    key = f"{lang}_{'gpu' if use_gpu else 'cpu'}"
    if cache and key in _PIPELINES:
        return _PIPELINES[key]

    # Download models if not present; idempotent
    stanza.download(lang, processors=STANZA_PROCESSORS)

    nlp = stanza.Pipeline(
        lang=lang,
        processors=STANZA_PROCESSORS,
        use_gpu=use_gpu,
    )

    if cache:
        _PIPELINES[key] = nlp

    return nlp


# ---------- Logging helpers ----------

def _ensure_log_dirs_and_headers(use_gpu: bool) -> None:
    """Create log directory and headers if needed."""
    global _LOG_DIR_INITIALISED
    if _LOG_DIR_INITIALISED:
        return

    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    if not THROUGHPUT_LOG_PATH.exists():
        THROUGHPUT_LOG_PATH.write_text(
            "timestamp\tlang\tfile\tline_count\tprocessed_mib\telapsed_min\t"
            "throughput_mib_per_min\tdevice\n",
            encoding="utf-8",
        )

    if use_gpu and not GPU_LOG_PATH.exists():
        GPU_LOG_PATH.write_text(
            "timestamp\tlang\tgpu_index\tname\tutil_gpu_pct\tmem_used_mb\tmem_total_mb\n",
            encoding="utf-8",
        )

    if not BAD_LINES_LOG_PATH.exists():
        BAD_LINES_LOG_PATH.write_text(
            "timestamp\tlang\tfile\tline_no\ttext_len\terror\n",
            encoding="utf-8",
        )

    _LOG_DIR_INITIALISED = True


def _log_gpu_usage(lang: str) -> None:
    """
    Append current GPU utilisation to GPU_LOG_PATH using nvidia-smi.

    Columns (TSV):
        timestamp, lang, gpu_index, name, util_gpu_pct, mem_used_mb, mem_total_mb
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception:
        # If nvidia-smi is unavailable or errors, just skip GPU logging.
        return

    out = result.stdout.strip()
    if not out:
        return

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return

    try:
        with GPU_LOG_PATH.open("a", encoding="utf-8") as f:
            for ln in lines:
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) != 6:
                    continue
                timestamp, idx, name, util, mem_used, mem_total = parts
                f.write(
                    f"{timestamp}\t{lang}\t{idx}\t{name}\t{util}\t{mem_used}\t{mem_total}\n"
                )
    except Exception:
        # Logging should never crash tagging
        return


def _log_progress(
    line_count: int,
    processed_bytes: int,
    start_time: float,
    file_label: str,
    use_gpu: bool,
    lang: str,
) -> None:
    """
    Log processed lines and average throughput in MiB/min since `start_time`.

    Console (stderr): only prints that another block of ~1000 has been done.
    Logs: append throughput (and GPU usage if available) to TSV files under out/logs/.
    """
    _ensure_log_dirs_and_headers(use_gpu)

    elapsed = time.time() - start_time
    safe_lang = lang or "NA"

    if elapsed <= 0 or processed_bytes <= 0:
        # Minimal console message
        sys.stderr.write(
            f"[INFO]     Processed {line_count} sentences in {file_label}\n"
        )
        sys.stderr.flush()

        # Still log a row with zero throughput
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            with THROUGHPUT_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(
                    f"{timestamp_str}\t{safe_lang}\t{file_label}\t{line_count}\t"
                    f"0.0000\t0.0000\t0.0000\t"
                    f"{'GPU' if use_gpu else 'CPU'}\n"
                )
        except Exception:
            pass

        # GPU usage (if applicable)
        if use_gpu:
            _log_gpu_usage(safe_lang)
        return

    mib = processed_bytes / (1024 * 1024)
    mins = elapsed / 60.0
    throughput = mib / mins if mins > 0 else 0.0

    # Console: just "another 1000" style message
    sys.stderr.write(
        f"[INFO]     Processed {line_count} sentences in {file_label}\n"
    )
    sys.stderr.flush()

    # Throughput log
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        with THROUGHPUT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(
                f"{timestamp_str}\t{safe_lang}\t{file_label}\t{line_count}\t"
                f"{mib:.4f}\t{mins:.4f}\t{throughput:.4f}\t"
                f"{'GPU' if use_gpu else 'CPU'}\n"
            )
    except Exception:
        # Don't let logging crash the run
        pass

    # GPU usage log (if on GPU)
    if use_gpu:
        _log_gpu_usage(safe_lang)


def _log_bad_line(
    lang: str,
    file_label: str,
    line_no: int,
    text_len: int,
    error_msg: str,
) -> None:
    """
    Log a line that could not be tagged (e.g. due to OOM or other errors).

    We do NOT store the actual text here to keep the log small and avoid
    leaking content; you can recover it from the original file and line_no.
    """
    _ensure_log_dirs_and_headers(use_gpu=False)

    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    safe_err = error_msg.replace("\t", " ").replace("\n", " ")

    try:
        with BAD_LINES_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(
                f"{timestamp_str}\t{lang}\t{file_label}\t"
                f"{line_no}\t{text_len}\t{safe_err}\n"
            )
    except Exception:
        # Logging should never crash tagging.
        pass


# ---------- CoNLL-style writers ----------

def _sentence_text(sent) -> str:
    """Return sentence text; fall back to joining tokens if needed."""
    if getattr(sent, "text", None):
        return sent.text
    return " ".join(w.text for w in sent.words)


def write_doc_basic(
    doc: stanza.Document,
    out_f: TextIO,
    sent_id_prefix: str,
    flush_every: Optional[int] = None,
) -> None:
    """
    Write a Stanza Document in a simplified CoNLL-ish format:

        # sent_id = <prefix>-sN
        # text = ...
        1   FORM   LEMMA   UPOS
        ...

    Columns: ID, FORM, LEMMA, UPOS
    """
    sent_counter = 0
    for i, sent in enumerate(doc.sentences, start=1):
        sent_id = f"{sent_id_prefix}-s{i}"
        text = _sentence_text(sent)

        out_f.write(f"# sent_id = {sent_id}\n")
        out_f.write(f"# text = {text}\n")

        for wid, word in enumerate(sent.words, start=1):
            form = word.text
            lemma = word.lemma if word.lemma is not None else "_"
            upos = word.upos if word.upos is not None else "_"
            out_f.write(f"{wid}\t{form}\t{lemma}\t{upos}\n")

        out_f.write("\n")

        sent_counter += 1
        if flush_every and sent_counter % flush_every == 0:
            out_f.flush()

    out_f.flush()


def write_doc_conllu(
    doc: stanza.Document,
    out_f: TextIO,
    sent_id_prefix: str,
    flush_every: Optional[int] = None,
) -> None:
    """
    Write a Stanza Document in full CoNLL-U:

        # sent_id = <prefix>-sN
        # text = ...
        [optionally MWT lines if the language has them]
        ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC

    We construct the 10 columns from Stanza word attributes.
    If some attributes (e.g. HEAD/DEPREL) are missing because the pipeline
    does not include depparse, they are filled with defaults (0 / "_").
    """
    sent_counter = 0
    for i, sent in enumerate(doc.sentences, start=1):
        sent_id = f"{sent_id_prefix}-s{i}"
        text = _sentence_text(sent)

        out_f.write(f"# sent_id = {sent_id}\n")
        out_f.write(f"# text = {text}\n")

        # Multi-word tokens only if the language/pipeline supports them.
        for token in sent.tokens:
            token_ids = token.id
            if isinstance(token_ids, list) and len(token_ids) > 1:
                first_id = token_ids[0]
                last_id = token_ids[-1]
                out_f.write(
                    f"{first_id}-{last_id}\t{token.text}\t_\t_\t_\t_\t_\t_\t_\t_\n"
                )

        # Actual word lines
        for word in sent.words:
            wid = word.id
            form = word.text
            lemma = word.lemma if word.lemma is not None else "_"
            upos = word.upos if word.upos is not None else "_"
            xpos = word.xpos if word.xpos is not None else "_"
            feats = word.feats if word.feats is not None else "_"
            head = word.head if word.head is not None else 0
            deprel = word.deprel if word.deprel is not None else "_"
            deps = word.deps if getattr(word, "deps", None) else "_"
            misc = word.misc if word.misc is not None else "_"

            out_f.write(
                f"{wid}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t"
                f"{head}\t{deprel}\t{deps}\t{misc}\n"
            )

        out_f.write("\n")

        sent_counter += 1
        if flush_every and sent_counter % flush_every == 0:
            out_f.flush()

    out_f.flush()


# ---------- Core tagging helpers ----------

def tag_text_to_stream(
    text: str,
    nlp: stanza.Pipeline,
    out_f: TextIO,
    output_format: str,
    sent_id_prefix: str,
    flush_every: Optional[int] = None,
) -> None:
    """Run the pipeline on text and write either basic or full CoNLL-U."""
    doc = nlp(text)

    if output_format == "basic":
        write_doc_basic(doc, out_f, sent_id_prefix, flush_every=flush_every)
    else:
        write_doc_conllu(doc, out_f, sent_id_prefix, flush_every=flush_every)


def _process_line_batch(
    nlp: stanza.Pipeline,
    lang: str,
    file_stem: str,
    file_label: str,
    batch_line_nos: List[int],
    batch_texts: List[str],
    out_f: TextIO,
    output_format: str,
    line_count_so_far: int,
) -> int:
    """
    Process a batch of lines using Stanza's multi-document interface.

    - Each text in `batch_texts` is wrapped as stanza.Document([], text=...).
    - We call nlp(doc_list) once.
    - For each resulting Document, we write out with a sent_id_prefix that
      encodes the original line number.

    Returns updated line_count.

    Robustness:
    - If the batch call fails (e.g. OOM), we fall back to per-line tagging.
    - If a specific line still fails, we log it and skip it.
    """
    if not batch_texts:
        return line_count_so_far

    try:
        in_docs = [stanza.Document([], text=t) for t in batch_texts]
        out_docs = nlp(in_docs)
    except Exception as e:
        # Batch failed (often OOM). Try per-line fallback; skip hopeless ones.
        sys.stderr.write(
            f"[ERROR] Batch tagging failed in {file_label} "
            f"(lines {batch_line_nos[0]}–{batch_line_nos[-1]}): {e}\n"
        )
        sys.stderr.write(
            "[INFO]   Falling back to line-by-line tagging for this batch.\n"
        )
        sys.stderr.flush()

        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        for line_no, text in zip(batch_line_nos, batch_texts):
            try:
                doc = nlp(text)
            except Exception as e_line:
                sys.stderr.write(
                    f"[ERROR] Failed to tag line {line_no} in {file_label}: {e_line}\n"
                )
                sys.stderr.flush()
                _log_bad_line(
                    lang=lang,
                    file_label=file_label,
                    line_no=line_no,
                    text_len=len(text),
                    error_msg=str(e_line),
                )
                if torch is not None:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                continue

            sent_id_prefix = f"{lang}-{file_stem}-L{line_no}"
            if output_format == "basic":
                write_doc_basic(doc, out_f, sent_id_prefix, flush_every=None)
            else:
                write_doc_conllu(doc, out_f, sent_id_prefix, flush_every=None)

            line_count_so_far += 1

        return line_count_so_far

    # Normal, successful batched path
    for line_no, doc in zip(batch_line_nos, out_docs):
        sent_id_prefix = f"{lang}-{file_stem}-L{line_no}"
        if output_format == "basic":
            write_doc_basic(doc, out_f, sent_id_prefix, flush_every=None)
        else:
            write_doc_conllu(doc, out_f, sent_id_prefix, flush_every=None)

        line_count_so_far += 1

    return line_count_so_far


def tag_file_to_disk(
    input_path: Path,
    output_path: Path,
    nlp: stanza.Pipeline,
    lang: str,
    output_format: str,
    chunk_lines: Optional[int] = None,
    use_gpu: bool = False,
) -> None:
    """
    Tag a single text file and write either basic or full CoNLL-U.

    Batch mode behaviour (for large corpora):

    - Process the file line by line.
    - Each non-empty line is treated as a separate document.
    - sent_id encodes the line number:

        <lang>-<file-stem>-L<line>-s<N>

    If chunk_lines is None or <= 1:
        - Process exactly one line per nlp() call (baseline behaviour).

    If chunk_lines >= 2:
        - Group lines into batches of up to `chunk_lines` and process them
          together using Stanza's multi-document interface.

    Robustness:
    - Lines longer than MAX_CHARS_PER_DOC are skipped and logged.
    - Any per-line tagging error is caught, logged, and skipped.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_stem = input_path.stem
    file_label = str(input_path)
    line_count = 0
    processed_bytes = 0
    start_time = time.time()

    with input_path.open("r", encoding="utf-8", errors="ignore") as in_f, \
            output_path.open("w", encoding="utf-8") as out_f:

        # --- No chunking: original per-line behaviour ---
        if chunk_lines is None or chunk_lines <= 1:
            for line_no, line in enumerate(in_f, start=1):
                text = line.strip()
                if not text:
                    continue

                # Skip absurdly long "documents" defensively
                if len(text) > MAX_CHARS_PER_DOC:
                    sys.stderr.write(
                        f"[WARN] Skipping line {line_no} in {input_path} "
                        f"({len(text)} chars > MAX_CHARS_PER_DOC={MAX_CHARS_PER_DOC}).\n"
                    )
                    sys.stderr.flush()
                    _log_bad_line(
                        lang=lang,
                        file_label=file_label,
                        line_no=line_no,
                        text_len=len(text),
                        error_msg="too_long",
                    )
                    continue

                # Count bytes of non-empty lines
                processed_bytes += len(line.encode("utf-8", errors="ignore"))

                sent_id_prefix = f"{lang}-{file_stem}-L{line_no}"

                try:
                    doc = nlp(text)
                except Exception as e:
                    sys.stderr.write(
                        f"[ERROR] Failed to tag line {line_no} in {input_path}: {e}\n"
                    )
                    sys.stderr.flush()
                    _log_bad_line(
                        lang=lang,
                        file_label=file_label,
                        line_no=line_no,
                        text_len=len(text),
                        error_msg=str(e),
                    )
                    if use_gpu and torch is not None:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    # Skip this line, continue with the rest of the file
                    continue

                if output_format == "basic":
                    write_doc_basic(doc, out_f, sent_id_prefix, flush_every=None)
                else:
                    write_doc_conllu(doc, out_f, sent_id_prefix, flush_every=None)

                line_count += 1
                if line_count % FLUSH_EVERY_SENTENCES == 0:
                    out_f.flush()
                    _log_progress(
                        line_count=line_count,
                        processed_bytes=processed_bytes,
                        start_time=start_time,
                        file_label=file_label,
                        use_gpu=use_gpu,
                        lang=lang,
                    )

            out_f.flush()
            if line_count == 0:
                sys.stderr.write(
                    f"[WARN] {input_path} contained no non-empty lines; "
                    f"wrote nothing.\n"
                )
                sys.stderr.flush()
            else:
                # Final summary for this file
                _log_progress(
                    line_count=line_count,
                    processed_bytes=processed_bytes,
                    start_time=start_time,
                    file_label=file_label,
                    use_gpu=use_gpu,
                    lang=lang,
                )
            return

        # --- Chunked mode: batch multiple lines together ---
        batch_nos: List[int] = []
        batch_texts: List[str] = []

        for line_no, line in enumerate(in_f, start=1):
            text = line.strip()
            if not text:
                continue

            if len(text) > MAX_CHARS_PER_DOC:
                sys.stderr.write(
                    f"[WARN] Skipping line {line_no} in {input_path} "
                    f"({len(text)} chars > MAX_CHARS_PER_DOC={MAX_CHARS_PER_DOC}).\n"
                )
                sys.stderr.flush()
                _log_bad_line(
                    lang=lang,
                    file_label=file_label,
                    line_no=line_no,
                    text_len=len(text),
                    error_msg="too_long",
                )
                continue

            processed_bytes += len(line.encode("utf-8", errors="ignore"))

            batch_nos.append(line_no)
            batch_texts.append(text)

            if len(batch_texts) >= chunk_lines:
                line_count = _process_line_batch(
                    nlp=nlp,
                    lang=lang,
                    file_stem=file_stem,
                    file_label=file_label,
                    batch_line_nos=batch_nos,
                    batch_texts=batch_texts,
                    out_f=out_f,
                    output_format=output_format,
                    line_count_so_far=line_count,
                )
                batch_nos = []
                batch_texts = []

                if line_count % FLUSH_EVERY_SENTENCES == 0:
                    out_f.flush()
                    _log_progress(
                        line_count=line_count,
                        processed_bytes=processed_bytes,
                        start_time=start_time,
                        file_label=file_label,
                        use_gpu=use_gpu,
                        lang=lang,
                    )

        # Leftover batch
        if batch_texts:
            line_count = _process_line_batch(
                nlp=nlp,
                lang=lang,
                file_stem=file_stem,
                file_label=file_label,
                batch_line_nos=batch_nos,
                batch_texts=batch_texts,
                out_f=out_f,
                output_format=output_format,
                line_count_so_far=line_count,
            )
            if line_count % FLUSH_EVERY_SENTENCES == 0:
                out_f.flush()
                _log_progress(
                    line_count=line_count,
                    processed_bytes=processed_bytes,
                    start_time=start_time,
                    file_label=file_label,
                    use_gpu=use_gpu,
                    lang=lang,
                )

        out_f.flush()
        if line_count == 0:
            sys.stderr.write(
                f"[WARN] {input_path} contained no non-empty lines; "
                f"wrote nothing.\n"
            )
            sys.stderr.flush()
        else:
            _log_progress(
                line_count=line_count,
                processed_bytes=processed_bytes,
                start_time=start_time,
                file_label=file_label,
                use_gpu=use_gpu,
                lang=lang,
            )


# ---------- Batch processing ----------

def batch_tag_corpus(
    raw_root: Path,
    out_root: Path,
    use_gpu: bool,
    output_format: str,
    resume: bool = False,
    chunk_lines: Optional[int] = None,
    target_langs: Optional[List[str]] = None,
) -> None:
    """
    Batch mode:

    - Iterate over language subfolders under raw_root,
    - Optionally restrict to a subset of language codes via `target_langs`,
    - For each *.txt file, create a corresponding output file in out_root,
      processing each file line-by-line to handle large corpora.

    If resume=True, skip files whose output already exists.
    If resume=False, always re-tag and overwrite outputs (fresh run).

    If chunk_lines is not None and >= 2, lines are processed in line-batches
    of that size using Stanza's multi-document mode.
    """
    if not raw_root.exists():
        sys.stderr.write(f"[ERROR] Raw data directory does not exist: {raw_root}\n")
        sys.exit(1)

    # Discover all language subfolders
    all_lang_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])

    # Optionally restrict to target languages
    if target_langs:
        # Normalise to lowercase, deduplicate
        target_langs_norm = sorted({lang.strip().lower() for lang in target_langs if lang.strip()})
        sys.stderr.write(
            "[INFO] Restricting to languages: "
            f"{', '.join(target_langs_norm)}\n"
        )
        sys.stderr.flush()

        target_set = set(target_langs_norm)
        langs = [p for p in all_lang_dirs if p.name.lower() in target_set]

        found_codes = {p.name.lower() for p in langs}
        missing = sorted(target_set - found_codes)
        if missing:
            sys.stderr.write(
                f"[WARN] The following requested languages were not found under "
                f"{raw_root}: {', '.join(missing)}\n"
            )
            sys.stderr.flush()
    else:
        langs = all_lang_dirs

    if not langs:
        sys.stderr.write(f"[WARN] No language subfolders found under {raw_root}\n")
        return

    dev_str = "GPU" if use_gpu else "CPU"
    sys.stderr.write(
        f"[INFO] Batch mode: scanning {raw_root} → {out_root} (device: {dev_str})\n"
    )
    sys.stderr.write(
        f"[INFO] Output format: {output_format} "
        f"({'ID,FORM,LEMMA,UPOS' if output_format == 'basic' else 'full CoNLL-U'})\n"
    )
    sys.stderr.write(
        f"[INFO] Resume mode: {'ON (skip existing outputs)' if resume else 'OFF (overwrite all outputs)'}\n"
    )
    if chunk_lines is None or chunk_lines <= 1:
        sys.stderr.write("[INFO] Chunking: OFF (one line per forward pass)\n")
    else:
        sys.stderr.write(
            f"[INFO] Chunking: ON (up to {chunk_lines} lines per forward pass)\n"
        )
    sys.stderr.flush()

    for lang_dir in langs:
        lang_code = lang_dir.name
        sys.stderr.write(f"[INFO] Language: {lang_code}\n")
        sys.stderr.flush()

        txt_files = sorted(lang_dir.glob("*.txt"))
        if not txt_files:
            sys.stderr.write(f"[WARN]  No *.txt files in {lang_dir}; skipping.\n")
            sys.stderr.flush()
            continue

        for in_path in txt_files:
            out_lang_dir = out_root / lang_code
            ext = ".pos" if output_format == "basic" else ".conllu"
            out_path = out_lang_dir / f"{in_path.stem}{ext}"

            if resume and out_path.exists():
                sys.stderr.write(f"[INFO]   Skipping existing {out_path}\n")
                sys.stderr.flush()
                continue

            sys.stderr.write(f"[INFO]   Tagging {in_path} → {out_path}\n")
            sys.stderr.flush()

            # Fresh pipeline per file, no caching, then hard clean-up.
            nlp = load_pipeline(lang_code, use_gpu=use_gpu, cache=False)

            try:
                tag_file_to_disk(
                    input_path=in_path,
                    output_path=out_path,
                    nlp=nlp,
                    lang=lang_code,
                    output_format=output_format,
                    chunk_lines=chunk_lines,
                    use_gpu=use_gpu,
                )
            finally:
                # Hard cleanup after each file to avoid GPU OOM.
                try:
                    del nlp
                except NameError:
                    pass
                gc.collect()
                if use_gpu and torch is not None:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal Stanza-based POS/UD tagger for AIWL."
    )
    parser.add_argument(
        "--lang",
        "-l",
        default="en",
        help=(
            "Language code for Stanza (e.g. 'en', 'de', 'fr', 'zh'). "
            "Used in single-file/stdin mode. Default: 'en'."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help=(
            "Path to input text file. If omitted, read from stdin "
            "OR, if stdin is a TTY and no data is piped, run batch mode."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help=f"Base raw data directory for batch mode. Default: {DEFAULT_RAW_ROOT}",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Base output directory for batch mode. "
            "Default depends on --format: data/pos-basic or data/pos-conllu."
        ),
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["basic", "conllu"],
        default="basic",
        help=(
            "Output format: 'basic' = ID, FORM, LEMMA, UPOS (+ sent_id/text comments); "
            "'conllu' = full 10-column CoNLL-U. Default: basic."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Batch mode only: skip files whose output already exists "
            "(continue tagging after interruption at the file level). "
            "Default (no --resume): start from scratch and overwrite outputs."
        ),
    )
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=None,
        help=(
            "Batch mode only: if set to an integer >= 2, process up to this many "
            "non-empty lines together in a single Stanza call using the "
            "multi-document interface (better GPU utilisation). "
            "If omitted or <= 1, process one line per call (baseline behaviour)."
        ),
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=None,
        help=(
            "Batch mode only: optional space-separated list of language codes "
            "to process (e.g. --langs en de fr). If omitted, all language "
            "subfolders under --raw-dir are processed."
        ),
    )

    args = parser.parse_args()

    use_gpu = _detect_use_gpu()
    dev_str = "GPU" if use_gpu else "CPU"
    sys.stderr.write(f"[INFO] Using device: {dev_str}\n")
    sys.stderr.flush()

    # ---- Single-file / stdin mode if --input is given or stdin is piped ----
    if args.input:
        input_path = Path(args.input)
        with input_path.open("r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            sys.stderr.write("No input text provided (file is empty). Exiting.\n")
            sys.stderr.flush()
            sys.exit(1)

        nlp = load_pipeline(args.lang, use_gpu=use_gpu, cache=True)
        sent_id_prefix = f"{args.lang}-{input_path.stem}"

        tag_text_to_stream(
            text,
            nlp,
            sys.stdout,
            output_format=args.format,
            sent_id_prefix=sent_id_prefix,
            flush_every=None,
        )
        return

    # No --input: decide between stdin mode and batch mode
    if not sys.stdin.isatty():
        # Stdin mode (piped input)
        text = sys.stdin.read()
        if not text.strip():
            sys.stderr.write("No input text provided on stdin. Exiting.\n")
            sys.stderr.flush()
            sys.exit(1)

        nlp = load_pipeline(args.lang, use_gpu=use_gpu, cache=True)
        sent_id_prefix = f"{args.lang}-stdin"

        tag_text_to_stream(
            text,
            nlp,
            sys.stdout,
            output_format=args.format,
            sent_id_prefix=sent_id_prefix,
            flush_every=None,
        )
        return

    # ---- Batch mode: no --input and stdin is a TTY ----
    raw_root = Path(args.raw_dir) if args.raw_dir else DEFAULT_RAW_ROOT
    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        out_root = (
            DEFAULT_OUT_ROOT_BASIC
            if args.format == "basic"
            else DEFAULT_OUT_ROOT_CONLLU
        )

    batch_tag_corpus(
        raw_root=raw_root,
        out_root=out_root,
        use_gpu=use_gpu,
        output_format=args.format,
        resume=args.resume,
        chunk_lines=args.chunk_lines,
        target_langs=args.langs,
    )


if __name__ == "__main__":
    main()
