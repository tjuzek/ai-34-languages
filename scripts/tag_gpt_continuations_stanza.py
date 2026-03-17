#!/usr/bin/env python
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
Stanza POS tagger for GPT continuations (folder version).

Input:
    A directory with JSONL files from the GPT continuation script
    (e.g. de_pregenhalves_2020-2021_gpt-4.1-mini_generations.jsonl).
    Each line contains a GPT continuation under "output.ai_continuation".

Output:
    One JSONL per input file, written to an output directory, preserving
    all fields and adding:

        "parse_ai": [
            {"lemma": "...", "pos": "..."},
            ...
        ]

    Output is in *basic* format (lemma + UPOS), suitable for LAS.


Usage example:

    python scripts/tag_gpt_continuations_stanza.py \
      --input-dir  data/gpt4.1-continuations-mini \
      --output-dir data/ig/gpt4.1-continuations-mini-pos

By default we tag all files matching "*_generations.jsonl" in the input dir.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Work around PyTorch 2.6+ weights_only default (same as tag_with_stanza.py)
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import stanza  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None

STANZA_PROCESSORS = "tokenize,pos,lemma"

# Heuristic safeguard: skip absurdly long continuations (in chars)
MAX_CHARS_PER_DOC = int(os.environ.get("AIWL_MAX_DOC_CHARS", "20000"))


# ---------- GPU / pipeline helpers ----------

def detect_use_gpu() -> bool:
    """Return True if a GPU is available and torch is installed, else False."""
    if torch is None:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


_PIPELINES: Dict[str, stanza.Pipeline] = {}


def load_pipeline(lang: str, use_gpu: bool) -> stanza.Pipeline:
    """
    Load a Stanza pipeline for the given language (tokenize,pos,lemma).

    Pipelines are cached per (lang, device) combo.
    """
    key = f"{lang}_{'gpu' if use_gpu else 'cpu'}"
    if key in _PIPELINES:
        return _PIPELINES[key]

    stanza.download(lang, processors=STANZA_PROCESSORS)

    nlp = stanza.Pipeline(
        lang=lang,
        processors=STANZA_PROCESSORS,
        use_gpu=use_gpu,
    )

    _PIPELINES[key] = nlp
    return nlp


# ---------- Core tagging helpers ----------

def detect_lang_from_filename(path: Path) -> str:
    """
    Infer language code from filename prefix: "de_..." -> "de".

    Assumes your filenames start with the 2-letter language code followed
    by an underscore, as in:

        de_pregenhalves_2020-2021_gpt-4.1-mini_generations.jsonl
    """
    name = path.name
    parts = name.split("_", 1)
    if len(parts) < 2:
        raise ValueError(f"Cannot infer language from filename: {name}")
    return parts[0]


def tag_text_basic(nlp: stanza.Pipeline, text: str) -> List[Dict[str, str]]:
    """
    Run Stanza and return a flat list of tokens in *basic* JSON format:

        [{"lemma": ..., "pos": ...}, ...]
    """
    doc = nlp(text)
    out: List[Dict[str, str]] = []
    for sent in doc.sentences:
        for w in sent.words:
            lemma = w.lemma if w.lemma is not None else "_"
            pos = w.upos if w.upos is not None else "_"
            out.append({"lemma": lemma, "pos": pos})
    return out


def tag_batch_basic(
    nlp: stanza.Pipeline,
    texts: List[str],
    file_label: str,
    batch_line_nos: List[int],
) -> List[Optional[List[Dict[str, str]]]]:
    """
    Tag a batch of texts using Stanza's multi-document interface.

    Returns a list of per-text results (same length as `texts`), where each
    element is either a list[{"lemma","pos"}, ...] on success, or None on error.

    Robustness:
    - If the batch call fails, we fall back to per-text tagging.
    - If a specific text still fails, we return None for that position.
    """
    if not texts:
        return []

    # First try: multi-document call
    docs = [stanza.Document([], text=t) for t in texts]
    try:
        proc = nlp(docs)
    except Exception as e_batch:
        sys.stderr.write(
            f"[ERROR] {file_label}: batch tagging failed: {e_batch}\n"
            "[INFO]  Falling back to per-text tagging for this batch.\n"
        )
        sys.stderr.flush()

        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        results: List[Optional[List[Dict[str, str]]]] = []
        for line_no, text in zip(batch_line_nos, texts):
            try:
                tokens = tag_text_basic(nlp, text)
            except Exception as e_line:
                sys.stderr.write(
                    f"[ERROR] {file_label}: line {line_no}: "
                    f"tagging failed in fallback: {e_line}\n"
                )
                sys.stderr.flush()
                if torch is not None:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                results.append(None)
            else:
                results.append(tokens)
        return results

    # Batch succeeded
    if isinstance(proc, stanza.Document):
        proc_docs = [proc]
    else:
        proc_docs = list(proc)

    if len(proc_docs) != len(texts):
        sys.stderr.write(
            f"[WARN]  {file_label}: expected {len(texts)} docs from batch, "
            f"got {len(proc_docs)}; aligning by min length.\n"
        )
        sys.stderr.flush()

    results: List[Optional[List[Dict[str, str]]]] = []
    for doc in proc_docs[: len(texts)]:
        tokens: List[Dict[str, str]] = []
        for sent in doc.sentences:
            for w in sent.words:
                lemma = w.lemma if w.lemma is not None else "_"
                pos = w.upos if w.upos is not None else "_"
                tokens.append({"lemma": lemma, "pos": pos})
        results.append(tokens)

    # If for some reason we got fewer docs than texts, pad with None
    while len(results) < len(texts):
        results.append(None)

    return results


def process_file(
    in_path: Path,
    out_path: Path,
    use_gpu: bool,
    batch_size: int,
) -> None:
    """
    Tag a single GPT-continuation JSONL file.

    - Detect language from filename (and optionally sanity-check against JSON).
    - Read line-by-line and build batches of up to `batch_size` items that
      actually need tagging (non-empty, not too long).
    - For each batch:
        * tag all ai_continuation texts via tag_batch_basic(...)
        * attach parse_ai (basic lemma+pos) or [] on failure
        * write JSON to out_path
    """
    lang = detect_lang_from_filename(in_path)
    nlp = load_pipeline(lang, use_gpu=use_gpu)

    sys.stderr.write(
        f"[INFO]   {in_path.name}: lang={lang}, device={'GPU' if use_gpu else 'CPU'}, "
        f"batch_size={batch_size}\n"
    )
    sys.stderr.flush()

    total = 0
    tagged = 0
    skipped_empty = 0
    skipped_too_long = 0
    failed = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Buffers for batching
    batch_objs: List[dict] = []
    batch_texts: List[str] = []
    batch_line_nos: List[int] = []

    def flush_batch():
        nonlocal tagged, failed

        if not batch_objs:
            return

        file_label = in_path.name
        results = tag_batch_basic(nlp, batch_texts, file_label, batch_line_nos)

        for obj, res in zip(batch_objs, results):
            if res is None:
                obj["parse_ai"] = []
                failed += 1
            else:
                obj["parse_ai"] = res
                tagged += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        batch_objs.clear()
        batch_texts.clear()
        batch_line_nos.clear()

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                obj = json.loads(line)
            except Exception as e:
                sys.stderr.write(
                    f"[ERROR] {in_path.name}: line {line_no}: JSON parse failed: {e}\n"
                )
                sys.stderr.flush()
                failed += 1
                # We still write something, to keep line counts aligned if needed
                obj = {"_parse_error": True, "raw_line": line, "parse_ai": []}
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            ai_text = (
                obj.get("output", {})
                   .get("ai_continuation", "")
            )

            if not ai_text or not ai_text.strip():
                # No continuation => empty parse
                obj["parse_ai"] = []
                skipped_empty += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            text = ai_text.strip()

            if len(text) > MAX_CHARS_PER_DOC:
                sys.stderr.write(
                    f"[WARN]  {in_path.name}: line {line_no}: "
                    f"ai_continuation too long ({len(text)} chars > MAX_CHARS_PER_DOC={MAX_CHARS_PER_DOC}); "
                    f"leaving parse_ai empty.\n"
                )
                sys.stderr.flush()
                obj["parse_ai"] = []
                skipped_too_long += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            # Queue for batch processing
            batch_objs.append(obj)
            batch_texts.append(text)
            batch_line_nos.append(line_no)

            if batch_size <= 1 or len(batch_objs) >= batch_size:
                flush_batch()
                if total % 1000 == 0:
                    sys.stderr.write(
                        f"[INFO]   {in_path.name}: processed {total} lines "
                        f"(tagged={tagged}, empty={skipped_empty}, "
                        f"too_long={skipped_too_long}, failed={failed})\n"
                    )
                    sys.stderr.flush()

        # Flush remaining batch
        flush_batch()

    sys.stderr.write(
        f"[INFO]   {in_path.name}: done. "
        f"total={total}, tagged={tagged}, empty={skipped_empty}, "
        f"too_long={skipped_too_long}, failed={failed}\n"
    )
    sys.stderr.flush()


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="POS-tag GPT continuations (folder) with Stanza."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory with *_generations.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to write tagged *_generations_pos.jsonl files.",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default="*_generations.jsonl",
        help="Glob pattern for input files (default: *_generations.jsonl).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Number of continuations to send to Stanza at once "
             "(default: 32; set to 1 to disable batching).",
    )

    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.is_dir():
        sys.stderr.write(f"[ERROR] Input dir does not exist or is not a directory: {in_dir}\n")
        sys.exit(1)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        sys.stderr.write(f"[ERROR] No files matching pattern {args.pattern!r} in {in_dir}\n")
        sys.exit(1)

    use_gpu = detect_use_gpu()
    sys.stderr.write(
        f"[INFO] Using device: {'GPU' if use_gpu else 'CPU'}\n"
        f"[INFO] Input dir:  {in_dir}\n"
        f"[INFO] Output dir: {out_dir}\n"
        f"[INFO] Files:      {len(files)}\n"
        f"[INFO] Batch size: {args.batch_size}\n"
    )
    sys.stderr.flush()

    for in_path in files:
        # Mirror filename: e.g. de_..._generations.jsonl -> de_..._generations_pos.jsonl
        out_name = in_path.name.replace(".jsonl", "_pos.jsonl")
        out_path = out_dir / out_name
        process_file(in_path, out_path, use_gpu=use_gpu, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
