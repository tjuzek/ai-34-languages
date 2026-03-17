#!/usr/bin/env python
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
from __future__ import annotations

"""
Generate Claude 3.5 Haiku continuations for split-halves prompts in 35 languages.

- Reads JSONL files with pre-generated first halves from --input-dir.
- Uses language-specific prompt frames and token_ratio from config/language-specific-prompts.tsv.
- Estimates model-token lengths from POS-token lengths via token_ratio.
- Calls the Anthropic Messages API (Claude 3.5 Haiku by default) with a fixed system prompt.
- Writes one JSONL per input file with rich metadata, plus a run-level summary JSON.
- Supports concurrency and an optional requests-per-minute (RPM) cap.

Input JSONL (per line, abbreviated):

{
  "id":        { "lang": "en", "year": 2020, "file": "en/2020.pos",
                 "sent_id": "en-2020-L115-s1", "doc_id": "L115", "sent_idx": 1 },
  "metrics":   { "n_tokens_first": 21, ... },
  "split_meta": { "index": 21, "method": "strict_alignment", ... },
  "text":      { "full_original": "...", "first_half": "...", "second_half": "..." },
  "parse":     { ... }  # ignored here
}

language-specific-prompts.tsv (tab-separated):

  lang_code  prompt_frame  token_ratio  verified

- lang_code: matches id.lang and file prefix.
- prompt_frame: format string with exactly one '{}' slot for first_half.
- token_ratio: model_tokens / POS_tokens for that language.

Length control (in model tokens):

  Let n = metrics["n_tokens_first"], r = token_ratio.

  min_raw   = n * r * 0.5
  min_floor = 20 * r
  min_required_tokens = ceil(min(min_raw, min_floor))

  max_raw   = n * r * 1.5
  max_ceiling = 200 * r
  max_output_tokens = ceil(min(max_raw, max_ceiling))

If max_output_tokens < min_required_tokens, set max_output_tokens = min_required_tokens.

The script enforces this approximately by passing max_tokens to the API
and retrying (up to --max-attempts) when usage.output_tokens < min_required_tokens.
Metadata (min_required_tokens, usage, min_tokens_satisfied, etc.) is logged
for downstream analysis.

Example usage:
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/generate_haiku_continuations.py \
  --input-dir data/ig/validation-models/pre-gen-halves \
  --output-dir data/ig/validation-models/haiku3.5-continuations \
  --model claude-3-haiku-20240307 \
  --temperature 0.0 \
  --top-p 1.0 \
  --concurrency 32 \
  --rpm 2500 \
  -v
"""

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from anthropic import Anthropic, BadRequestError  # hard import, no runtime fallback

SCRIPT_VERSION = "0.1.0"

# Default relative path for language-specific prompts (from repo root).
# Assumes this script lives in e.g. `scripts/` or `tools/` under the repo root.
DEFAULT_PROMPT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "language-specific-prompts.tsv"
)

SYSTEM_PROMPT = (
    "You are a helpful and knowledgeable assistant. "
    "Follow the user's instructions and focus on the task they provide. "
    "Your task is to provide the immediate continuation of the provided text fragment. "
    "Guidelines:\n"
    "1. Do NOT repeat the input text.\n"
    "2. If the input text is machine-formatted or technical data, output an empty string (\"\") and nothing else.\n"
    "3. Do NOT provide any conversational preface or acknowledgments (e.g., 'Here is the continuation...').\n"
    "4. Output ONLY natural language.\n"
    "5. Output ONLY the continuation text."
)

# How often to flush+fsync output files.
FLUSH_EVERY_N_RECORDS = 50


@dataclass
class LanguagePromptConfig:
    lang_code: str
    prompt_frame: str
    token_ratio: float
    verified: bool


@dataclass
class GenerationArgs:
    model: str
    temperature: float
    top_p: float
    max_attempts: int
    num_completions: int
    seed: Optional[int] = None


@dataclass
class RequestResult:
    continuation: str
    usage: Dict[str, Any]
    request_payload: Dict[str, Any]
    attempts: int
    min_tokens_satisfied: bool


@dataclass
class RunStats:
    total_items: int = 0          # number of (item × completion) records written
    total_requests: int = 0       # same as total_items (one request per completion)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    per_lang: Dict[str, Dict[str, int]] = dataclasses.field(default_factory=dict)


# ---------- Simple RPM limiter (global across threads) ----------
class RpmLimiter:
    """
    Enforces a minimum spacing between requests: interval = 60 / rpm seconds.
    Thread-safe. If rpm is falsy/None/<=0, limiter is disabled.
    """

    def __init__(self, rpm: Optional[int]):
        if rpm is None or rpm <= 0:
            self.enabled = False
            self.interval = 0.0
        else:
            self.enabled = True
            self.interval = 60.0 / float(rpm)
        self._lock = threading.Lock()
        self._next_time = time.monotonic()

    def wait(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                time.sleep(self._next_time - now)
                now = self._next_time
            self._next_time = now + self.interval


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger("aiwl.generate_haiku")


def load_language_prompts(path: Path) -> Dict[str, LanguagePromptConfig]:
    """
    Load language-specific prompt frames and token_ratio from TSV.

    Expected header:
        lang_code  prompt_frame  token_ratio  verified
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompt config TSV not found at {path}")

    prompts: Dict[str, LanguagePromptConfig] = {}
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        cols = header.split("\t")
        expected = ["lang_code", "prompt_frame", "token_ratio", "verified"]
        if cols != expected:
            raise ValueError(f"Unexpected header in {path}: {cols} (expected {expected})")

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"Expected 4 columns in {path}, got {len(parts)}: {line!r}")

            lang_code, prompt_frame_raw, token_ratio_s, verified_s = parts
            # Convert \n escapes to real newlines
            prompt_frame = prompt_frame_raw.replace("\\n", "\n")

            try:
                token_ratio = float(token_ratio_s)
            except ValueError as e:
                raise ValueError(f"Invalid token_ratio {token_ratio_s!r} for lang={lang_code}") from e

            if token_ratio <= 0:
                raise ValueError(f"token_ratio must be positive for lang={lang_code}, got {token_ratio}")

            verified = verified_s.strip() == "1"

            prompts[lang_code] = LanguagePromptConfig(
                lang_code=lang_code,
                prompt_frame=prompt_frame,
                token_ratio=token_ratio,
                verified=verified,
            )

    return prompts


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e


def ensure_trailing_newline(path: Path) -> None:
    """
    Ensure output JSONL file ends with a newline. This matches LREC utilities and
    helps with simple `cat`/`tail` usage and resuming.
    """
    if not path.exists():
        return
    try:
        if path.stat().st_size == 0:
            return
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            with path.open("a", encoding="utf-8") as out:
                out.write("\n")
    except Exception:
        # Best-effort only.
        logger.warning("Failed to ensure trailing newline for %s", path, exc_info=True)


def load_completed_item_ids(out_path: Path) -> Set[str]:
    """
    If the output JSONL already exists, read item_ids so we can resume.
    """
    completed: Set[str] = set()
    if not out_path.exists():
        return completed

    logger.info("Resuming from existing output %s", out_path)
    for rec in iter_jsonl(out_path):
        item_id = rec.get("item_id")
        if item_id is not None:
            completed.add(item_id)
    logger.info("Found %d completed items in %s", len(completed), out_path)
    return completed


def build_user_prompt(first_half: str, cfg: LanguagePromptConfig) -> str:
    """
    Render the user-facing prompt by inserting the first half into the prompt frame.
    Leading/trailing whitespace is stripped from the first half.
    """
    fh = first_half.strip()
    try:
        prompt = cfg.prompt_frame.format(fh)
    except Exception as e:
        raise ValueError(f"Error formatting prompt for lang={cfg.lang_code!r}") from e
    return prompt


def create_client() -> Any:
    """
    Create an Anthropic client.

    Uses ANTHROPIC_API_KEY from the environment.
    """
    return Anthropic()


def call_anthropic_with_retries(
    client: Any,
    system_prompt: str,
    user_prompt: str,
    min_required_tokens: int,
    max_output_tokens: int,
    gen_args: GenerationArgs,
    rpm_limiter: Optional[RpmLimiter] = None,
) -> RequestResult:
    """
    Call the Anthropic Messages API with retries and minimal length control
    in terms of model tokens.

    Returns the continuation text, usage dict, request payload, attempts, and whether
    the min_required_tokens target was met (based on usage.output_tokens).

    Special cases:
      - If the model returns an empty string (""), accept it immediately as the final
        answer and do not retry further (used for technical / machine-formatted inputs).
      - If Anthropic returns a BadRequestError with "Output blocked by content filtering policy",
        treat this as an empty continuation and do NOT crash the run.
    """
    base_payload: Dict[str, Any] = {
        "model": gen_args.model,
        "max_tokens": max_output_tokens,
        "temperature": gen_args.temperature,
        "top_p": gen_args.top_p,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    }

    last_exc: Optional[Exception] = None
    attempts = 0
    usage: Dict[str, Any] = {}
    continuation_text = ""
    min_satisfied = False

    for attempt in range(1, gen_args.max_attempts + 1):
        attempts = attempt
        try:
            if rpm_limiter is not None:
                rpm_limiter.wait()

            payload = dict(base_payload)
            logger.debug(
                "Calling Anthropic (attempt %d/%d, max_tokens=%d, min_required_tokens=%d)",
                attempt,
                gen_args.max_attempts,
                max_output_tokens,
                min_required_tokens,
            )

            response = client.messages.create(**payload)

            # Extract continuation text: response.content[0].text
            try:
                if not response.content:
                    continuation_text = ""
                else:
                    block0 = response.content[0]
                    continuation_text = getattr(block0, "text", "") or ""
            except Exception as e:  # pragma: no cover - defensive
                raise RuntimeError(f"Unexpected response structure: {response}") from e

            # Usage accounting
            usage_raw = getattr(response, "usage", None)
            if usage_raw is not None:
                if isinstance(usage_raw, dict):
                    usage = dict(usage_raw)
                elif hasattr(usage_raw, "model_dump"):
                    usage = usage_raw.model_dump()
                else:
                    raw_dict = getattr(usage_raw, "__dict__", {})
                    usage = {
                        k: v
                        for k, v in raw_dict.items()
                        if not k.startswith("_") and not callable(v)
                    }
            else:
                usage = {}

            # SPECIAL CASE: empty string => accept immediately
            if continuation_text == "":
                logger.info(
                    "Model returned empty string; accepting as final output and skipping retries."
                )
                break

            out_tokens = int(usage.get("output_tokens") or 0)
            if out_tokens >= min_required_tokens:
                min_satisfied = True
                break

            logger.info(
                "Output shorter than min_required_tokens (%d < %d); attempt %d/%d",
                out_tokens,
                min_required_tokens,
                attempt,
                gen_args.max_attempts,
            )
        except BadRequestError as e:
            # Special handling for Anthropic safety filter: "Output blocked by content filtering policy"
            msg = str(e)
            if "Output blocked by content filtering policy" in msg:
                logger.warning(
                    "Anthropic content filter blocked output; returning empty continuation for this item."
                )
                request_payload = dict(base_payload)
                return RequestResult(
                    continuation="",
                    usage={},
                    request_payload=request_payload,
                    attempts=attempts,
                    min_tokens_satisfied=False,
                )
            # Other BadRequestError: treat as normal failure, retry/backoff
            last_exc = e
            logger.warning(
                "Anthropic BadRequestError on attempt %d/%d: %s",
                attempt,
                gen_args.max_attempts,
                e,
            )
            sleep_s = min(60.0, 2.0 ** attempt + random.random())
            time.sleep(sleep_s)
            continue
        except Exception as e:  # pragma: no cover
            last_exc = e
            logger.warning(
                "Anthropic call failed on attempt %d/%d: %s",
                attempt,
                gen_args.max_attempts,
                e,
            )
            sleep_s = min(60.0, 2.0 ** attempt + random.random())
            time.sleep(sleep_s)
            continue

    if continuation_text == "" and last_exc is not None:
        # This path means we never got a successful response at all.
        raise last_exc

    request_payload = dict(base_payload)
    return RequestResult(
        continuation=continuation_text,
        usage=usage,
        request_payload=request_payload,
        attempts=attempts,
        min_tokens_satisfied=min_satisfied,
    )


def generate_single_completion(
    record: Dict[str, Any],
    idx: int,
    in_path: Path,
    lang_default: str,
    prompt_cfg: LanguagePromptConfig,
    gen_args: GenerationArgs,
    run_id: str,
    client: Any,
    rpm_limiter: Optional[RpmLimiter],
    completion_index: int,
) -> Dict[str, Any]:
    """
    Worker function: generate one completion for a single record.

    Returns a dict with:
      {
        "out_record": <JSON-serialisable dict>,
        "lang_code": <lang>,
        "input_tokens": int,
        "output_tokens": int,
      }
    """
    _id = record.get("id", {})
    lang_code = _id.get("lang", lang_default)

    metrics = record.get("metrics") or {}
    split_meta = record.get("split_meta") or {}
    text = record.get("text") or {}
    first_half_raw = text.get("first_half")
    if first_half_raw is None:
        item_id = _id.get("sent_id")
        raise ValueError(f"Record missing text.first_half for item_id={item_id}")

    item_id = _id.get("sent_id")
    if item_id is None:
        raise ValueError(f"Record in {in_path} missing id.sent_id: {record}")

    n_tokens_first = int(metrics.get("n_tokens_first") or 0)
    if n_tokens_first <= 0:
        logger.warning(
            "Non-positive n_tokens_first for item_id=%s in %s; defaulting to 1",
            item_id,
            in_path,
        )
        n_tokens_first = 1

    r = prompt_cfg.token_ratio

    # Compute min and max in model tokens (floats before ceil)
    min_raw = n_tokens_first * r * 0.5
    min_floor = 20.0 * r
    min_required_tokens = int(math.ceil(min(min_raw, min_floor)))

    max_raw = n_tokens_first * r * 1.5
    max_ceiling = 200.0 * r
    max_output_tokens = int(math.ceil(min(max_raw, max_ceiling)))

    # Safety: ensure max_output_tokens >= min_required_tokens
    if max_output_tokens < min_required_tokens:
        logger.debug(
            "Adjusting max_output_tokens (%d) to min_required_tokens (%d) for item_id=%s",
            max_output_tokens,
            min_required_tokens,
            item_id,
        )
        max_output_tokens = min_required_tokens

    user_prompt = build_user_prompt(first_half_raw, prompt_cfg)

    # Call Anthropic
    result = call_anthropic_with_retries(
        client=client,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        min_required_tokens=min_required_tokens,
        max_output_tokens=max_output_tokens,
        gen_args=gen_args,
        rpm_limiter=rpm_limiter,
    )

    continuation = result.continuation
    first_half_trimmed = first_half_raw.strip()
    full_with_ai = first_half_trimmed + continuation

    input_tokens = int(result.usage.get("input_tokens") or 0)
    output_tokens = int(result.usage.get("output_tokens") or 0)

    out_record: Dict[str, Any] = {
        "item_id": item_id,
        "lang": lang_code,
        "source": {
            "id": _id,
            "metrics": metrics,
            "split_meta": split_meta,
            "text": text,
            "source_file": str(in_path),
            "source_idx": idx,
        },
        "input": {
            "first_half": first_half_trimmed,
            "user_prompt": user_prompt,
            "system_prompt": SYSTEM_PROMPT,
        },
        "output": {
            "ai_continuation": continuation,
            "full_with_ai": full_with_ai,
        },
        "generation": {
            "model": gen_args.model,
            "run_id": run_id,
            "script_version": SCRIPT_VERSION,
            "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "decoding": {
                "temperature": gen_args.temperature,
                "top_p": gen_args.top_p,
                "max_output_tokens": max_output_tokens,
                "min_required_tokens": min_required_tokens,
                "min_tokens_satisfied": result.min_tokens_satisfied,
                "num_attempts": result.attempts,
                "seed": gen_args.seed,
            },
            "request_payload": result.request_payload,
            "usage": result.usage,
            "meta": {
                "prompt_frame_lang_code": prompt_cfg.lang_code,
                "token_ratio": prompt_cfg.token_ratio,
                "min_token_ratio_factor": 0.5,
                "max_token_ratio_factor": 1.5,
                "min_tokens_floor": min_floor,
                "max_tokens_ceiling": max_ceiling,
                "prompt_frame_verified": prompt_cfg.verified,
                "completion_index": completion_index,
            },
        },
    }

    return {
        "out_record": out_record,
        "lang_code": lang_code,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def process_language_file(
    client: Any,
    in_path: Path,
    out_dir: Path,
    lang: str,
    prompt_cfg: LanguagePromptConfig,
    gen_args: GenerationArgs,
    run_id: str,
    stats: RunStats,
    concurrency: int,
    rpm_limiter: Optional[RpmLimiter],
    max_items: Optional[int] = None,
) -> None:
    """
    Process a single language JSONL file and append generations to the corresponding output JSONL.

    Uses a thread pool to run multiple Anthropic requests concurrently.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = in_path.stem  # e.g., "en_pregenhalves_2020-2021"
    model_tag = gen_args.model.replace("/", "_").replace(":", "_")
    out_path = out_dir / f"{stem}__{model_tag}_generations.jsonl"

    ensure_trailing_newline(out_path)
    completed = load_completed_item_ids(out_path)

    num_written_since_flush = 0
    processed_records = 0

    logger.info(
        "Processing lang=%s from %s with concurrency=%d",
        lang,
        in_path,
        max(1, int(concurrency)),
    )

    with out_path.open("a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as ex:
            futures = []

            for idx, record in enumerate(iter_jsonl(in_path)):
                _id = record.get("id", {})
                lang_code_rec = _id.get("lang", lang)
                if lang_code_rec != lang:
                    logger.warning(
                        "Language mismatch in %s: expected %s but record has %s (sent_id=%s)",
                        in_path,
                        lang,
                        lang_code_rec,
                        _id.get("sent_id"),
                    )

                item_id = _id.get("sent_id")
                if item_id is None:
                    raise ValueError(f"Record in {in_path} missing id.sent_id: {record}")

                if item_id in completed:
                    logger.debug("Skipping already completed item_id=%s", item_id)
                    continue

                if max_items is not None and processed_records >= max_items:
                    logger.info(
                        "Reached max_items=%d records for lang=%s (input file %s)",
                        max_items,
                        lang,
                        in_path,
                    )
                    break

                processed_records += 1

                for completion_idx in range(gen_args.num_completions):
                    fut = ex.submit(
                        generate_single_completion,
                        record,
                        idx,
                        in_path,
                        lang,
                        prompt_cfg,
                        gen_args,
                        run_id,
                        client,
                        rpm_limiter,
                        completion_index=completion_idx,
                    )
                    futures.append(fut)

            for fut in as_completed(futures):
                result = fut.result()
                out_record = result["out_record"]
                lang_code_res = result["lang_code"]
                input_tokens = result["input_tokens"]
                output_tokens = result["output_tokens"]

                out_f.write(json.dumps(out_record, ensure_ascii=False))
                out_f.write("\n")
                num_written_since_flush += 1

                stats.total_items += 1
                stats.total_requests += 1

                lang_stats = stats.per_lang.setdefault(
                    lang_code_res,
                    {
                        "items": 0,
                        "requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    },
                )
                lang_stats["items"] += 1
                lang_stats["requests"] += 1

                stats.total_input_tokens += input_tokens
                stats.total_output_tokens += output_tokens
                lang_stats["input_tokens"] += input_tokens
                lang_stats["output_tokens"] += output_tokens

                if num_written_since_flush >= FLUSH_EVERY_N_RECORDS:
                    out_f.flush()
                    os.fsync(out_f.fileno())
                    num_written_since_flush = 0

        if num_written_since_flush > 0:
            out_f.flush()
            os.fsync(out_f.fileno())

    logger.info(
        "Finished lang=%s from %s; processed %d records (items, excluding skipped) with %d completions each; wrote to %s",
        lang,
        in_path,
        processed_records,
        gen_args.num_completions,
        out_path,
    )


def discover_language_files(input_dir: Path) -> Dict[str, Path]:
    """
    Discover pre-gen halves JSONL files in the input directory.

    Expected pattern: <lang>_pregenhalves_*.jsonl
    Returns mapping: lang_code -> Path.
    """
    mapping: Dict[str, Path] = {}
    for path in sorted(input_dir.glob("*_pregenhalves_*.jsonl")):
        stem = path.stem
        if "_pregenhalves_" not in stem:
            continue
        lang_code = stem.split("_pregenhalves_", 1)[0]
        mapping[lang_code] = path
    return mapping


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Claude 3.5 Haiku continuations for split-halves prompts in 35 languages."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *_pregenhalves_*.jsonl files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-language JSONL outputs.",
    )
    p.add_argument(
        "--prompt-config",
        type=Path,
        default=DEFAULT_PROMPT_CONFIG_PATH,
        help=(
            "Path to language-specific-prompts.tsv "
            "(default: config/language-specific-prompts.tsv relative to repo root)."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="Anthropic model name to use (default: claude-3-5-haiku-20241022).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy behaviour).",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling parameter (default: 1.0).",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts per item for enforcing min_required_tokens (default: 3).",
    )
    p.add_argument(
        "--num-completions",
        type=int,
        default=1,
        help="Number of completions to generate per item (default: 1).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed (used for backoff jitter etc., not for deterministic decoding).",
    )
    p.add_argument(
        "--langs",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of language codes to process (default: all discovered in input-dir).",
    )
    p.add_argument(
        "--max-items-per-lang",
        type=int,
        default=None,
        help=(
            "Optional cap on number of input records (items) to process per language "
            "(for debugging). Each record may yield multiple completions."
        ),
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of concurrent Anthropic requests per language (default: 1). "
            "Increase to speed up generation, subject to your rate limits."
        ),
    )
    p.add_argument(
        "--rpm",
        type=int,
        default=0,
        help=(
            "Optional global requests-per-minute cap across threads (0 or negative to disable). "
            "Useful to respect provider rate limits."
        ),
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use -v or -vv).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)

    if args.seed is not None:
        random.seed(args.seed)

    input_dir: Path = args.input-dir if False else args.input_dir  # just to avoid typos in reasoning; real line below
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    prompt_config_path: Path = args.prompt_config

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    prompt_cfgs = load_language_prompts(prompt_config_path)

    lang_files = discover_language_files(input_dir)
    if not lang_files:
        raise RuntimeError(f"No *_pregenhalves_*.jsonl files found in {input_dir}")

    if args.langs:
        unknown = [l for l in args.langs if l not in lang_files]
        if unknown:
            raise ValueError(f"Requested langs not found in input-dir: {unknown}")
        langs_to_run = args.langs
    else:
        langs_to_run = sorted(lang_files.keys())

    missing_prompts = [l for l in langs_to_run if l not in prompt_cfgs]
    if missing_prompts:
        raise RuntimeError(
            f"Missing prompt config for languages: {missing_prompts}. "
            f"Check {prompt_config_path}"
        )

    client = create_client()
    rpm_limiter = RpmLimiter(args.rpm)

    gen_args = GenerationArgs(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_attempts=args.max_attempts,
        num_completions=args.num_completions,
        seed=args.seed,
    )

    run_id = dt.datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")
    logger.info(
        "Starting Claude 3.5 Haiku continuation generation run_id=%s for langs=%s with concurrency=%d, rpm=%d",
        run_id,
        ",".join(langs_to_run),
        args.concurrency,
        args.rpm,
    )

    stats = RunStats()
    run_start_utc = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for lang in langs_to_run:
        in_path = lang_files[lang]
        prompt_cfg = prompt_cfgs[lang]
        process_language_file(
            client=client,
            in_path=in_path,
            out_dir=output_dir,
            lang=lang,
            prompt_cfg=prompt_cfg,
            gen_args=gen_args,
            run_id=run_id,
            stats=stats,
            concurrency=args.concurrency,
            rpm_limiter=rpm_limiter,
            max_items=args.max_items_per_lang,
        )

    run_end_utc = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    summary_path = meta_dir / f"generation_summary_{run_id}.json"

    prompt_cfg_hash = sha256_file(prompt_config_path)

    summary = {
        "run_id": run_id,
        "timestamp_utc_start": run_start_utc,
        "timestamp_utc_end": run_end_utc,
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "prompt_config": {
            "path": str(prompt_config_path),
            "sha256": prompt_cfg_hash,
        },
        "flags": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_attempts": args.max_attempts,
            "num_completions": args.num_completions,
            "seed": args.seed,
            "langs": langs_to_run,
            "max_items_per_lang": args.max_items_per_lang,
            "concurrency": args.concurrency,
            "rpm": args.rpm,
        },
        "stats": dataclasses.asdict(stats),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Wrote run summary to %s", summary_path)


if __name__ == "__main__":
    main()
