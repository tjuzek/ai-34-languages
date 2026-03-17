#!/usr/bin/env python3
# Developed with AI assistance (GPT, Gemini, Claude). All code reviewed and validated by the authors.
"""
WMT News Crawl downloader & structurer.

- Crawls https://data.statmt.org/news-crawl/
- Downloads files like news.2024.en.shuffled.deduped.gz
- Decompresses to data/{lang}/{year}.txt (one sentence per line)
- Writes a manifest.csv with metadata (sizes, hashes, line counts)
- Optional: builds a Parquet dataset (lang, year, line_no, text)
"""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# Optional: only if you pass --parquet
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None


ROOT_URL = "https://data.statmt.org/news-crawl/"
DEFAULT_OUTDIR = "data"
MANIFEST_NAME = "manifest.csv"
USER_AGENT = "news-crawl-downloader/1.0 (+for research; contact: your_email@example.com)"
LANG_DIR_RE = re.compile(r'href="([a-z\-]{2,5})/"/>?')  # generous but safe
FILE_RE = re.compile(r'href="(news\.(\d{4})\.([a-z\-]{2,5})\.shuffled\.deduped\.gz)"')
CHUNK = 1024 * 1024  # 1 MiB


def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=10,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    s.headers.update({"User-Agent": USER_AGENT})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


def fetch_text(url: str, session: requests.Session) -> str:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    # Directory listings are small; decode as utf-8
    return r.text


def list_languages(session: requests.Session) -> List[str]:
    html = fetch_text(ROOT_URL, session)
    langs = sorted({m.group(1) for m in re.finditer(r'href="([a-z\-]{2,5})/"/?>', html)})
    return langs


def list_files_for_lang(lang: str, session: requests.Session) -> List[Tuple[str, int, int]]:
    """
    Returns list of tuples: (relative_name, year, year_int)
    Example: ("news.2024.en.shuffled.deduped.gz", "2024", 2024)
    """
    url = urljoin(ROOT_URL, f"{lang}/")
    html = fetch_text(url, session)
    matches = list(re.finditer(FILE_RE, html))
    files = []
    for m in matches:
        rel = m.group(1)
        year = int(m.group(2))
        files.append((rel, year))
    return files


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_gzip_lines(stream: io.BufferedReader) -> Iterable[str]:
    with gzip.GzipFile(fileobj=stream, mode="rb") as gz:
        for raw in gz:
            yield raw.decode("utf-8", errors="replace").rstrip("\n")


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def download_and_process(
    lang: str,
    year: int,
    rel_name: str,
    outdir: Path,
    session: requests.Session,
    build_parquet: bool = False,
) -> dict:
    """
    Downloads, saves .gz, writes decompressed .txt,
    counts lines, computes hashes, and (optionally) appends to Parquet.
    Returns a manifest row dict.
    """
    lang_dir_url = urljoin(ROOT_URL, f"{lang}/")
    url = urljoin(lang_dir_url, rel_name)

    # Paths
    gz_path = outdir / "gz" / lang / f"{rel_name}"
    txt_path = outdir / lang / f"{year}.txt"
    ensure_parent(gz_path)
    ensure_parent(txt_path)

    # Download (streaming)
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        tmp_gz = gz_path.with_suffix(gz_path.suffix + ".part")
        with tmp_gz.open("wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=f"GET {lang}/{year}",
            leave=False,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        tmp_gz.replace(gz_path)

    gz_bytes = gz_path.stat().st_size
    sha_gz = sha256_file(gz_path)

    # Decompress to .txt and count lines
    line_count = 0
    tmp_txt = txt_path.with_suffix(".txt.part")
    with gz_path.open("rb") as fbin, tmp_txt.open("w", encoding="utf-8") as fout:
        for line in iter_gzip_lines(fbin):
            fout.write(line + "\n")
            line_count += 1
    tmp_txt.replace(txt_path)
    txt_bytes = txt_path.stat().st_size
    sha_txt = sha256_file(txt_path)

    # Optional Parquet append (one row per line)
    parquet_rows = 0
    if build_parquet:
        if pa is None or pq is None:
            print("pyarrow not available; skipping Parquet.", file=sys.stderr)
        else:
            parquet_dir = outdir / "_parquet"
            parquet_dir.mkdir(parents=True, exist_ok=True)
            # Stream rows in chunks to avoid huge memory
            batch_size = 200_000
            buf_texts: List[str] = []
            buf_years: List[int] = []
            buf_langs: List[str] = []
            buf_lines: List[int] = []
            with txt_path.open("r", encoding="utf-8") as fin:
                for i, line in enumerate(fin, start=1):
                    buf_texts.append(line.rstrip("\n"))
                    buf_years.append(year)
                    buf_langs.append(lang)
                    buf_lines.append(i)
                    if len(buf_texts) >= batch_size:
                        write_parquet_batch(parquet_dir, lang, year, buf_langs, buf_years, buf_lines, buf_texts)
                        parquet_rows += len(buf_texts)
                        buf_texts.clear(); buf_years.clear(); buf_langs.clear(); buf_lines.clear()
                if buf_texts:
                    write_parquet_batch(parquet_dir, lang, year, buf_langs, buf_years, buf_lines, buf_texts)
                    parquet_rows += len(buf_texts)

    return dict(
        lang=lang,
        year=year,
        url=url,
        gz_path=str(gz_path),
        txt_path=str(txt_path),
        gz_bytes=gz_bytes,
        txt_bytes=txt_bytes,
        lines=line_count,
        sha256_gz=sha_gz,
        sha256_txt=sha_txt,
        parquet_rows=parquet_rows,
        downloaded_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )


def write_parquet_batch(
    parquet_dir: Path,
    lang: str,
    year: int,
    langs: List[str],
    years: List[int],
    lines: List[int],
    texts: List[str],
):
    table = pa.Table.from_pydict({
        "lang": langs,
        "year": years,
        "line_no": lines,
        "text": texts,
    })
    # One file per lang/year chunk to keep it simple and appendable
    # Use a time-based suffix to avoid collisions
    suffix = int(time.time() * 1000)
    fname = parquet_dir / f"{lang}-{year}-{suffix}.parquet"
    pq.write_table(table, fname)


def write_manifest_header(path: Path):
    ensure_parent(path)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "lang", "year", "url", "gz_path", "txt_path",
                "gz_bytes", "txt_bytes", "lines",
                "sha256_gz", "sha256_txt", "parquet_rows", "downloaded_at"
            ])
            w.writeheader()


def append_manifest_row(path: Path, row: dict):
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "lang", "year", "url", "gz_path", "txt_path",
            "gz_bytes", "txt_bytes", "lines",
            "sha256_gz", "sha256_txt", "parquet_rows", "downloaded_at"
        ])
        w.writerow(row)


def parse_years(years: Optional[str]) -> Optional[Tuple[int, int]]:
    if not years:
        return None
    if "-" in years:
        a, b = years.split("-", 1)
        return int(a), int(b)
    y = int(years)
    return (y, y)


def main():
    ap = argparse.ArgumentParser(
        description="Crawl and structure WMT News Crawl (monolingual news) for given languages/years."
    )
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory (default: data)")
    ap.add_argument("--langs", nargs="*", help="Language codes to include (e.g., en nl de). Default: all available.")
    ap.add_argument("--years", help="Year or range (e.g., 2019 or 2019-2024). Default: all available.")
    ap.add_argument("--max-workers", type=int, default=4, help="Concurrent downloads (default: 4).")
    ap.add_argument("--parquet", action="store_true", help="Also build a Parquet dataset (requires pyarrow).")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    session = make_session()
    manifest_path = outdir / MANIFEST_NAME
    write_manifest_header(manifest_path)

    # Discover languages
    print("Discovering languages...")
    all_langs = list_languages(session)
    if args.langs:
        target_langs = [l for l in args.langs if l in all_langs]
        missing = set(args.langs) - set(target_langs)
        if missing:
            print(f"Warning: unknown or unavailable languages ignored: {sorted(missing)}", file=sys.stderr)
    else:
        target_langs = all_langs
    print(f"Languages: {', '.join(target_langs)}")

    # Year filter
    year_range = parse_years(args.years)

    # Collect all tasks
    tasks = []
    print("Enumerating files...")
    for lang in target_langs:
        files = list_files_for_lang(lang, session)
        for rel, year in files:
            if year_range and not (year_range[0] <= year <= year_range[1]):
                continue
            tasks.append((lang, year, rel))

    # Sort deterministically (by lang, then year)
    tasks.sort(key=lambda t: (t[0], t[1]))
    print(f"Planned downloads: {len(tasks)} files")

    # Download concurrently
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        fut2task = {
            ex.submit(download_and_process, lang, year, rel, outdir, session, args.parquet): (lang, year, rel)
            for (lang, year, rel) in tasks
        }
        for fut in tqdm(as_completed(fut2task), total=len(fut2task), desc="Processing", unit="file"):
            lang, year, rel = fut2task[fut]
            try:
                row = fut.result()
                results.append(row)
                append_manifest_row(manifest_path, row)
            except Exception as e:
                print(f"ERROR [{lang}/{year} {rel}]: {e}", file=sys.stderr)

    print(f"Done. Manifest written to: {manifest_path}")
    if args.parquet:
        print(f"Parquet shards in: {outdir / '_parquet'}")
        print("Tip: create a dataset with pyarrow.dataset.dataset() over that directory for fast reads.")


if __name__ == "__main__":
    main()

