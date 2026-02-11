"""Estimate average tokens per document for each source dataset.

Streams documents from HuggingFace, batch-tokenizes with gemma-3-1b-it,
and tracks the running average tokens/doc. Stops once the average
converges (relative change < 0.1% between consecutive windows).
Multiplies by total row count from dataset metadata to estimate total tokens.

Row count resolution order:
  1. load_dataset_builder (fast, from dataset card metadata)
  2. HF datasets server /size API (fast, works for most datasets)
  3. Parquet metadata via list_repo_tree + pyarrow in parallel (slower, exact, always works)

Usage:
  python estimate_tokens.py

Results (gemma-3-1b-it, 2026-02-11):
  dclm:         3,468,923,154,406 tokens (3.5T)    [avg=1270 tok/doc x 2,732,074,726 docs]
  fw:          16,909,638,933,587 tokens (16.9T)   [avg=653 tok/doc x 25,886,364,489 docs]
  fw-edu:       1,567,210,463,942 tokens (1.6T)    [avg=1028 tok/doc x 1,525,223,056 docs]
  finepdfs:       726,126,526,327 tokens (726.1B)  [avg=3509 tok/doc x 206,917,202 docs]
  finepdfs-edu:   135,905,064,391 tokens (135.9B)  [avg=5903 tok/doc x 23,023,372 docs]
"""

import logging
import time

import pyarrow.parquet as pq
import requests
from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfApi, HfFileSystem
from joblib import Parallel, delayed
from tokenizers import Tokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TOKENIZER_NAME = "google/gemma-3-1b-it"
TEXT_KEY = "text"

# Convergence parameters
BATCH = 1000  # Tokenize and check convergence every BATCH docs
THRESHOLD = 0.001  # Stop when relative change < 0.1%
MIN_DOCS = 10_000  # Minimum docs before checking convergence
MAX_DOCS = 100_000  # Hard cap

# name -> (hf_id, config | None, split)
SOURCES: dict[str, tuple[str, str | None, str]] = {
    "dclm": ("mlfoundations/dclm-baseline-1.0-parquet", None, "train"),
    "fw": ("HuggingFaceFW/fineweb", None, "train"),
    "fw-edu": ("HuggingFaceFW/fineweb-edu", None, "train"),
    "finepdfs": ("HuggingFaceFW/finepdfs", "eng_Latn", "train"),
    "finepdfs-edu": ("HuggingFaceFW/finepdfs-edu", "eng_Latn", "train"),
}


def _num_rows_from_builder(hf_id: str, config: str | None, split: str) -> int | None:
    """Try load_dataset_builder for row count (fast, from dataset card)."""
    try:
        builder = load_dataset_builder(hf_id, config)
        if builder.info.splits and split in builder.info.splits:
            return builder.info.splits[split].num_examples
    except Exception:
        pass
    return None


def _num_rows_from_server_api(hf_id: str, config: str | None, split: str) -> int | None:
    """Try the HF datasets server /size endpoint.

    Uses estimated_num_rows when the dataset is only partially processed.
    When config is None, sums rows across all configs for the given split.
    """
    try:
        resp = requests.get(
            "https://datasets-server.huggingface.co/size",
            params={"dataset": hf_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        partial = data.get("partial", False)

        total = 0
        for s in data["size"]["splits"]:
            if s["split"] != split:
                continue
            if config is not None and s["config"] != config:
                continue
            # Prefer estimated_num_rows for partially processed datasets
            if partial and "estimated_num_rows" in s:
                total += s["estimated_num_rows"]
            else:
                total += s["num_rows"]
        return total if total > 0 else None
    except Exception:
        pass
    return None


def _num_rows_from_parquet_metadata(hf_id: str, config: str | None) -> int:
    """Get exact row count by reading all parquet file metadata footers.

    Lists all parquet files via the HF API, then reads each file's
    lightweight metadata footer (a few KB, no data download) in parallel
    via HfFileSystem + pyarrow to sum the exact row counts.
    """
    api = HfApi()

    # List all parquet files in the repo
    logger.info("  Listing parquet files in repo (may take a moment)...")
    parquet_paths: list[str] = []
    for item in tqdm(
        api.list_repo_tree(hf_id, recursive=True, repo_type="dataset"),
        desc="  Listing files",
        unit="file",
    ):
        if hasattr(item, "path") and item.path.endswith(".parquet"):
            if config is None or item.path.startswith(f"{config}/"):
                parquet_paths.append(item.path)

    n_files = len(parquet_paths)
    if n_files == 0:
        raise ValueError(f"No parquet files found for {hf_id} config={config}")

    logger.info(f"  Found {n_files:,} parquet files, reading all metadata footers...")

    # Read all parquet metadata footers in parallel (each is a few KB)
    fs = HfFileSystem()

    def _read_num_rows(path: str) -> int:
        return pq.read_metadata(f"datasets/{hf_id}/{path}", filesystem=fs).num_rows

    # Cap concurrency to avoid HF Hub rate limits (429s)
    row_counts = Parallel(n_jobs=16, backend="threading")(
        delayed(_read_num_rows)(p)
        for p in tqdm(parquet_paths, desc="  Reading parquet metadata", unit="file")
    )

    total = sum(row_counts)
    logger.info(f"  Exact row count: {total:,} (from {n_files:,} parquet files)")
    return total


def get_num_rows(hf_id: str, config: str | None, split: str) -> int:
    """Get total row count, trying three methods in order of speed."""
    logger.info(f"  Fetching row count for {hf_id} (config={config})...")

    # Method 1: dataset builder metadata
    n = _num_rows_from_builder(hf_id, config, split)
    if n is not None:
        logger.info(f"  Row count (from builder): {n:,}")
        return n

    # Method 2: datasets server API
    logger.info("  Builder had no split info, trying datasets server API...")
    n = _num_rows_from_server_api(hf_id, config, split)
    if n is not None:
        logger.info(f"  Row count (from API): {n:,}")
        return n

    # Method 3: read all parquet metadata footers
    logger.info("  API had no info, falling back to parquet metadata...")
    return _num_rows_from_parquet_metadata(hf_id, config)


def estimate_avg_tokens(
    tokenizer: Tokenizer,
    hf_id: str,
    config: str | None,
    split: str,
) -> tuple[float, int]:
    """Stream docs, batch-tokenize, return (avg_tokens_per_doc, n_docs_sampled).

    Stops early once the running average stabilizes (relative change
    between consecutive windows falls below THRESHOLD).
    """
    kwargs: dict = {"streaming": True, "split": split}
    if config:
        kwargs["name"] = config
    logger.info(f"  Loading streaming dataset {hf_id}...")
    ds = load_dataset(hf_id, **kwargs)
    logger.info("  Stream ready, starting estimation...")

    total_tokens = 0
    n_docs = 0
    prev_avg: float | None = None
    batch_texts: list[str] = []
    t0 = time.time()

    pbar = tqdm(total=MAX_DOCS, desc=f"  {hf_id}", unit="doc", miniters=1)

    for doc in ds:
        batch_texts.append(doc[TEXT_KEY])
        if len(batch_texts) < BATCH:
            continue

        # Batch tokenize via tokenizers (pure Rust, no torch)
        encoded = tokenizer.encode_batch(batch_texts)
        batch_tokens = sum(len(e.ids) for e in encoded)
        total_tokens += batch_tokens
        n_docs += len(batch_texts)
        batch_texts = []

        # Update progress bar with running tallies
        avg = total_tokens / n_docs
        elapsed = time.time() - t0
        pbar.set_postfix(
            avg_tok=f"{avg:.0f}",
            total_tok=f"{total_tokens / 1e6:.0f}M",
            doc_s=f"{n_docs / elapsed:.0f}",
            ordered=True,
        )
        pbar.update(BATCH)

        # Check convergence after MIN_DOCS
        if n_docs >= MIN_DOCS:
            if prev_avg is not None:
                rel_change = abs(avg - prev_avg) / avg
                if rel_change < THRESHOLD:
                    pbar.close()
                    logger.info(
                        f"  Converged: n={n_docs:,}  avg_tok={avg:.1f}  Δ={rel_change:.6f}"
                    )
                    return avg, n_docs
            prev_avg = avg

        if n_docs >= MAX_DOCS:
            pbar.close()
            logger.info(f"  Hit max docs ({MAX_DOCS:,}), stopping")
            break

    pbar.close()

    # Process remaining docs in buffer
    if batch_texts:
        encoded = tokenizer.encode_batch(batch_texts)
        total_tokens += sum(len(e.ids) for e in encoded)
        n_docs += len(batch_texts)

    return total_tokens / n_docs, n_docs


def _human_number(n: int | float) -> str:
    """Format large number as e.g. '16.9T', '726.1B', '23.0M'."""
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n / 1e3:.1f}K"


def main() -> None:
    logger.info(f"Loading tokenizer {TOKENIZER_NAME}...")
    tokenizer = Tokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    logger.info("Tokenizer loaded.")

    results: dict[str, dict] = {}
    for i, (name, (hf_id, config, split)) in enumerate(SOURCES.items(), 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[{i}/{len(SOURCES)}] Dataset: {name} ({hf_id})")
        logger.info(f"{'=' * 60}")

        num_rows = get_num_rows(hf_id, config, split)
        avg_tokens, n_sampled = estimate_avg_tokens(tokenizer, hf_id, config, split)
        total_tokens = round(avg_tokens * num_rows)

        results[name] = {
            "avg_tokens_per_doc": avg_tokens,
            "n_sampled": n_sampled,
            "num_rows": num_rows,
            "total_tokens": total_tokens,
        }

        logger.info(f"  avg_tokens/doc: {avg_tokens:,.0f}")
        logger.info(f"  sampled docs:   {n_sampled:,}")
        logger.info(f"  total docs:     {num_rows:,} ({_human_number(num_rows)})")
        logger.info(f"  total tokens:   {total_tokens:,} ({_human_number(total_tokens)})")

    # Summary table
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    for name, r in results.items():
        logger.info(
            f"  {name:>15s}: {_human_number(r['total_tokens']):>8s} tokens  "
            f"[avg={r['avg_tokens_per_doc']:.0f} tok/doc × {_human_number(r['num_rows'])} docs]"
        )


if __name__ == "__main__":
    main()
