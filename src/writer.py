"""Sharded Parquet writer — PyArrow direct, no Pandas/HF Dataset overhead.

Fixes:
- Memory: writes via pyarrow.ParquetWriter directly; never holds >shard_size rows in RAM
- Removed pandas + datasets dependency (saves ~3× memory per shard flush)
- CRITICAL FIX: shard filenames are now globally unique via uuid prefix to prevent
  cross-dataset shard overwriting when write_shards is called in a loop
- Guard against shard_size <= 0 which would cause an infinite flush loop
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq

# Unified output schema
SCHEMA = pa.schema([
    pa.field("text",   pa.string()),
    pa.field("source", pa.string()),
    pa.field("subset", pa.string()),
    pa.field("split",  pa.string()),
    pa.field("meta",   pa.string()),   # JSON-serialised dict
])


def write_shards(
    examples: Iterable[Dict[str, Any]],
    out_dir: str | Path,
    shard_size: int,
    *,
    dataset_prefix: str = "",
) -> List[str]:
    """
    Write examples to sharded Parquet files.

    Parameters
    ----------
    examples       : iterable of unified dicts
    out_dir        : directory to write shards into
    shard_size     : max rows per shard (must be >= 1)
    dataset_prefix : optional human-readable prefix (sanitised dataset name);
                     combined with a uuid4 hex to guarantee globally unique filenames
                     even when this function is called many times in the same out_dir.
    """
    import orjson

    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a stable, filesystem-safe prefix for this call's shard files.
    # Format: <sanitised_name>-<8-char-uuid>  e.g. "MetaMathQA-3f9a1c2b"
    uid = uuid.uuid4().hex[:8]
    if dataset_prefix:
        # Replace characters that are problematic in filenames
        safe_prefix = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in dataset_prefix
        )[:40]  # cap length
        shard_prefix = f"{safe_prefix}-{uid}"
    else:
        shard_prefix = uid

    shard_paths: List[str] = []
    buf: List[Dict[str, Any]] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal shard_idx
        if not buf:
            return
        arrays = {
            "text":   pa.array([r["text"]   for r in buf], type=pa.string()),
            "source": pa.array([r["source"] for r in buf], type=pa.string()),
            "subset": pa.array([r.get("subset") or "" for r in buf], type=pa.string()),
            "split":  pa.array([r["split"]  for r in buf], type=pa.string()),
            "meta":   pa.array([orjson.dumps(r.get("meta") or {}).decode() for r in buf], type=pa.string()),
        }
        table = pa.table(arrays, schema=SCHEMA)
        # e.g. data-MetaMathQA-3f9a1c2b-00000.parquet
        shard_path = out_dir / f"data-{shard_prefix}-{shard_idx:05d}.parquet"
        pq.write_table(table, str(shard_path), compression="zstd")
        shard_paths.append(str(shard_path))
        buf.clear()
        shard_idx += 1

    for ex in examples:
        buf.append(ex)
        if len(buf) >= shard_size:
            flush()

    flush()
    return shard_paths
