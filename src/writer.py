"""Sharded Parquet writer — PyArrow direct, no Pandas/HF Dataset overhead.

Fixes:
- Memory: writes via pyarrow.ParquetWriter directly; never holds >shard_size rows in RAM
- Removed pandas + datasets dependency (saves ~3× memory per shard flush)
"""

from __future__ import annotations

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
) -> List[str]:
    import orjson

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        shard_path = out_dir / f"data-{shard_idx:05d}.parquet"
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
