"""Sharded parquet writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from datasets import Dataset


def write_shards(examples: Iterable[Dict[str, Any]], out_dir: str | Path, shard_size: int) -> List[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: List[str] = []
    buf: List[Dict[str, Any]] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal shard_idx
        if not buf:
            return
        df = pd.DataFrame(buf)
        ds = Dataset.from_pandas(df, preserve_index=False)
        shard_path = out_dir / f"data-{shard_idx:05d}.parquet"
        ds.to_parquet(str(shard_path))
        shard_paths.append(str(shard_path))
        buf.clear()
        shard_idx += 1

    for ex in examples:
        buf.append(ex)
        if len(buf) >= shard_size:
            flush()

    flush()
    return shard_paths
