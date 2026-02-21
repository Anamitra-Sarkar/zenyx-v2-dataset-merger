"""HuggingFace dataset adapters.

Fixes:
- trust_remote_code=True for datasets with custom loading scripts
- Split fallback: train → default → first available split
- Silent failure watchdog: if skip ratio hits 100% after WATCHDOG_MIN_ROWS, raises RuntimeError

Unified example:
{
  "text": "...",
  "source": "org/name",
  "subset": null,
  "split": "train",
  "meta": {...}
}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from datasets import load_dataset, get_dataset_split_names

from .textnorm import coerce_text

log = logging.getLogger("ZenyxMerger")

# After this many rows, if ALL of them were skipped → raise
WATCHDOG_MIN_ROWS = 500


@dataclass
class DatasetSpec:
    id: str
    subset: Optional[str]
    split: str


def _resolve_split(spec: DatasetSpec, token: Optional[str]) -> str:
    """Return the best available split for this dataset."""
    desired = spec.split or "train"
    try:
        available = get_dataset_split_names(
            spec.id,
            config_name=spec.subset,
            trust_remote_code=True,
            token=token,
        )
    except Exception:
        return desired  # can't query — try as-is

    if desired in available:
        return desired
    # fallbacks
    for fallback in ("train", "default", "all"):
        if fallback in available:
            log.warning(
                f"[{spec.id}] split '{desired}' not found, "
                f"falling back to '{fallback}'. Available: {available}"
            )
            return fallback
    # last resort — first split
    log.warning(
        f"[{spec.id}] no standard split found, using first: '{available[0]}'. "
        f"Available: {available}"
    )
    return available[0]


def iter_dataset(
    spec: DatasetSpec,
    *,
    streaming: bool,
    token: Optional[str],
    text_key_candidates: list[str],
) -> Iterator[Dict[str, Any]]:

    split = _resolve_split(spec, token)

    try:
        ds = load_dataset(
            spec.id,
            spec.subset,
            split=split,
            streaming=streaming,
            token=token,
            trust_remote_code=True,
        )
    except Exception as e:
        log.error(f"[{spec.id}] Failed to load: {e}")
        return

    total   = 0
    skipped = 0

    for ex in ds:
        total += 1
        txt = coerce_text(ex, text_key_candidates)

        if not txt:
            skipped += 1
            # Watchdog: 100% skip rate after WATCHDOG_MIN_ROWS
            if total >= WATCHDOG_MIN_ROWS and skipped == total:
                raise RuntimeError(
                    f"[{spec.id}] SILENT FAILURE DETECTED: "
                    f"all {total} rows skipped — schema likely unrecognised. "
                    f"Sample keys: {list(ex.keys())}"
                )
            continue

        yield {
            "text":   txt,
            "source": spec.id,
            "subset": spec.subset or "",
            "split":  split,
            "meta":   {},
        }

    if total > 0:
        skip_pct = 100.0 * skipped / total
        if skip_pct > 50.0:
            log.warning(
                f"[{spec.id}] High skip rate: {skipped}/{total} rows skipped ({skip_pct:.1f}%). "
                f"Check schema."
            )
        else:
            log.info(f"[{spec.id}] Done: {total - skipped:,} rows yielded, {skipped:,} skipped.")
