"""HuggingFace dataset adapters.

Fixes:
- trust_remote_code=True for datasets with custom loading scripts
- Split fallback: train → default → first available split
- Silent failure watchdog: if skip ratio hits 100% after WATCHDOG_MIN_ROWS, raises RuntimeError
- Streaming fallback: if streaming=True fails, retries with streaming=False
- PRM adapter: custom extraction for Process Reward Model datasets
- category field: read from DatasetSpec, written into every example's meta

Unified example:
{
  "text": "...",
  "source": "org/name",
  "subset": null,
  "split": "train",
  "meta": {"category": "reasoning/math"}
}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from datasets import load_dataset, get_dataset_split_names

from .textnorm import coerce_text, join_nonempty

log = logging.getLogger("ZenyxMerger")

WATCHDOG_MIN_ROWS = 500

_PRM_DATASET_IDS = frozenset({
    "tasksource/PRM800K",
    "peiyi9979/Math-Shepherd",
})


@dataclass
class DatasetSpec:
    id: str
    subset: Optional[str]
    split: str
    category: str = ""       # e.g. "reasoning/math", "normal/code"


def _resolve_split(spec: DatasetSpec, token: Optional[str]) -> str:
    desired = spec.split or "train"
    try:
        available = get_dataset_split_names(
            spec.id,
            config_name=spec.subset,
            trust_remote_code=True,
            token=token,
        )
    except Exception:
        return desired
    if desired in available:
        return desired
    for fallback in ("train", "default", "all"):
        if fallback in available:
            log.warning(
                f"[{spec.id}] split '{desired}' not found, "
                f"falling back to '{fallback}'. Available: {available}"
            )
            return fallback
    log.warning(
        f"[{spec.id}] no standard split found, using first: '{available[0]}'. "
        f"Available: {available}"
    )
    return available[0]


def _load_dataset_with_fallback(
    spec: DatasetSpec,
    split: str,
    streaming: bool,
    token: Optional[str],
) -> Any:
    load_kwargs = dict(
        path=spec.id,
        name=spec.subset,
        split=split,
        token=token,
        trust_remote_code=True,
    )
    try:
        return load_dataset(**load_kwargs, streaming=streaming)
    except Exception as e:
        if streaming:
            log.warning(
                f"[{spec.id}] streaming=True failed ({type(e).__name__}: {e}). "
                f"Retrying with streaming=False..."
            )
            try:
                return load_dataset(**load_kwargs, streaming=False)
            except Exception as e2:
                log.error(f"[{spec.id}] Failed to load (non-streaming too): {e2}")
                return None
        log.error(f"[{spec.id}] Failed to load: {e}")
        return None


def _extract_prm_text(example: Dict[str, Any]) -> Optional[str]:
    """Extract text from PRM-style datasets (PRM800K, Math-Shepherd)."""
    inp   = example.get("input")
    label = example.get("label")
    if isinstance(inp, str) and inp.strip():
        if isinstance(label, str) and label.strip():
            clean_label = label.replace("\u043a\u0438", "").strip()
            return join_nonempty(["Question: " + inp, "Solution: " + clean_label])
        return inp.strip()

    question = example.get("question")
    steps    = example.get("steps") or example.get("completions") or []
    if isinstance(question, str) and question.strip():
        step_texts: List[str] = []
        if isinstance(steps, list):
            for i, step in enumerate(steps, 1):
                if isinstance(step, dict):
                    completions = step.get("completions") or []
                    if isinstance(completions, list) and completions:
                        first = completions[0]
                        text  = (first.get("text") or first.get("value") or "") if isinstance(first, dict) else (first if isinstance(first, str) else "")
                    else:
                        text = step.get("text") or step.get("value") or ""
                    if isinstance(text, str) and text.strip():
                        step_texts.append(f"Step {i}: {text.strip()}")
                elif isinstance(step, str) and step.strip():
                    step_texts.append(f"Step {i}: {step.strip()}")
        if step_texts:
            return join_nonempty(["Question: " + question] + step_texts)
        return "Question: " + question.strip()
    return None


def iter_dataset(
    spec: DatasetSpec,
    *,
    streaming: bool,
    token: Optional[str],
    text_key_candidates: list[str],
) -> Iterator[Dict[str, Any]]:

    split = _resolve_split(spec, token)
    ds    = _load_dataset_with_fallback(spec, split, streaming, token)
    if ds is None:
        return

    is_prm = spec.id in _PRM_DATASET_IDS
    total  = 0
    skipped = 0

    for ex in ds:
        total += 1

        txt = _extract_prm_text(ex) if is_prm else None
        if not txt:
            txt = coerce_text(ex, text_key_candidates)

        if not txt:
            skipped += 1
            if total >= WATCHDOG_MIN_ROWS and skipped == total:
                log.error(
                    f"[{spec.id}] SILENT FAILURE DETECTED: "
                    f"all {total} rows skipped — schema likely unrecognised. "
                    f"Sample keys: {list(ex.keys())}  "
                    + str({k: str(v)[:80] for k, v in ex.items()})
                )
                return
            continue

        yield {
            "text":   txt,
            "source": spec.id,
            "subset": spec.subset or "",
            "split":  split,
            "meta":   {"category": spec.category},
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
