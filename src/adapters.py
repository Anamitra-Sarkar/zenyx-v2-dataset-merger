"""HuggingFace dataset adapters.

We keep this module small but extensible.
Each adapter returns an iterable of unified dicts.

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

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional

from datasets import load_dataset

from .textnorm import coerce_text


@dataclass
class DatasetSpec:
    id: str
    subset: Optional[str]
    split: str


def iter_dataset(spec: DatasetSpec, *, streaming: bool, token: Optional[str], text_key_candidates: list[str]) -> Iterator[Dict[str, Any]]:
    ds = load_dataset(spec.id, spec.subset, split=spec.split, streaming=streaming, token=token)

    for ex in ds:
        txt = coerce_text(ex, text_key_candidates)
        if not txt:
            continue
        yield {
            "text": txt,
            "source": spec.id,
            "subset": spec.subset,
            "split": spec.split,
            "meta": {},
        }
