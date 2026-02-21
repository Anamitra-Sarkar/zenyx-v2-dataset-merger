"""Text normalization helpers.

Fixes:
- render_messages: handles ShareGPT format (from/value) alongside OpenAI format (role/content)
- coerce_text: checks 'conversations' (plural) key for ShareGPT datasets
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def is_str(x: Any) -> bool:
    return isinstance(x, str)


def join_nonempty(parts: Iterable[str], sep: str = "\n") -> str:
    out = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return sep.join(out).strip()


def render_messages(messages: Any) -> Optional[str]:
    """Render chat-style messages into a single training text.

    Handles both:
    - OpenAI format: [{"role": "user", "content": "..."}]
    - ShareGPT format: [{"from": "human", "value": "..."}]
    """
    if not isinstance(messages, list):
        return None

    rendered = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        # OpenAI style first, ShareGPT fallback
        role    = m.get("role") or m.get("from") or ""
        content = m.get("content") or m.get("value")
        if not isinstance(content, str) or not content.strip():
            continue
        if role:
            rendered.append(f"{role}: {content.strip()}")
        else:
            rendered.append(content.strip())

    txt = "\n".join(rendered).strip()
    return txt or None


# Chat-style keys — both singular and plural, OpenAI and ShareGPT
_CHAT_KEYS = frozenset(("messages", "conversation", "conversations", "chat"))


def coerce_text(example: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    """Best-effort extraction of training text from heterogeneous schemas."""

    for k in candidates:
        if k not in example:
            continue
        v = example.get(k)

        if isinstance(v, str) and v.strip():
            return v.strip()

        # nested dict — try common sub-keys
        if isinstance(v, dict):
            for kk in ("text", "content", "value", "final", "answer"):
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()

        # chat / conversation format (OpenAI + ShareGPT)
        if k in _CHAT_KEYS and isinstance(v, list):
            txt = render_messages(v)
            if txt:
                return txt

    # also check top-level 'conversations' even if not in candidates
    for ck in _CHAT_KEYS:
        if ck in example and ck not in candidates:
            txt = render_messages(example[ck])
            if txt:
                return txt

    # instruction + (optional input) + output
    ins = example.get("instruction")
    inp = example.get("input")
    out = (
        example.get("output")
        or example.get("response")
        or example.get("completion")
    )
    if isinstance(ins, str) and ins.strip() and isinstance(out, str) and out.strip():
        if isinstance(inp, str) and inp.strip():
            return join_nonempty(["Instruction: " + ins, "Input: " + inp, "Output: " + out])
        return join_nonempty(["Instruction: " + ins, "Output: " + out])

    # question + solution/answer
    q = example.get("question") or example.get("problem")
    a = example.get("answer") or example.get("solution")
    if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
        return join_nonempty(["Question: " + q, "Answer: " + a])

    return None
