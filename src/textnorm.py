"""Text normalization helpers.

Fixes:
- render_messages: handles ShareGPT format (from/value) alongside OpenAI format (role/content)
- render_messages: handles nested-list content (multimodal / newer HF datasets)
- coerce_text: CRITICAL FIX — pairs and Q/A checked BEFORE single-key candidates to prevent
  silent data loss where e.g. MetaMathQA's 'response' key is picked up alone,
  dropping the 'query' (the question) entirely
- coerce_text: added 'query' to Q/A pair detection
- coerce_text: checks 'conversations' (plural) key for ShareGPT datasets
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def is_str(x: Any) -> bool:
    return isinstance(x, str)


def join_nonempty(parts: Iterable[str], sep: str = "\n") -> str:
    out = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return sep.join(out).strip()


def _extract_content(content: Any) -> Optional[str]:
    """Extract a plain string from a content field that may be:
    - a plain string
    - a list of dicts like [{"type": "text", "text": "..."}, ...] (multimodal)
    """
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                # multimodal / OpenAI vision format
                text_val = item.get("text") or item.get("value") or item.get("content")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip() or None
    return None


def render_messages(messages: Any) -> Optional[str]:
    """Render chat-style messages into a single training text.

    Handles:
    - OpenAI format:  [{"role": "user", "content": "..."}]
    - ShareGPT format: [{"from": "human", "value": "..."}]
    - Multimodal content: [{"role": "user", "content": [{"type": "text", "text": "..."}]}]
    """
    if not isinstance(messages, list):
        return None

    rendered = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role    = m.get("role") or m.get("from") or ""
        content = m.get("content") or m.get("value")
        # Use _extract_content so nested lists are handled gracefully
        text = _extract_content(content)
        if not text:
            continue
        if role:
            rendered.append(f"{role}: {text}")
        else:
            rendered.append(text)

    txt = "\n".join(rendered).strip()
    return txt or None


# Chat-style keys — both singular and plural, OpenAI and ShareGPT
_CHAT_KEYS = frozenset(("messages", "conversation", "conversations", "chat"))


def coerce_text(example: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    """Best-effort extraction of training text from heterogeneous schemas.

    Priority order (high → low):
    1. Chat / conversation formats  — always structurally complete
    2. Paired fields (instruction+output, question+answer)  — preserve Q+A together
    3. Single-key candidates  — last resort, avoids swallowing half a pair
    """

    # ── 1. Chat / conversation formats ───────────────────────────────────────
    # Check both candidates that are chat keys AND top-level chat keys not in candidates.
    for ck in _CHAT_KEYS:
        if ck in example:
            txt = render_messages(example[ck])
            if txt:
                return txt

    # ── 2a. Instruction + (optional Input) + Output ───────────────────────────
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

    # ── 2b. Question / Problem / Query  +  Answer / Solution / Response ───────
    # NOTE: 'query' added here — fixes MetaMathQA and similar datasets that use
    # query/response keys (previously 'response' was grabbed alone by candidate loop)
    q = (
        example.get("question")
        or example.get("problem")
        or example.get("query")
    )
    a = (
        example.get("answer")
        or example.get("solution")
        or example.get("response")
    )
    if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
        return join_nonempty(["Question: " + q, "Answer: " + a])

    # ── 3. Single-key candidates (last resort) ────────────────────────────────
    for k in candidates:
        if k not in example:
            continue
        v = example[k]

        if isinstance(v, str) and v.strip():
            return v.strip()

        # nested dict — try common sub-keys
        if isinstance(v, dict):
            for kk in ("text", "content", "value", "final", "answer"):
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()

        # chat / conversation list in a non-standard key
        if isinstance(v, list):
            txt = render_messages(v)
            if txt:
                return txt

    return None
