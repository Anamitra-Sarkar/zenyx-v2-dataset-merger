"""Text normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def is_str(x: Any) -> bool:
    return isinstance(x, str)


def join_nonempty(parts: Iterable[str], sep: str = "\n") -> str:
    out = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return sep.join(out).strip()


def render_messages(messages: Any) -> Optional[str]:
    """Render chat-style messages into a single training text.

    Accepts list of dicts like {role, content}.
    """
    if not isinstance(messages, list):
        return None

    rendered = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if role:
            rendered.append(f"{role}: {content}")
        else:
            rendered.append(content)

    txt = "\n".join(rendered).strip()
    return txt or None


def coerce_text(example: Dict[str, Any], candidates: list[str]) -> Optional[str]:
    """Best-effort extraction of training text from heterogeneous schemas."""
    for k in candidates:
        if k not in example:
            continue
        v = example.get(k)

        if isinstance(v, str) and v.strip():
            return v.strip()

        # some datasets store nested dicts
        if isinstance(v, dict):
            for kk in ("text", "content", "value", "final", "answer"):
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()

        # chat format
        if k in ("messages", "conversation"):
            txt = render_messages(v)
            if txt:
                return txt

    # common pattern: instruction + input + output
    ins = example.get("instruction")
    inp = example.get("input")
    out = example.get("output") or example.get("response") or example.get("completion")
    if isinstance(ins, str) and isinstance(out, str):
        if isinstance(inp, str) and inp.strip():
            return join_nonempty(["Instruction: " + ins, "Input: " + inp, "Output: " + out])
        return join_nonempty(["Instruction: " + ins, "Output: " + out])

    # question + solution
    q = example.get("question") or example.get("problem")
    a = example.get("answer") or example.get("solution")
    if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
        return join_nonempty(["Question: " + q, "Answer: " + a])

    return None
