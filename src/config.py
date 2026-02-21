"""Utilities for loading YAML/JSON configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import orjson
import yaml  # pyyaml — listed in requirements.txt


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return orjson.loads(read_text(path))


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
