"""JSONL persistence: data/<mode>/raw/<grade>.jsonl + raw/index.json {pageid: {file,offset,length}}
(SPEC §4 layout, §5 P1). Resume = read existing pageids from the grade file, no refetch."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wqa.config import ROOT


def raw_dir(mode: str) -> Path:
    d = ROOT / "data" / mode / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def existing_pageids(mode: str, grade: str) -> set[int]:
    p = raw_dir(mode) / f"{grade}.jsonl"
    if not p.exists():
        return set()
    ids: set[int] = set()
    with p.open() as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["pageid"])
    return ids


def append(mode: str, grade: str, record: dict[str, Any], index: dict[str, Any]) -> None:
    """Append one record; record its byte range in `index` (mutated in place)."""
    p = raw_dir(mode) / f"{grade}.jsonl"
    line = json.dumps(record, ensure_ascii=False) + "\n"
    offset = p.stat().st_size if p.exists() else 0
    with p.open("a", encoding="utf-8") as f:
        f.write(line)
    index[str(record["pageid"])] = {"file": f"raw/{grade}.jsonl", "offset": offset,
                                    "length": len(line.encode("utf-8"))}


def load_index(mode: str) -> dict[str, Any]:
    p = raw_dir(mode) / "index.json"
    return dict(json.loads(p.read_text())) if p.exists() else {}


def save_index(mode: str, index: dict[str, Any]) -> None:
    (raw_dir(mode) / "index.json").write_text(json.dumps(index, indent=1))
