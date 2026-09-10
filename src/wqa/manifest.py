"""data/<mode>/manifest.json helpers (SPEC §4). Idempotent: dataset rows keyed by name.
In: mode, dataset row fields. Out: updated manifest. Usage: update_dataset("small", name=..., ...)."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wqa.config import ROOT


def path(mode: str) -> Path:
    return ROOT / "data" / mode / "manifest.json"


def load(mode: str) -> dict[str, Any]:
    p = path(mode)
    if p.exists():
        out: dict[str, Any] = json.loads(p.read_text())
        return out
    return {"mode": mode, "updated": "", "target_per_grade": None, "per_grade": {},
            "excluded": {}, "datasets": []}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_dataset(mode: str, *, name: str, stage: str, file: Path, rows: int, cols: int,
                   producer_target: str, parent: str | None = None) -> None:
    m = load(mode)
    row = {"name": name, "stage": stage, "path": str(file.relative_to(ROOT)), "rows": rows,
           "cols": cols, "bytes": file.stat().st_size, "sha256": sha256(file),
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "producer_target": producer_target, "parent": parent}
    m["datasets"] = [d for d in m["datasets"] if d["name"] != name] + [row]
    m["updated"] = row["created"]
    save(mode, m)


def save(mode: str, m: dict[str, Any]) -> None:
    p = path(mode)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=1))
