"""Config loading. In: configs/<name>.yaml. Out: dict. Usage: cfg = load("collection")."""
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> dict[str, Any]:
    with (ROOT / "configs" / f"{name}.yaml").open() as f:
        out: dict[str, Any] = yaml.safe_load(f)
    return out


def user_agent(cfg: dict[str, Any]) -> str:
    email = (cfg.get("email") or "").strip()
    if not email:
        raise SystemExit("blank email in configs/collection.yaml (SPEC §4.3): refusing to run")
    ua: str = cfg["user_agent"].format(email=email)
    return ua
