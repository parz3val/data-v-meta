"""P2 features: data/<mode>/interim/full.parquet -> data/<mode>/features/features.parquet
(SPEC §5 P2, configs/features.yaml). Reads interim only -- wikitext is parsed once in
wqa.clean, features here are pure arithmetic over its output.
Usage: from wqa.features.build import run; run("small")."""
from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from wqa import config, manifest
from wqa.config import ROOT
from wqa.features import history as history_family
from wqa.features import reference as reference_family
from wqa.features import structural as structural_family
from wqa.monitor.budget import budget

log = structlog.get_logger()
STRUCTURAL_LIKE = {"structural": structural_family.FEATURES, "reference": reference_family.FEATURES}
# g(H) = history + contributor + pageviews (THEORY split, configs/features.yaml comment).
GH_FEATURES = set(history_family.FEATURES)


def run(mode: str) -> dict[str, Any]:
    with budget("P2_clean"):
        cfg = config.load("features")
        interim = pd.read_parquet(ROOT / "data" / mode / "interim" / "full.parquet")
        _check_named(cfg)
        out = pd.DataFrame({"pageid": interim["pageid"], "label": interim["label"],
                            "grade_rank": interim["grade_rank"]})
        for fam in ("structural", "reference", "history", "contributor", "pageviews"):
            funcs = _family_funcs(fam)
            for name in cfg[fam]:
                out[name] = funcs[name](interim)
        out_dir = ROOT / "data" / mode / "features"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "features.parquet"
        out.to_parquet(path, index=False)
        manifest.update_dataset(mode, name="features", stage="features", file=path,
                                rows=len(out), cols=len(out.columns),
                                producer_target="features", parent="interim")
    log.info("features_done", rows=len(out), cols=len(out.columns))
    return {"rows": len(out), "cols": len(out.columns)}


def _family_funcs(spec_family: str) -> dict[str, Any]:
    if spec_family in ("contributor", "pageviews", "history"):
        return history_family.FEATURES
    return STRUCTURAL_LIKE[spec_family]


def _check_named(cfg: dict[str, Any]) -> None:
    """Every feature named in configs/features.yaml must have an implementing function."""
    all_funcs = {**structural_family.FEATURES, **reference_family.FEATURES, **history_family.FEATURES}
    named = [n for fam in ("structural", "reference", "history", "contributor", "pageviews")
            for n in cfg[fam]]
    missing = [n for n in named if n not in all_funcs]
    if missing:
        raise KeyError(f"features.yaml names not implemented: {missing}")
