"""Stratified frozen splits (SPEC §5 P2, configs/splits.yaml). Refuses to overwrite existing
splits/*.parquet once frozen. temporal_holdout: latest temporal_holdout_frac of each grade by
created date, flagged (not removed) so downstream code can opt in/out.
Usage: from wqa.features.splits import run; run("small")."""
from __future__ import annotations

from typing import Any

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from wqa import config, manifest
from wqa.config import ROOT

log = structlog.get_logger()


def run(mode: str) -> dict[str, int]:
    cfg = config.load("splits")
    out_dir = ROOT / "data" / mode / "splits"
    existing = [out_dir / f"{s}.parquet" for s in ("train", "val", "test")]
    if cfg.get("freeze") and any(p.exists() for p in existing):
        raise FileExistsError(f"splits already frozen under {out_dir}; refusing to overwrite")

    features = pd.read_parquet(ROOT / "data" / mode / "features" / "features.parquet")
    interim = pd.read_parquet(ROOT / "data" / mode / "interim" / "full.parquet",
                              columns=["pageid", "created"])
    df = features.merge(interim, on="pageid", how="left")
    df["temporal_holdout"] = _temporal_holdout_flag(df, cfg["stratify"], cfg["temporal_holdout_frac"])

    seed = cfg["seed"]
    train_df, rest = train_test_split(df, test_size=cfg["val"] + cfg["test"], random_state=seed,
                                      stratify=df[cfg["stratify"]])
    val_frac = cfg["val"] / (cfg["val"] + cfg["test"])
    val_df, test_df = train_test_split(rest, test_size=1 - val_frac, random_state=seed,
                                       stratify=rest[cfg["stratify"]])

    out_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        p = out_dir / f"{name}.parquet"
        part.drop(columns=["created"]).to_parquet(p, index=False)
        manifest.update_dataset(mode, name=f"splits_{name}", stage="splits", file=p,
                                rows=len(part), cols=part.shape[1] - 1, producer_target="splits",
                                parent="features")
        sizes[name] = len(part)
    log.info("splits_done", **sizes)
    return sizes


def _temporal_holdout_flag(df: pd.DataFrame, stratify_col: str, frac: float) -> pd.Series:
    """Latest `frac` of each stratify_col group by created date -> True."""
    created = pd.to_datetime(df["created"], utc=True, errors="coerce")
    flag = pd.Series(False, index=df.index)
    for _, idx in df.groupby(stratify_col).groups.items():
        grp_dates = created.loc[idx].sort_values()
        cutoff_n = max(1, int(len(grp_dates) * frac))
        flag.loc[grp_dates.index[-cutoff_n:]] = True
    return flag
