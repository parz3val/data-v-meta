"""P2 clean: data/<mode>/raw/*.jsonl -> data/<mode>/interim/{full,sample}.parquet (SPEC §5 P2).
One row per article; wikitext parsed once here so wqa.features stays arithmetic-only.
Usage: from wqa.clean.run import run; run("small")."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import structlog

from wqa import manifest
from wqa.clean.parse import parse_article
from wqa.collect.assess import RANK
from wqa.collect.persist import raw_dir
from wqa.config import ROOT
from wqa.monitor.budget import budget

log = structlog.get_logger()
GRADES = list(RANK)
SAMPLE_ROWS = 30


def run(mode: str) -> dict[str, Any]:
    with budget("P2_clean"):
        rows = list(_iter_raw(mode))
        df = pd.DataFrame(rows)
        n0 = len(df)
        # collector resume can re-append a page; keep first occurrence (fix 2026-08-25)
        df = df.drop_duplicates("pageid", keep="first").reset_index(drop=True)
        if len(df) != n0:
            print(f"clean: dropped {n0 - len(df)} duplicate pageid rows")
        out_dir = ROOT / "data" / mode / "interim"
        out_dir.mkdir(parents=True, exist_ok=True)
        full_path = out_dir / "full.parquet"
        df.to_parquet(full_path, index=False)
        sample = _sample(df)
        sample_path = out_dir / "sample.parquet"
        sample.to_parquet(sample_path, index=False)
        manifest.update_dataset(mode, name="interim", stage="interim", file=full_path,
                                rows=len(df), cols=len(df.columns), producer_target="clean-data")
        manifest.update_dataset(mode, name="sample", stage="sample", file=sample_path,
                                rows=len(sample), cols=len(sample.columns),
                                producer_target="clean-data", parent="interim")
    log.info("clean_done", rows=len(df), cols=len(df.columns))
    return {"rows": len(df), "cols": len(df.columns)}


def _iter_raw(mode: str):
    for grade in GRADES:
        p = raw_dir(mode) / f"{grade}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            yield _clean_record(rec)


def _clean_record(rec: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_article(rec["wikitext"], rec.get("categories", []), rec.get("templates", []))
    return {
        "pageid": rec["pageid"], "title": rec["title"], "label": rec["label"],
        "grade_rank": RANK[rec["label"]], "bytes": rec["bytes"], "revid": rec["revid"],
        "n_projects": rec["n_projects"], "n_disagree": rec["n_disagree"],
        "created": rec.get("created"), "last_edit": rec.get("last_edit"),
        "fetch_ts": rec.get("fetch_ts"), "edits": rec["edits"], "editors": rec["editors"],
        "anon_n": rec["anon_n"], "bot_n": rec["bot_n"], "reverted_n": rec["reverted_n"],
        "edits_90d": rec["edits_90d"], "pageviews_90d_mean": rec["pageviews_90d_mean"],
        **parsed,
    }


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    """<= SAMPLE_ROWS rows, stratified across grades, columns per UI_SPEC §4b."""
    per_grade = max(1, SAMPLE_ROWS // len(GRADES))
    parts = [g.head(per_grade) for _, g in df.groupby("label", sort=False)]
    picked = pd.concat(parts).head(SAMPLE_ROWS) if parts else df.head(0)
    cols = ["pageid", "title", "label", "bytes", "ref_n", "edits", "n_projects", "lead_text"]
    out = picked[cols].rename(columns={"label": "grade", "ref_n": "refs", "lead_text": "lead300"})
    out["lead300"] = out["lead300"].str.slice(0, 300)
    return out.reset_index(drop=True)
