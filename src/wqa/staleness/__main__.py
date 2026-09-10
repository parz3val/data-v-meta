"""P7 staleness: D = expected ordinal distance; Spearman vs age proxies AND, when
results/<mode>/staleness/grade_dates.json exists (scripts/grade_dates.py, talk-page bisect),
vs RECOVERED grade age — the claim-2 validation proper (upgrade 2026-08-25).
In: best model + features. Out: staleness session + review CSV. Usage: python -m wqa.staleness --mode small"""
import csv
import json
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import structlog

from wqa.config import ROOT
from wqa.models import common

log = structlog.get_logger()


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    from scipy.stats import spearmanr
    args = sys.argv[1:]
    mode = args[args.index("--mode") + 1] if "--mode" in args else "small"
    df = common.load_table(mode)
    mdl_dir = ROOT / "results" / mode / "models"
    pkls = sorted(mdl_dir.glob("rung1_*_s42.pkl"))
    if not pkls:
        raise SystemExit("no trained rung1 model")
    sel = json.loads((ROOT / "status" / "models.json").read_text())["metadata_arm"]
    pick = next((p for p in pkls if sel in p.stem), pkls[0])
    # prefer the ordinal model when trained — best macro-F1/QWK metadata arm 2026-08-25
    ordinal = mdl_dir / "custom_ordinal_coral_s42.pkl"
    if ordinal.exists():
        pick = ordinal
    blob = pickle.load(pick.open("rb"))
    est, family = blob["est"], blob["family"]
    part = df[df["split"].isin(["val", "test"])].copy()
    cols = common.family_columns(df, family)
    probs = est.predict_proba(part[cols].fillna(0))
    y = part["label"].to_numpy(int)
    D = np.abs(np.arange(probs.shape[1])[None, :] - y[:, None])
    d_score = (probs * D).sum(axis=1)
    part["D"] = d_score
    # PRIMARY validation when available: recovered grade age (talk-page bisect)
    grade_age = None
    gd_p = ROOT / "results" / mode / "staleness" / "grade_dates.json"
    if gd_p.exists():
        gd = json.loads(gd_p.read_text())
        now = datetime.now(timezone.utc)
        ages = []
        for pid in part["pageid"].astype(int):
            rec = gd.get(str(pid))
            if rec and rec.get("grade_ts"):
                ts = datetime.fromisoformat(rec["grade_ts"].replace("Z", "+00:00"))
                ages.append((now - ts).days)
            else:
                ages.append(np.nan)
        part["grade_age_days"] = ages
        ok = part["grade_age_days"].notna()
        if ok.sum() > 30:
            r = spearmanr(d_score[ok.to_numpy()], part.loc[ok, "grade_age_days"])
            grade_age = {"n": int(ok.sum()), "spearman": round(float(r.statistic), 4),
                         "p_one_sided": round(float(r.pvalue) / 2, 6)}
            log.info("staleness.grade_age", **grade_age)

    proxies = {}
    for proxy in ("age_days", "days_since_edit", "edits_90d", "edits"):
        if proxy in part.columns:
            r = spearmanr(d_score, part[proxy].fillna(0))
            proxies[proxy] = {"spearman": round(float(r.statistic), 4),
                              "p_one_sided": round(float(r.pvalue) / 2, 5)}
    # partial on article age: residualise D and days_since_edit on age_days ranks
    partial = None
    if "age_days" in part.columns and "days_since_edit" in part.columns:
        def rank(v: np.ndarray) -> np.ndarray:
            return np.argsort(np.argsort(v)).astype(float)
        a, b, c = rank(d_score), rank(part["days_since_edit"].fillna(0).to_numpy()), \
            rank(part["age_days"].fillna(0).to_numpy())
        ra = a - np.polyval(np.polyfit(c, a, 1), c)
        rb = b - np.polyval(np.polyfit(c, b, 1), c)
        partial = round(float(spearmanr(ra, rb).statistic), 4)
    order = np.argsort(-d_score)
    n = len(order)
    idx = list(order[:20]) + list(order[n // 2 - 10:n // 2 + 10]) + list(order[-20:])
    sdir = common.session_dir(mode, "staleness", _now_id())
    with (sdir / "staleness_review.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pageid", "title", "label", "D", "bucket", "human_grade"])
        for i, ix in enumerate(idx):
            r = part.iloc[ix]
            w.writerow([r.get("pageid"), r.get("title", ""), int(r["label"]),
                        round(float(r["D"]), 3),
                        "high" if i < 20 else ("mid" if i < 40 else "low"), ""])
    out = {"created": _now_id(), "model": pick.stem,
           "grade_age_validation": grade_age,
           "coverage_note": ("grade-date recovery RUN via scripts/grade_dates.py (talk-page bisect); "
                             "grade_age_validation is the claim-2 test." if grade_age else
                             "grade-date recovery NOT RUN (talk-page parsing deferred; SMALL API budget)."
                             " Proxies used: days_since_edit, age; identifiability caveat THEORY §4."),
           "spearman_vs_proxies": proxies, "partial_days_since_edit_given_age": partial,
           "positioning": "vs TeBlunthuis 2021 / arXiv 2111.01496: they model quality change; "
                          "we flag stale grades via model-human disagreement"}
    (sdir / "staleness.json").write_text(json.dumps(out, indent=1))
    (sdir / "session.json").write_text(json.dumps({"created": out["created"],
        "summary": f"proxies={ {k: v['spearman'] for k, v in proxies.items()} } partial={partial}"},
        indent=1))
    charts = ROOT / "results" / mode / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    (charts / "staleness_corr.json").write_text(json.dumps({"id": "staleness_corr",
        "title": "Spearman(D, staleness proxies)", "kind": "bar", "x": list(proxies),
        "series": [{"name": "spearman", "y": [proxies[k]["spearman"] for k in proxies]}],
        "xlabel": "proxy", "ylabel": "rho",
        "notes": f"one-sided; partial(days_since_edit|age)={partial}"}, indent=1))
    tables = ROOT / "results" / mode / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    try:
        from wqa.charts import to_tex
        to_tex(json.loads((charts / "staleness_corr.json").read_text()),
               tables / "staleness_corr.tex")
    except Exception as e:
        log.warning("charts.unavailable", err=str(e))
    print(f"staleness: proxies={proxies} partial={partial}; review CSV 60 rows")


if __name__ == "__main__":
    main()
