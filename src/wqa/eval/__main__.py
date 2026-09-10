"""P6 eval: test ONCE on trained models. In: results/<mode>/models + features. Out: eval session.
Usage: python -m wqa.eval --mode small"""
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

from wqa.config import ROOT
from wqa.eval.metrics import all_metrics, bootstrap_ci, macro_f1, qwk
from wqa.models import common

log = structlog.get_logger()


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ece(y: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


def main() -> None:
    args = sys.argv[1:]
    mode = args[args.index("--mode") + 1] if "--mode" in args else "small"
    mdl_dir = ROOT / "results" / mode / "models"
    pkls = sorted(mdl_dir.glob("*_s42.pkl"))  # seed 42 for headline; CI covers sampling noise
    if not pkls:
        raise SystemExit("no trained models — run `make train-all` first")
    df = common.load_table(mode)
    sdir = common.session_dir(mode, "eval", _now_id())
    rows: list[dict[str, Any]] = []
    confusion_specs = []
    for p in pkls:
        blob = pickle.load(p.open("rb"))
        est, family = blob["est"], blob["family"]
        Xte, yte = common.load_xy(mode, family, "test", df)
        t0 = time.monotonic()
        probs = est.predict_proba(Xte)
        infer_s = time.monotonic() - t0
        pred = probs.argmax(axis=1)
        rr = common.expected_grade_round(probs)
        m = all_metrics(yte, pred)
        lo, hi = bootstrap_ci(yte, pred, "macro_f1")
        # temporal holdout subset
        th = df[(df["split"] == "test") & (df.get("temporal_holdout", False))]
        th_f1 = None
        if len(th) > 10:
            cols = common.family_columns(df, family)
            th_f1 = macro_f1(th["label"].to_numpy(int), est.predict_proba(th[cols].fillna(0)).argmax(axis=1))
        cm = np.zeros((6, 6), int)
        for a, b in zip(yte, pred):
            cm[a, b] += 1
        confusion_specs.append({"id": f"confusion_{p.stem}", "title": f"Confusion {p.stem}",
                                "kind": "table", "x": list("012345"),
                                "series": [{"name": str(i), "y": cm[i].tolist()} for i in range(6)],
                                "xlabel": "pred", "ylabel": "true", "notes": "counts; test split"})
        rows.append({"config": p.stem.replace("_s42", ""), "family": family,
                     "macro_f1": m["macro_f1"], "ci_lo": lo, "ci_hi": hi, "qwk_argmax": m["qwk"],
                     "qwk_regress_round": qwk(yte, rr), "within_one": m["within_one"],
                     "acc": m["acc"], "ordinal_mae": m["ordinal_mae"], "ece": ece(yte, probs),
                     "temporal_holdout_f1": th_f1,
                     "infer_s_per_1k": round(infer_s / max(len(yte), 1) * 1000, 4)})
        log.info("eval", config=rows[-1]["config"], f1=round(m["macro_f1"], 4))
    # merge rows produced out-of-band (rung3 encoder / text_plus_H write their own metrics
    # because a transformer can't be pickled into the *_s42.pkl glob)
    enc_rows_p = ROOT / "results" / mode / "encoder" / "eval_rows.json"
    if enc_rows_p.exists():
        have = {r["config"] for r in rows}
        for r in json.loads(enc_rows_p.read_text()):
            if r["config"] not in have:
                rows.append(r)
                log.info("eval.merged_external", config=r["config"])
    charts_dir = ROOT / "results" / mode / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    headline = {"id": "headline", "title": "Macro-F1 and QWK vs inference cost", "kind": "scatter",
                "x": [r["infer_s_per_1k"] for r in rows],
                "series": [{"name": "macro_f1", "y": [round(r["macro_f1"], 4) for r in rows],
                            "ci": [[round(r["ci_lo"], 4), round(r["ci_hi"], 4)] for r in rows]},
                           {"name": "qwk", "y": [round(r["qwk_argmax"], 4) for r in rows]}],
                "xlabel": "inference s / 1k articles", "ylabel": "score", "xscale": "log",
                "notes": "labels: " + ", ".join(r["config"] for r in rows)}
    (charts_dir / "headline.json").write_text(json.dumps(headline, indent=1))
    for cs in confusion_specs:
        (charts_dir / f"{cs['id']}.json").write_text(json.dumps(cs, indent=1))
    (sdir / "results.json").write_text(json.dumps(rows, indent=1))
    (sdir / "session.json").write_text(json.dumps({"created": _now_id(),
        "summary": f"{len(rows)} models on test; best macro-F1 "
                   f"{max(r['macro_f1'] for r in rows):.3f}"}, indent=1))
    tables = ROOT / "results" / mode / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    try:
        from wqa.charts import to_png, to_tex
        to_tex(headline, tables / "headline.tex")
        for cs in confusion_specs:
            to_tex(cs, tables / f"{cs['id']}.tex")
        to_png(headline, ROOT / "results" / mode / "figures" / "fig_eval_headline.png")
        th_rows = [r for r in rows if r["temporal_holdout_f1"] is not None]
        to_tex({"id": "temporal_holdout", "title": "Temporal holdout vs random test", "kind": "table",
                "x": [r["config"] for r in th_rows],
                "series": [{"name": "test_f1", "y": [round(r["macro_f1"], 4) for r in th_rows]},
                           {"name": "temporal_f1", "y": [round(r["temporal_holdout_f1"], 4) for r in th_rows]}],
                "xlabel": "config", "ylabel": "", "notes": "latest-10% grade dates"},
               tables / "temporal_holdout.tex")
        to_tex({"id": "cost_table", "title": "Cost per 1k articles", "kind": "table",
                "x": [r["config"] for r in rows],
                "series": [{"name": "infer_s_per_1k", "y": [r["infer_s_per_1k"] for r in rows]},
                           {"name": "ece", "y": [round(r["ece"], 4) for r in rows]}],
                "xlabel": "config", "ylabel": "", "notes": "CPU M4; SMALL"},
               tables / "cost_table.tex")
    except Exception as e:
        log.warning("charts.unavailable", err=str(e))
    print(f"eval: {len(rows)} models; test touched once")


if __name__ == "__main__":
    main()
