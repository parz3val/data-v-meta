"""P8 XAI: SHAP global/per-grade + faithfulness (deletion/insertion) + stability + plausibility map.
In: best trained booster + features. Out: xai session. Usage: python -m wqa.xai --mode small"""
import json
import pickle
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

from wqa.config import ROOT
from wqa.eval.metrics import macro_f1
from wqa.models import common

log = structlog.get_logger()
FA_MAP = {  # top-feature -> FA criteria / B-checklist hook (owner writes prose)
    "ref_n": "FA 1c well-researched: citations", "refs_per_1k_words": "FA 1c citation density",
    "bytes": "FA 1b comprehensive: length proxy", "words": "FA 1b comprehensive",
    "sections_l2": "FA 2b structure", "images": "FA 3 media", "wikilinks": "B1 linking",
    "cleanup_tags": "B4 grammar/cleanup (inverse)", "edits": "history: attention",
    "editors": "history: many eyes", "age_days": "history: maturity",
    "views_90d_log": "demand-side salience (not a criterion)",
}


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def deletion_curve(est: Any, X: Any, y: np.ndarray, order: list[str], med: Any,
                   insert: bool = False) -> list[float]:
    Xw = X.copy()
    if insert:
        base = X.copy()
        for c in X.columns:
            base[c] = med[c]
        Xw = base
    scores = [macro_f1(y, est.predict_proba(Xw).argmax(axis=1))]
    for c in order:
        Xw = Xw.copy()
        Xw[c] = X[c] if insert else med[c]
        scores.append(macro_f1(y, est.predict_proba(Xw).argmax(axis=1)))
    return [round(float(s), 4) for s in scores]


def main() -> None:
    import shap
    args = sys.argv[1:]
    mode = args[args.index("--mode") + 1] if "--mode" in args else "small"
    sel_p = ROOT / "results" / mode / "select" / "current" / "models.json"
    if not sel_p.exists():
        sel_p = ROOT / "status" / "models.json"
    sel = json.loads(sel_p.read_text())
    best = sel["metadata_arm"] if isinstance(sel, dict) else "catboost"
    df = common.load_table(mode)
    pkls = sorted((ROOT / "results" / mode / "models").glob(f"rung1_{best}_s*.pkl")) or \
        sorted((ROOT / "results" / mode / "models").glob("rung1_*_s*.pkl"))
    if not pkls:
        raise SystemExit("no rung1 pickles — run train-all")
    Xva, yva = common.load_xy(mode, "metadata", "val", df)
    Xtr, _ = common.load_xy(mode, "metadata", "train", df)
    med = Xtr.median()
    imps = []
    for p in pkls:
        est = pickle.load(p.open("rb"))["est"]
        try:
            ex = shap.TreeExplainer(est)
            sv = ex.shap_values(Xva)
            g = np.mean([np.abs(np.asarray(s)).mean(axis=0) for s in sv], axis=0) \
                if isinstance(sv, list) else np.abs(sv).mean(axis=(0, 2) if np.ndim(sv) == 3 else 0)
        except Exception:
            ex = shap.explainers.Permutation(est.predict_proba, Xtr.sample(50, random_state=0))
            g = np.abs(ex(Xva.sample(min(100, len(Xva)), random_state=0)).values).mean(axis=(0, 2))
        imps.append(np.asarray(g, dtype=float))
    global_imp = np.mean(imps, axis=0)
    order = [c for _, c in sorted(zip(global_imp, Xva.columns), reverse=True)]
    # stability: Spearman between per-seed importance vectors
    from scipy.stats import spearmanr
    stab = [float(spearmanr(imps[i], imps[j]).statistic)
            for i in range(len(imps)) for j in range(i + 1, len(imps))] or [1.0]
    est0 = pickle.load(pkls[0].open("rb"))["est"]
    dele = deletion_curve(est0, Xva, yva, order, med)
    ins = deletion_curve(est0, Xva, yva, order, med, insert=True)
    auc_del = float(np.trapezoid(dele) / len(dele))
    auc_ins = float(np.trapezoid(ins) / len(ins))
    per_grade = {}
    sv0 = None
    try:
        sv0 = shap.TreeExplainer(est0).shap_values(Xva)
    except Exception:
        pass
    if isinstance(sv0, list):
        for k in range(len(sv0)):
            gk = np.abs(sv0[k]).mean(axis=0)
            per_grade[str(k)] = [order.index(c) for c in Xva.columns][:0] or \
                [c for _, c in sorted(zip(gk, Xva.columns), reverse=True)][:5]
    sdir = common.session_dir(mode, "xai", _now_id())
    top15 = order[:15]
    out = {"created": _now_id(), "model": best, "global_top15": top15,
           "global_importance": {c: round(float(v), 5) for c, v in zip(Xva.columns, global_imp)},
           "per_grade_top5": per_grade,
           "faithfulness": {"deletion": dele, "insertion": ins,
                            "auc_deletion": round(auc_del, 4), "auc_insertion": round(auc_ins, 4)},
           "stability_spearman": {"pairs": [round(s, 4) for s in stab],
                                  "mean": round(float(np.mean(stab)), 4),
                                  "note": "across seed pickles; n-dependent (THEORY §5)"},
           "plausibility": [{"feature": f, "criterion": FA_MAP.get(f, "%% OWNER: map")}
                            for f in top15]}
    (sdir / "xai.json").write_text(json.dumps(out, indent=1))
    (sdir / "session.json").write_text(json.dumps({"created": out["created"],
        "summary": f"top: {', '.join(top15[:3])}; stab {out['stability_spearman']['mean']}"}, indent=1))
    charts = ROOT / "results" / mode / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    (charts / "shap_global.json").write_text(json.dumps({"id": "shap_global",
        "title": "Global mean |SHAP|", "kind": "bar", "x": top15,
        "series": [{"name": "mean_abs_shap", "y": [out["global_importance"][c] for c in top15]}],
        "xlabel": "feature", "ylabel": "mean |SHAP|", "notes": "val split"}, indent=1))
    (charts / "faithfulness.json").write_text(json.dumps({"id": "faithfulness",
        "title": "Deletion/insertion curves", "kind": "line", "x": list(range(len(dele))),
        "series": [{"name": "deletion", "y": dele}, {"name": "insertion", "y": ins}],
        "xlabel": "# features ablated to train median", "ylabel": "macro-F1",
        "notes": f"AUC del {auc_del:.3f} ins {auc_ins:.3f}"}, indent=1))
    tables = ROOT / "results" / mode / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    try:
        from wqa.charts import to_tex
        to_tex(json.loads((charts / "shap_global.json").read_text()), tables / "shap_global.tex")
        to_tex({"id": "stability", "title": "Importance rank stability", "kind": "table",
                "x": [f"pair{i}" for i in range(len(stab))],
                "series": [{"name": "spearman", "y": [round(s, 4) for s in stab]}],
                "xlabel": "", "ylabel": "", "notes": "seeds"}, tables / "stability.tex")
        to_tex({"id": "faithfulness", "title": "Faithfulness AUCs", "kind": "table",
                "x": ["deletion", "insertion"],
                "series": [{"name": "auc", "y": [round(auc_del, 4), round(auc_ins, 4)]}],
                "xlabel": "", "ylabel": "", "notes": "ERASER-style"}, tables / "faithfulness.tex")
    except Exception as e:
        log.warning("charts.unavailable", err=str(e))
    print(f"xai: top3={top15[:3]} stability={out['stability_spearman']['mean']}")


if __name__ == "__main__":
    main()
