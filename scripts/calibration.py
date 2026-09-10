"""Reliability (calibration) table+chart for the metadata arm, ordinal_coral and wikilite on the test split.
Out: results/<mode>/charts/calibration.json, tables/calibration.tex, figures/calibration.png.
Usage: python scripts/calibration.py --mode medium
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wqa.charts import to_png, to_tex  # noqa: E402
from wqa.models import common  # noqa: E402

BINS = np.linspace(0, 1, 11)


def reliability(p, y):
    conf, pred = p.max(1), p.argmax(1)
    ys = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (conf > lo) & (conf <= hi)
        ys.append(round(float((pred[m] == y[m]).mean()), 3) if m.sum() >= 3 else None)
    return ys


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="small"); a = ap.parse_args()
    df = common.load_table(a.mode); te = df[df["split"] == "test"]
    sel = json.loads((ROOT / "results" / a.mode / "select" / "current" / "models.json").read_text())
    arm = sel.get("metadata_arm", "catboost")
    mdir = ROOT / "results" / a.mode / "models"
    series = []
    for label, stem in [(f"rung1_{arm}", f"rung1_{arm}_s42"), ("ordinal_coral", "custom_ordinal_coral_s42"), ("wikilite", "custom_wikilite_s42")]:
        p = mdir / f"{stem}.pkl"
        if not p.exists():
            continue
        b = pickle.load(p.open("rb"))
        proba = b["est"].predict_proba(te[common.family_columns(df, b["family"])].fillna(0))
        series.append({"name": label, "y": reliability(np.asarray(proba), te["label"].to_numpy(int))})
    x = [round(float(v), 2) for v in (BINS[:-1] + BINS[1:]) / 2]
    series.append({"name": "perfect", "y": x})
    spec = {"id": "calibration", "title": "Reliability — test split (bin accuracy vs confidence)", "kind": "line",
            "x": x, "series": series, "xlabel": "confidence (max prob)", "ylabel": "accuracy in bin",
            "notes": f"mode={a.mode}; bins with <3 samples omitted; n_test={len(te)}"}
    out = ROOT / "results" / a.mode
    (out / "charts").mkdir(parents=True, exist_ok=True)
    (out / "charts" / "calibration.json").write_text(json.dumps(spec, indent=1))
    to_png(spec, out / "figures" / "calibration.png")
    to_tex(spec, out / "tables" / "calibration.tex")
    print(f"calibration: {[s['name'] for s in series]} n_test={len(te)}")


if __name__ == "__main__":
    main()
