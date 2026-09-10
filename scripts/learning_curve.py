"""Learning curve: metadata arm vs training-set size (owner data-level experiment 2026-08-25).
Subsamples the train split per grade at n_per_grade levels, retrains the selected rung1
model (params from status/models.json), evaluates on the FIXED val+test splits.
Usage: python scripts/learning_curve.py --mode small [--levels 25,50,75,100] [--seeds 7,42,1337]
Out: results/<mode>/curves/learning_curve.json + charts/learning_curve.json (+tex/png).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wqa.config import ROOT as WROOT  # noqa: E402
from wqa.eval.metrics import macro_f1, qwk  # noqa: E402
from wqa.models import common  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="small")
    ap.add_argument("--levels", default="25,50,75,100,125")
    ap.add_argument("--seeds", default="7,42,1337")
    ap.add_argument("--kind", default=None, help="override model kind (default: selected arm)")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    sel = json.loads((WROOT / "status" / "models.json").read_text())
    kind = args.kind or sel["metadata_arm"]
    params = next((c["params"] for c in sel["candidates"] if c["kind"] == kind), {})

    df = common.load_table(args.mode)
    Xva, yva = common.load_xy(args.mode, "metadata", "val", df)
    Xte, yte = common.load_xy(args.mode, "metadata", "test", df)
    tr = df[df["split"] == "train"]
    cols = common.family_columns(df, "metadata")

    rows = []
    for n in levels:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            picks = []
            for g, grp in tr.groupby("label"):
                take = min(n, len(grp))
                picks.append(grp.iloc[rng.permutation(len(grp))[:take]])
            import pandas as pd
            trn = pd.concat(picks)
            Xtr, ytr = trn[cols].fillna(0), trn["label"].to_numpy(int)
            t0 = time.monotonic()
            est = common.make_estimator(kind, params, seed)
            est.fit(Xtr, ytr)
            wall = time.monotonic() - t0
            pv = est.predict_proba(Xva).argmax(1)
            pt = est.predict_proba(Xte).argmax(1)
            rows.append({"n_per_grade": n, "n_train": int(len(ytr)), "seed": seed,
                         "kind": kind, "val_macro_f1": macro_f1(yva, pv),
                         "test_macro_f1": macro_f1(yte, pt), "test_qwk": qwk(yte, pt),
                         "wall_s": round(wall, 2)})
            print(f"n={n} seed={seed} test_f1={rows[-1]['test_macro_f1']:.4f}")

    out = WROOT / "results" / args.mode / "curves"
    out.mkdir(parents=True, exist_ok=True)
    (out / "learning_curve.json").write_text(json.dumps(
        {"mode": args.mode, "kind": kind, "levels": levels, "seeds": seeds, "rows": rows}, indent=1))

    xs = levels
    mean = lambda n, k: float(np.mean([r[k] for r in rows if r["n_per_grade"] == n]))  # noqa: E731
    sd = lambda n, k: float(np.std([r[k] for r in rows if r["n_per_grade"] == n]))  # noqa: E731
    spec = {"id": "learning_curve", "title": f"Learning curve — {kind} (metadata, {args.mode})",
            "kind": "line", "x": xs,
            "series": [{"name": "test_macro_f1",
                        "y": [round(mean(n, "test_macro_f1"), 4) for n in xs],
                        "ci": [[round(mean(n, "test_macro_f1") - sd(n, "test_macro_f1"), 4),
                                round(mean(n, "test_macro_f1") + sd(n, "test_macro_f1"), 4)]
                               for n in xs]},
                       {"name": "val_macro_f1",
                        "y": [round(mean(n, "val_macro_f1"), 4) for n in xs]}],
            "xlabel": "train articles per grade", "ylabel": "macro-F1",
            "notes": f"seeds {seeds}; fixed val/test; ±1 sd band"}
    ch = WROOT / "results" / args.mode / "charts"
    ch.mkdir(parents=True, exist_ok=True)
    (ch / "learning_curve.json").write_text(json.dumps(spec, indent=1))
    try:
        from wqa.charts import to_png, to_tex
        to_png(spec, WROOT / "results" / args.mode / "figures" / "fig_learning_curve.png")
        to_tex(spec, WROOT / "results" / args.mode / "tables" / "learning_curve.tex")
    except Exception as e:
        print(f"charts skipped: {e}")


if __name__ == "__main__":
    main()
