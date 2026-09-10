"""text_plus_H ablation (OPS §8 / THEORY): encoder logits + g(H) history features -> rung1 learner.
Needs results/<mode>/encoder/logits_<tag>_s<seed>.parquet from wqa.models.encoder.
Usage: python -m wqa.models.text_plus_h --mode small --tag rung3_distilroberta_base_headtail --seed 42
Writes its eval row into results/<mode>/encoder/eval_rows.json (merged by wqa.eval).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from wqa.config import ROOT
from wqa.eval.metrics import all_metrics, bootstrap_ci, macro_f1, qwk
from wqa.models import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="small")
    ap.add_argument("--tag", default="rung3_distilroberta_base_headtail")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fit-split", default="train", choices=["train", "val"],
                    help="val: fit the fusion learner on out-of-sample encoder logits (validation partition)")
    args = ap.parse_args()

    lg_p = ROOT / "results" / args.mode / "encoder" / f"logits_{args.tag}_s{args.seed}.parquet"
    if not lg_p.exists():
        raise SystemExit(f"missing {lg_p} — run wqa.models.encoder first")
    lg = pd.read_parquet(lg_p)
    df = common.load_table(args.mode)
    gh_cols = common.family_columns(df, "gH")
    logit_cols = [c for c in lg.columns if c.startswith("logit_")]
    m = df.merge(lg.drop(columns=["split"]), on="pageid", how="inner")
    cols = gh_cols + logit_cols

    sel = json.loads((ROOT / "status" / "models.json").read_text())
    kind = sel["metadata_arm"]
    params = next((c["params"] for c in sel["candidates"] if c["kind"] == kind), {})

    def xy(split):
        part = m[m["split"] == split]
        return part[cols].fillna(0), part["label"].to_numpy(int), part

    Xtr, ytr, _ = xy(args.fit_split)
    Xte, yte, te = xy("test")
    est = common.make_estimator(kind, params, args.seed)
    est.fit(Xtr, ytr)
    t0 = time.monotonic()
    probs = est.predict_proba(Xte)
    infer_s = time.monotonic() - t0
    pred = probs.argmax(1)
    rr = common.expected_grade_round(probs)
    pd.DataFrame({"pageid": te["pageid"].to_numpy(), "label": yte, "pred": pred,
                  **{f"prob_{i}": probs[:, i] for i in range(6)}}).to_parquet(
        ROOT / "results" / args.mode / "encoder" / f"preds_text_plus_H_s{args.seed}.parquet", index=False)
    met = all_metrics(yte, pred)
    lo, hi = bootstrap_ci(yte, pred, "macro_f1")
    th = te[te.get("temporal_holdout", pd.Series(False, index=te.index)).astype(bool)]
    th_f1 = None
    if len(th) > 10:
        th_f1 = macro_f1(th["label"].to_numpy(int),
                         est.predict_proba(th[cols].fillna(0)).argmax(1))
    conf = probs.max(1)
    ece = 0.0
    for b in range(10):
        msk = (conf > b / 10) & (conf <= (b + 1) / 10)
        if msk.sum():
            ece += msk.mean() * abs((pred[msk] == yte[msk]).mean() - conf[msk].mean())
    row = {"config": "text_plus_H", "family": "text_plus_H", "macro_f1": met["macro_f1"],
           "ci_lo": lo, "ci_hi": hi, "qwk_argmax": met["qwk"],
           "qwk_regress_round": qwk(yte, rr), "within_one": met["within_one"],
           "acc": met["acc"], "ordinal_mae": met["ordinal_mae"], "ece": float(ece),
           "temporal_holdout_f1": th_f1,
           "infer_s_per_1k": round(infer_s / max(len(yte), 1) * 1000, 4),
           "base": kind, "text_tag": args.tag, "fit_split": args.fit_split,
           "note": "gH+logits; encoder infer cost NOT included in infer_s_per_1k"}
    rows_p = ROOT / "results" / args.mode / "encoder" / "eval_rows.json"
    rows = json.loads(rows_p.read_text()) if rows_p.exists() else []
    rows = [r for r in rows if r["config"] != "text_plus_H"] + [row]
    rows_p.write_text(json.dumps(rows, indent=1))
    print(f"text_plus_H: macro_f1={met['macro_f1']:.4f} (base {kind} on {len(cols)} cols)")


if __name__ == "__main__":
    main()
