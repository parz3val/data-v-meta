"""Trained-model space visualization data (owner request 2026-08-25).
Two 2D projections (val+test articles), rendered interactively by the panel Models tab:
- metadata: scaled feature space -> PCA -> t-SNE; pred = ordinal_coral (best metadata arm)
- text: encoder class-logit space (results/<mode>/encoder/logits_*.parquet) -> t-SNE; pred = argmax
Out: results/<mode>/viz/embedding.json (contract shared with panel/js/tabs.mjs renderModels).
Usage: python scripts/embedding_space.py --mode small
"""
import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wqa.models import common  # noqa: E402


def tsne2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    per = min(30, max(5, len(X) // 8))
    return TSNE(n_components=2, perplexity=per, random_state=seed, init="pca",
                max_iter=1000).fit_transform(X)


def pts(xy, grades, preds, pageids, splits):
    return [{"x": round(float(x), 2), "y": round(float(y), 2), "grade": int(g),
             "pred": int(p), "pageid": int(pid), "split": s}
            for (x, y), g, p, pid, s in zip(xy, grades, preds, pageids, splits)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="small")
    args = ap.parse_args()

    df = common.load_table(args.mode)
    part = df[df["split"].isin(["val", "test"])].copy()
    cols = common.family_columns(df, "metadata")
    X = part[cols].fillna(0).to_numpy(float)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    Xs = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=min(20, Xs.shape[1]), random_state=42).fit_transform(Xs)
    xy_meta = tsne2d(Xp)

    mdl = ROOT / "results" / args.mode / "models" / "custom_ordinal_coral_s42.pkl"
    if not mdl.exists():
        mdl = sorted((ROOT / "results" / args.mode / "models").glob("rung1_*_s42.pkl"))[0]
    blob = pickle.load(mdl.open("rb"))
    pred_meta = blob["est"].predict_proba(part[common.family_columns(df, blob["family"])]
                                          .fillna(0)).argmax(1)

    sets = {"metadata": {
        "method": f"scaled->PCA20->tSNE ({mdl.stem} predictions)",
        "points": pts(xy_meta, part["label"].to_numpy(int), pred_meta,
                      part["pageid"].to_numpy(int), part["split"].tolist())}}

    lg_files = sorted((ROOT / "results" / args.mode / "encoder").glob("logits_*headtail_s42.parquet"))
    if lg_files:
        import pandas as pd
        lg = pd.read_parquet(lg_files[-1])
        lg = lg[lg["split"].isin(["val", "test"])].merge(
            part[["pageid", "label"]], on="pageid", how="inner")
        L = lg[[f"logit_{i}" for i in range(6)]].to_numpy(float)
        xy_text = tsne2d(L)
        sets["text"] = {
            "method": f"encoder class-logit space -> tSNE ({lg_files[-1].stem})",
            "points": pts(xy_text, lg["label"].to_numpy(int), L.argmax(1),
                          lg["pageid"].to_numpy(int), lg["split"].tolist())}

    out = ROOT / "results" / args.mode / "viz"
    out.mkdir(parents=True, exist_ok=True)
    (out / "embedding.json").write_text(json.dumps(
        {"generated": datetime.now(timezone.utc).isoformat(), "sets": sets}))
    print(f"embedding.json: {', '.join(f'{k}={len(v['points'])}pts' for k, v in sets.items())}")


if __name__ == "__main__":
    main()
