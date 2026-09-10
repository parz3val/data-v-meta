"""Score one live Wikipedia article with every saved model (metadata arm, WikiLite, optional encoder).
Fetches through the same collector code as training, builds the same features, keeps nothing."""
from __future__ import annotations
import os, pickle, re, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # CatBoost/LightGBM and torch each ship an OpenMP runtime
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wqa import config
from wqa.config import ROOT
from wqa.collect import assess, enrich
from wqa.collect.client import ApiClient
from wqa.clean.parse import parse_article
from wqa.clean.run import _clean_record
from wqa.features import build as fbuild
from wqa.models import common

GRADES = ["Stub", "Start", "C", "B", "GA", "FA"]
MODELS = [("rung1_catboost", "CatBoost (metadata arm)"), ("rung1_rf", "Random forest"), ("rung0_logreg", "Logistic regression"),
          ("rung2_mlp", "Multilayer perceptron"), ("metadata_noH", "CatBoost without history"), ("custom_wikilite", "WikiLite (distilled)")]
ENC_TAG = "rung3_distilroberta_base_headtail"


class Scorer:
    def __init__(self, mode: str = "full", seed: int = 42, encoder: bool = True) -> None:
        self.mode, self.seed = mode, seed
        self.cfg = config.load("collection"); self.fcfg = config.load("features")
        self.client = ApiClient(self.cfg)
        self.enc = None; self.dev = None
        ckpt = ROOT / "results" / mode / "models" / f"{ENC_TAG}_s{seed}.pt"
        if encoder and ckpt.exists():
            import multiprocessing as mp
            from app.encoder_worker import serve
            ctx = mp.get_context("spawn"); self._conn, child = ctx.Pipe()
            self._proc = ctx.Process(target=serve, args=(child, str(ckpt)), daemon=True); self._proc.start()
            status, self.dev = self._conn.recv(); self.enc = self._proc
            print(f"scorer: encoder worker {status} on {self.dev}", flush=True)
        self.models: dict[str, Any] = {}
        for key, _ in MODELS + [("custom_ordinal_coral", "")]:
            p = ROOT / "results" / mode / "models" / f"{key}_s{seed}.pkl"
            if p.exists():
                self.models[key] = pickle.load(p.open("rb"))
        import pyarrow.parquet as pq
        head = pq.ParquetFile(ROOT / "data" / mode / "features" / "features.parquet").read_row_group(0).to_pandas()
        self.cols = {fam: common.family_columns(head, fam) for fam in ("metadata", "metadata_noH")}
        print("scorer: metadata models loaded", flush=True)
    # ---- fetch
    def fetch(self, title: str) -> dict[str, Any]:
        title = re.sub(r"^(https?://)?(en\.(m\.)?wikipedia\.org)?/?(wiki/)?", "", title.strip()).split("#")[0].replace("_", " ")
        from urllib.parse import unquote
        title = unquote(title)
        core = enrich.batch_core(self.client, [title])
        if not core:
            raise ValueError(f"no article called '{title}'")
        title, c = next(iter(core.items()))
        if c.get("is_redirect") or c.get("is_disambig"):
            raise ValueError(f"'{title}' is a redirect or disambiguation page")
        graded = assess.batch_assessments(self.client, [title]).get(title, {})
        lists = enrich.lists_for(self.client, title)
        hist = enrich.history_counts(self.client, title, self.cfg["pageviews_days"])
        views = enrich.pageviews_90d(self.client, title, self.cfg["pageviews_days"])
        from datetime import datetime, timezone
        rec = {"pageid": c["pageid"], "title": title, "project_classes": graded.get("pairs", []), "label": graded.get("label") or "Stub",
               "n_projects": graded.get("n_projects", 0), "n_disagree": graded.get("n_disagree", 0), "revid": c["revid"], "wikitext": c["wikitext"],
               "bytes": c["bytes"], "protection": c["protection"], "categories": lists["categories"], "templates": lists["templates"],
               "pageviews_90d_mean": views, "edits": hist["edits"], "editors": hist["editors"], "anon_n": hist["anon_n"], "bot_n": hist["bot_n"],
               "reverted_n": hist["reverted_n"], "edits_90d": hist["edits_90d"], "created": hist["created"] or c["last_edit"],
               "last_edit": c["last_edit"], "fetch_ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        rec["grade_on_record"] = graded.get("label")
        return rec

    # ---- features
    def features(self, rec: dict[str, Any]) -> pd.DataFrame:
        interim = pd.DataFrame([_clean_record(rec)])
        out = pd.DataFrame({"pageid": interim["pageid"]})
        for fam in ("structural", "reference", "history", "contributor", "pageviews"):
            funcs = fbuild._family_funcs(fam)
            for name in self.fcfg[fam]:
                out[name] = funcs[name](interim)
        return out

    # ---- score
    def score(self, title: str) -> dict[str, Any]:
        rec = self.fetch(title)
        X = self.features(rec)
        out: dict[str, Any] = {"title": rec["title"], "pageid": rec["pageid"], "grade_on_record": rec["grade_on_record"], "models": [],
                               "features": {k: (None if pd.isna(v) else float(v)) for k, v in X.iloc[0].items() if k != "pageid"}}
        for key, name in MODELS:
            blob = self.models.get(key)
            if not blob:
                continue
            fam = blob.get("family", "metadata"); fam = fam if fam in self.cols else "metadata"
            xrow = X[self.cols[fam]].fillna(0)
            t0 = time.perf_counter(); p = np.asarray(blob["est"].predict_proba(xrow))[0]; ms = (time.perf_counter() - t0) * 1000
            out["models"].append(self._entry(name, p, ms, "metadata" if fam == "metadata" else "metadata without history"))
        if "rung1_catboost" in self.models:
            out["shap"] = self._shap(X)
        if self.enc is not None:
            text = parse_article(rec["wikitext"], rec["categories"], rec["templates"])["clean_text"]
            out["models"].insert(0, self._encoder(text))
        probs_for_stale = next((m["probs"] for m in out["models"] if m["name"].startswith("Ordinal")), None)
        base = self.models.get("custom_ordinal_coral")
        if base is not None:
            probs_for_stale = np.asarray(base["est"].predict_proba(X[self.cols["metadata"]].fillna(0)))[0].tolist()
        elif "rung1_catboost" in self.models:
            probs_for_stale = next(m["probs"] for m in out["models"] if m["name"].startswith("CatBoost"))
        if rec["grade_on_record"] and probs_for_stale:
            yh = GRADES.index(rec["grade_on_record"])
            out["staleness"] = round(float(sum(p * abs(k - yh) for k, p in enumerate(probs_for_stale))), 3)
        return out

    def _entry(self, name: str, p: np.ndarray, ms: float, arm: str) -> dict[str, Any]:
        exp = float(np.dot(p, np.arange(6)))
        return {"name": name, "arm": arm, "probs": [round(float(v), 4) for v in p], "argmax": GRADES[int(np.argmax(p))],
                "expected_grade": GRADES[int(round(exp))], "expected_index": round(exp, 2), "ms": round(ms, 2)}

    def _encoder(self, text: str) -> dict[str, Any]:
        self._conn.send(text); probs, ms, _ = self._conn.recv()
        return self._entry("DistilRoBERTa encoder (text arm)", np.asarray(probs), ms, "text")

    def _shap(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        try:
            import shap
            est = self.models["rung1_catboost"]["est"]; cols = self.cols["metadata"]
            xrow = X[cols].fillna(0)
            sv = shap.TreeExplainer(est).shap_values(xrow)
            sv = np.asarray(sv); k = int(np.argmax(est.predict_proba(xrow)[0]))
            vals = sv[0, :, k] if sv.ndim == 3 and sv.shape[0] == 1 else sv[k][0]
            order = np.argsort(-np.abs(vals))[:8]
            return [{"feature": cols[i], "value": float(xrow.iloc[0, i]), "shap": round(float(vals[i]), 3)} for i in order]
        except Exception as e:  # noqa: BLE001
            return [{"feature": "unavailable", "value": 0, "shap": 0, "error": str(e)}]
