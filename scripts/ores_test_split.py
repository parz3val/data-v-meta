"""Score the test partition with Wikimedia's deployed article-quality model (Lift Wing, enwiki-articlequality)
at the same revision ids the dataset holds. Writes results/<mode>/extra/ores_test.json (pageid -> prediction, probs)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wqa import config
from wqa.config import ROOT
from wqa.models import common

mode = sys.argv[1] if len(sys.argv) > 1 else "full"
URL = "https://api.wikimedia.org/service/lw/inference/v1/models/enwiki-articlequality:predict"
cfg = config.load("collection"); ua = config.user_agent(cfg)
df = common.load_table(mode); te = df[df["split"] == "test"][["pageid", "label"]]
rev = pd.read_parquet(ROOT / "data" / mode / "interim" / "full.parquet", columns=["pageid", "revid"]).drop_duplicates("pageid")
te = te.merge(rev, on="pageid", how="left")
out_p = ROOT / "results" / mode / "extra" / "ores_test.json"; out_p.parent.mkdir(parents=True, exist_ok=True)
out = json.loads(out_p.read_text()) if out_p.exists() else {}
s = requests.Session(); s.headers["User-Agent"] = ua
t0 = time.time(); fails = 0
for i, (pid, y, rid) in enumerate(te[["pageid", "label", "revid"]].itertuples(index=False)):
    if str(pid) in out or pd.isna(rid):
        continue
    for attempt in range(5):
        try:
            r = s.post(URL, json={"rev_id": int(rid)}, timeout=(10, 40))
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt); continue
            j = r.json()
            sc = j["enwiki"]["scores"][str(int(rid))]["articlequality"]["score"]
            out[str(pid)] = {"label": y, "revid": int(rid), "prediction": sc["prediction"], "probability": sc["probability"]}
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                fails += 1; out[str(pid)] = {"label": y, "revid": int(rid), "error": str(e)[:120]}
            time.sleep(1 + attempt)
    if i % 100 == 0:
        out_p.write_text(json.dumps(out)); print(f"{i}/{len(te)} done, {fails} failed, {time.time()-t0:.0f}s", flush=True)
    time.sleep(0.02)
out_p.write_text(json.dumps(out)); print("done", len(out), "fails", fails)
