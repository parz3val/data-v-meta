"""Refit the non-selected pre-registered learners with the settings their own validation search chose
(train_all used to pass tuned settings only to the selected kind). Overwrites results/<mode>/models/<config>_s<seed>.pkl."""
from __future__ import annotations
import importlib, json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wqa.config import ROOT
from wqa.models import common
M = importlib.import_module("wqa.models.__main__")
mode = sys.argv[1] if len(sys.argv) > 1 else "full"
sel = json.load(open(ROOT / "results" / mode / "select/current/models.json"))
params_for = {c["kind"]: c.get("params", {}) for c in sel["candidates"]}
CONFIGS = {"rung0_logreg": "logreg", "rung1_rf": "random_forest", "rung2_mlp": "mlp", "rung1_xgboost": "xgboost", "rung1_lightgbm": "lightgbm", "rung1_histgb": "histgb"}
df = common.load_table(mode)
Xtr, ytr = common.load_xy(mode, "metadata", "train", df); Xva, yva = common.load_xy(mode, "metadata", "val", df)
mdl = ROOT / "results" / mode / "models"; live = ROOT / "results" / mode / "live"
for name, kind in CONFIGS.items():
    params = params_for.get(kind, {})
    for seed in (7, 42, 1337):
        data = {"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva}
        res = M._fit_score(kind, params, seed, data)
        with (mdl / f"{name}_s{seed}.pkl").open("wb") as fb:
            pickle.dump({"est": data["est"], "family": "metadata", "kind": kind, "params": params, "seed": seed}, fb)
        (live / f"{name}_s{seed}.jsonl").write_text(json.dumps({"epoch": 1, "ts": M._now_id(), **{k: res[k] for k in ("val_macro_f1", "val_acc", "val_qwk_argmax", "wall_s")}}) + "\n")
        print(name, seed, params, round(res["val_macro_f1"], 4), round(res["wall_s"], 1), flush=True)
