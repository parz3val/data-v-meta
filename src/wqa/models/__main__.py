"""P4 select + P5 train (SMALL-first). In: features+splits. Out: models.json, sessions, pickles, MLflow.
Usage: python -m wqa.models --mode small --select | --train <config> | --train-all"""
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from typing import Any

import structlog
import yaml

from wqa.config import ROOT, load
from wqa.eval.metrics import all_metrics, macro_f1, qwk
from wqa.models import common

log = structlog.get_logger()
RUNGS: dict[str, list[str]] = {
    "rung0": ["majority", "logreg"],
    "rung1": ["xgboost", "lightgbm", "catboost", "histgb", "random_forest"],
    "rung2": ["mlp"],
}


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _fit_score(kind: str, params: dict[str, Any], seed: int, data: dict[str, Any]) -> dict[str, Any]:
    t0 = time.monotonic()
    if kind == "majority":
        import numpy as np
        maj = int(np.bincount(data["ytr"]).argmax())
        pv = np.full(len(data["yva"]), maj)
        probs = np.zeros((len(data["yva"]), 6))
        probs[:, maj] = 1.0
    else:
        est = common.make_estimator(kind, params, seed)
        est.fit(data["Xtr"], data["ytr"])
        probs = est.predict_proba(data["Xva"])
        pv = probs.argmax(axis=1)
        data["est"] = est
    rr = common.expected_grade_round(probs)
    m = all_metrics(data["yva"], pv)
    return {"kind": kind, "params": params, "seed": seed, "wall_s": round(time.monotonic() - t0, 2),
            "val_macro_f1": m["macro_f1"], "val_acc": m["acc"], "val_qwk_argmax": m["qwk"],
            "val_qwk_regress_round": qwk(data["yva"], rr),
            "val_macro_f1_regress_round": macro_f1(data["yva"], rr)}


def select(mode: str) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = load("budgets")["optuna"]["small_trials" if mode == "small" else "rung1_trials"]
    df = common.load_table(mode)
    data = {}
    data["Xtr"], data["ytr"] = common.load_xy(mode, "metadata", "train", df)
    data["Xva"], data["yva"] = common.load_xy(mode, "metadata", "val", df)
    candidates: list[dict[str, Any]] = []
    for rung, kinds in RUNGS.items():
        for kind in kinds:
            if kind == "majority":
                candidates.append({"rung": rung, **_fit_score(kind, {}, 42, data)})
                continue
            if kind == "logreg":
                best_params: dict[str, Any] = {}
            else:
                study = optuna.create_study(direction="maximize",
                                            sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(lambda t, k=kind: _fit_score(k, common.search_space(k, t), 42, data)
                               ["val_macro_f1"], n_trials=n_trials, show_progress_bar=False)
                best_params = study.best_params
                if kind == "mlp":
                    best_params = common.search_space(kind, optuna.trial.FixedTrial(study.best_params))
            candidates.append({"rung": rung, **_fit_score(kind, best_params, 42, data)})
            log.info("select.candidate", kind=kind, f1=round(candidates[-1]["val_macro_f1"], 4))
    # pre-registered rule (docs/model_selection.md): best val macro-F1 per rung; tie<=0.005 -> faster
    sel = {}
    for rung in RUNGS:
        rc = [c for c in candidates if c["rung"] == rung]
        best = max(rc, key=lambda c: (round(c["val_macro_f1"], 3), -c["wall_s"]))
        sel[rung] = best["kind"]
    r1 = max((c["val_macro_f1"] for c in candidates if c["rung"] == "rung1"), default=0)
    r2 = max((c["val_macro_f1"] for c in candidates if c["rung"] == "rung2"), default=0)
    metadata_arm = sel["rung2"] if r2 >= r1 + 0.01 else sel["rung1"]
    out = {"mode": mode, "created": _now_id(), "rule": "docs/model_selection.md (pre-registered)",
           "candidates": candidates, "selected_per_rung": sel, "metadata_arm": metadata_arm,
           "rung3": {"status": "NOT RUN (no GPU; CPU smoke deferred)", "planned": ["deberta-v3-base"]},
           "text_plus_H": {"status": "NOT RUN (needs text arm)"}}
    sdir = common.session_dir(mode, "select", _now_id())
    (sdir / "models.json").write_text(json.dumps(out, indent=1))
    (sdir / "session.json").write_text(json.dumps({"created": out["created"],
        "summary": f"{len(candidates)} candidates; metadata arm={metadata_arm}"}, indent=1))
    (ROOT / "status" / "models.json").write_text(json.dumps(out, indent=1))
    _write_candidates_chart(mode, candidates)
    print(f"select: {len(candidates)} candidates, metadata_arm={metadata_arm}")


def _write_candidates_chart(mode: str, candidates: list[dict[str, Any]]) -> None:
    spec = {"id": "select_candidates", "title": "Model ladder candidates (val)",
            "kind": "table", "x": [c["kind"] for c in candidates],
            "series": [{"name": k, "y": [round(float(c[k]), 4) for c in candidates]}
                       for k in ("val_macro_f1", "val_acc", "val_qwk_argmax",
                                 "val_qwk_regress_round", "wall_s")],
            "xlabel": "model", "ylabel": "", "notes": "SMALL smoke; rule pre-registered"}
    d = ROOT / "results" / mode / "charts"
    d.mkdir(parents=True, exist_ok=True)
    (d / "select_candidates.json").write_text(json.dumps(spec, indent=1))
    try:
        from wqa.charts import to_png, to_tex
        to_png(spec, ROOT / "results" / mode / "figures" / "fig_select_candidates.png")
        to_tex(spec, ROOT / "results" / mode / "tables" / "select_candidates.tex")
    except Exception as e:  # charts module is pipe-agent's; degrade gracefully
        log.warning("charts.unavailable", err=str(e))


def train_all(mode: str) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
        mlflow.set_experiment(f"wqa-{mode}")
        have_mlflow = True
    except Exception:
        have_mlflow = False
    sel_p = ROOT / "status" / "models.json"
    if not sel_p.exists():
        raise SystemExit("run --select first (status/models.json missing)")
    selected = json.loads(sel_p.read_text())
    best_kind = selected["metadata_arm"]
    best_params = next(c["params"] for c in selected["candidates"] if c["kind"] == best_kind)
    df = common.load_table(mode)
    live = ROOT / "results" / mode / "live"
    mdl_dir = ROOT / "results" / mode / "models"
    live.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)
    configs = sorted((ROOT / "configs" / "models").glob("*.yaml"))
    trained = []
    for cf in configs:
        cfg = yaml.safe_load(cf.read_text())
        name = cf.stem
        if name.startswith("rung3") or cfg["features"] == "text_plus_H":
            trained.append({"config": name, "status": "NOT RUN (no text arm on CPU tonight)"})
            continue
        family = cfg["features"] if cfg["features"] in ("metadata", "metadata_noH") else "metadata"
        kind = best_kind if cfg["model"] == "best_rung1" else cfg["model"]
        if kind not in ("majority", "logreg", "xgboost", "lightgbm", "catboost", "histgb",
                        "random_forest", "mlp", *common.custom_registry()):
            trained.append({"config": name, "status": f"SKIP unknown kind {kind}"})
            continue
        # every pre-registered learner is refitted with the settings its own validation search chose
        params = best_params if cfg["model"] == "best_rung1" else next(
            (c["params"] for c in selected["candidates"] if c["kind"] == kind and c.get("params")), {})
        Xtr, ytr = common.load_xy(mode, family, "train", df)
        Xva, yva = common.load_xy(mode, family, "val", df)
        data = {"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva}
        for seed in cfg.get("seeds", [42]):
            if common.stop_requested():
                trained.append({"config": name, "status": "STOPPED (data/STOP)"})
                print("train-all: STOP honoured")
                _finish(mode, trained)
                return
            res = _fit_score(kind, params, seed, data)
            with (live / f"{name}_s{seed}.jsonl").open("a") as f:
                f.write(json.dumps({"epoch": 1, "ts": _now_id(), **{k: res[k] for k in
                        ("val_macro_f1", "val_acc", "val_qwk_argmax", "wall_s")}}) + "\n")
            if "est" in data:
                with (mdl_dir / f"{name}_s{seed}.pkl").open("wb") as fb:
                    pickle.dump({"est": data.pop("est"), "family": family, "kind": kind,
                                 "params": params, "seed": seed}, fb)
            if have_mlflow:
                import mlflow
                with mlflow.start_run(run_name=f"{name}_s{seed}"):
                    mlflow.log_params({"config": name, "kind": kind, "seed": seed, **{
                        f"p_{k}": v for k, v in params.items()}})
                    mlflow.log_metrics({k: float(res[k]) for k in
                                        ("val_macro_f1", "val_acc", "val_qwk_argmax", "wall_s")})
            trained.append({"config": name, "seed": seed, "val_macro_f1": res["val_macro_f1"],
                            "wall_s": res["wall_s"]})
            log.info("train", config=name, seed=seed, f1=round(res["val_macro_f1"], 4))
    _finish(mode, trained)


def _finish(mode: str, trained: list[dict[str, Any]]) -> None:
    sdir = common.session_dir(mode, "train", _now_id())
    (sdir / "trained.json").write_text(json.dumps(trained, indent=1))
    (sdir / "session.json").write_text(json.dumps(
        {"created": _now_id(), "summary": f"{len(trained)} config-seed runs"}, indent=1))
    print(f"train-all: {len(trained)} entries")


def train_one(mode: str, name: str) -> None:
    """Single-config train (panel 'train' target). Custom plugins get a short Optuna pass
    (10 trials) when they define search_space; results land in live/, models/, train session."""
    cf = ROOT / "configs" / "models" / f"{name}.yaml"
    if not cf.exists():
        raise SystemExit(f"no configs/models/{name}.yaml")
    cfg = yaml.safe_load(cf.read_text())
    kind = cfg["model"]
    if kind == "best_rung1":
        selected = json.loads((ROOT / "status" / "models.json").read_text())
        kind = selected["metadata_arm"]
    family = cfg["features"] if cfg["features"] in ("metadata", "metadata_noH") else "metadata"
    df = common.load_table(mode)
    Xtr, ytr = common.load_xy(mode, family, "train", df)
    Xva, yva = common.load_xy(mode, family, "val", df)
    data = {"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva}
    params: dict[str, Any] = {}
    if kind != "majority":
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            # probe: empty space returns {} on a FixedTrial({}); a real space raises -> tune it
            has_space = True
            try:
                has_space = common.search_space(kind, optuna.trial.FixedTrial({})) != {}
            except Exception:
                has_space = True
            if has_space:
                study = optuna.create_study(direction="maximize",
                                            sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(lambda t: _fit_score(kind, common.search_space(kind, t), 42, data)
                               ["val_macro_f1"], n_trials=10, show_progress_bar=False)
                params = common.search_space(kind, optuna.trial.FixedTrial(study.best_params))
        except Exception as e:
            log.warning("train_one.optuna_skipped", err=str(e))
    live = ROOT / "results" / mode / "live"
    mdl_dir = ROOT / "results" / mode / "models"
    live.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)
    trained = []
    for seed in cfg.get("seeds", [42]):
        if common.stop_requested():
            trained.append({"config": name, "status": "STOPPED (data/STOP)"})
            break
        res = _fit_score(kind, params, seed, data)
        with (live / f"{name}_s{seed}.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": 1, "ts": _now_id(), **{k: res[k] for k in
                    ("val_macro_f1", "val_acc", "val_qwk_argmax", "wall_s")}}) + "\n")
        if "est" in data:
            with (mdl_dir / f"{name}_s{seed}.pkl").open("wb") as fb:
                pickle.dump({"est": data.pop("est"), "family": family, "kind": kind,
                             "params": params, "seed": seed}, fb)
        trained.append({"config": name, "seed": seed, "val_macro_f1": res["val_macro_f1"],
                        "wall_s": res["wall_s"]})
        log.info("train_one", config=name, seed=seed, f1=round(res["val_macro_f1"], 4))
    _finish(mode, trained)


def main() -> None:
    args = sys.argv[1:]
    mode = args[args.index("--mode") + 1] if "--mode" in args else "small"
    if "--select" in args:
        select(mode)
    elif "--train-all" in args:
        train_all(mode)
    elif "--train" in args:
        train_one(mode, args[args.index("--train") + 1])
    else:
        raise SystemExit("need --select | --train-all | --train <config>")


if __name__ == "__main__":
    main()
