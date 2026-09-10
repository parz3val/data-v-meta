"""Shared model-arm helpers (WP5/6). In: features parquet + splits. Out: X/y per family+split.
Usage: X, y = load_xy("small", "metadata", "train")."""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wqa.config import ROOT, load

NON_FEATURES = {"pageid", "title", "label", "split", "temporal_holdout", "grade", "grade_rank"}
GH_FAMILIES = ("history", "contributor", "pageviews")  # g(H) per THEORY.md §1


def _features_path(mode: str) -> Path:
    d = ROOT / "data" / mode / "features"
    cands = sorted(d.glob("*.parquet"))
    if not cands:
        raise SystemExit(f"no features parquet in {d} — run `make features MODE={mode}`")
    return cands[0]


def load_table(mode: str) -> pd.DataFrame:
    df = pd.read_parquet(_features_path(mode))
    if "split" not in df.columns:
        sp = ROOT / "data" / mode / "splits"
        named = [(n, sp / f"{n}.parquet") for n in ("train", "val", "test")]
        if all(p.exists() for _, p in named):
            # per-split files (P2 layout): map pageid -> split name
            def _cols(p):
                have = pd.read_parquet(p).columns
                return [c for c in ("pageid", "temporal_holdout") if c in have]
            asg = pd.concat([pd.read_parquet(p, columns=_cols(p)).assign(split=n)
                             for n, p in named])
        else:
            f = sorted(sp.glob("*.parquet"))
            if not f:
                raise SystemExit(f"no split column and no splits parquet in {sp}")
            asg = pd.read_parquet(f[0])
        # guard: a pageid must map to exactly one split (dup raw rows would leak train->test)
        asg = asg.drop_duplicates("pageid", keep="first")
        df = df.drop_duplicates("pageid", keep="first").merge(asg, on="pageid", how="inner")
    if df["label"].dtype.kind not in "iu" and "grade_rank" in df.columns:
        # P2 layout: label = grade string, grade_rank = 0-5 ordinal
        df["grade"] = df["label"]
        df["label"] = df["grade_rank"].astype(int)
    return df


def family_columns(df: pd.DataFrame, family: str) -> list[str]:
    cfg = load("features")
    gh = [c for fam in GH_FAMILIES for c in cfg.get(fam, [])]
    numeric = [c for c in df.columns if c not in NON_FEATURES and df[c].dtype.kind in "ifb"]
    if family == "metadata":
        return numeric
    if family == "metadata_noH":
        return [c for c in numeric if c not in gh]
    if family == "gH":
        return [c for c in numeric if c in gh]
    raise SystemExit(f"unknown family {family}")


def load_xy(mode: str, family: str, split: str,
            df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    df = load_table(mode) if df is None else df
    part = df[df["split"] == split]
    cols = family_columns(df, family)
    return part[cols].fillna(0), part["label"].to_numpy(dtype=int)


def expected_grade_round(probs: np.ndarray) -> np.ndarray:
    """Regress-then-round decision rule: round(sum_k k*p(k)); QWK-optimal (THEORY.md §3)."""
    return np.clip(np.rint(probs @ np.arange(probs.shape[1])), 0, 5).astype(int)


def stop_requested() -> bool:
    return (ROOT / "data" / "STOP").exists()


def session_dir(mode: str, stage: str, run_id: str) -> Path:
    d = ROOT / "results" / mode / stage / run_id
    d.mkdir(parents=True, exist_ok=True)
    cur = ROOT / "results" / mode / stage / "current"
    if cur.is_symlink() or cur.exists():
        cur.unlink()
    cur.symlink_to(d.name)
    return d


def custom_registry() -> dict[str, Any]:
    """Plugin registry: src/wqa/models/custom/*.py exposing KIND + build(params, seed)
    (optional search_space(trial)). Panel-created models land here and are first-class."""
    import importlib
    reg: dict[str, Any] = {}
    cdir = Path(__file__).parent / "custom"
    if not cdir.is_dir():
        return reg
    for p in sorted(cdir.glob("*.py")):
        if p.stem.startswith("_"):
            continue
        try:
            m = importlib.import_module(f"wqa.models.custom.{p.stem}")
            if hasattr(m, "KIND") and hasattr(m, "build"):
                reg[m.KIND] = m
        except Exception as e:  # a broken plugin must not sink the whole run
            import structlog
            structlog.get_logger().warning("custom_plugin.broken", plugin=p.stem, err=str(e))
    return reg


def make_estimator(kind: str, params: dict[str, Any], seed: int) -> Any:
    reg = custom_registry()
    if kind in reg:
        return reg[kind].build(params, seed)
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed, **params))
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="multi:softprob", num_class=6, random_state=seed,
                             n_jobs=4, verbosity=0, **params)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective="multiclass", num_class=6, random_state=seed,
                              n_jobs=4, verbose=-1, **params)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(loss_function="MultiClass", random_seed=seed, verbose=False, **params)
    if kind == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed, **params)
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=4, **params)
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), MLPClassifier(max_iter=400, random_state=seed, **params))
    raise SystemExit(f"unknown estimator kind {kind}")


def search_space(kind: str, trial: Any) -> dict[str, Any]:
    reg = custom_registry()
    if kind in reg:
        return reg[kind].search_space(trial) if hasattr(reg[kind], "search_space") else {}
    if kind == "logreg":
        return {"C": trial.suggest_float("C", 0.01, 100, log=True)}
    if kind == "xgboost":
        return {"max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0)}
    if kind == "lightgbm":
        return {"num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 800)}
    if kind == "catboost":
        return {"depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "iterations": trial.suggest_int("iterations", 100, 800)}
    if kind == "histgb":
        return {"max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 600)}
    if kind == "random_forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)}
    if kind == "mlp":
        width = trial.suggest_categorical("width", [64, 128, 256])
        layers = trial.suggest_int("layers", 2, 3)
        return {"hidden_layer_sizes": tuple([width] * layers),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)}
    return {}
