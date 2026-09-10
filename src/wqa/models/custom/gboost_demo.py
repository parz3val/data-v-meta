"""Custom model plugin: gboost_demo (created from panel).
Registry contract (src/wqa/models/custom/): module defines KIND and build(params, seed).
The estimator must be sklearn-compatible: fit(X, y), predict_proba(X) -> (n, 6).
Optional: search_space(trial) -> params dict for Optuna.
Train from panel: run target "train" with config "custom_gboost_demo".
"""
from typing import Any

KIND = "gboost_demo"


def build(params: dict[str, Any], seed: int) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    defaults: dict[str, Any] = {"n_estimators": 200, "max_depth": 3}
    defaults.update(params)
    return make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=seed, **defaults))


def search_space(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    }
