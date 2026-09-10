"""Stacked heterogeneous ensemble (Wolpert 1992): XGB + LGBM + HistGB + RF probs -> logreg meta.
Rung1 families err differently (tree structure vs bagging); stacking their probability
outputs with a multinomial logreg meta-learner is the cheapest route past the single-model
plateau (~0.70 macro-F1 small). Cost multiplies ~4x at inference - reported honestly in
the cost table; this probes the metadata ceiling, not the speed frontier.
"""
from typing import Any

KIND = "stack_meta"


def build(params: dict[str, Any], seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    c = float(params.get("meta_C", 1.0))
    estimators = [
        ("xgb", XGBClassifier(objective="multi:softprob", num_class=6, random_state=seed,
                              n_jobs=4, verbosity=0, max_depth=6, n_estimators=300,
                              learning_rate=0.1, subsample=0.9)),
        ("lgbm", LGBMClassifier(objective="multiclass", num_class=6, random_state=seed,
                                n_jobs=4, verbose=-1, num_leaves=63, n_estimators=300)),
        ("histgb", HistGradientBoostingClassifier(random_state=seed, max_iter=300)),
        ("rf", RandomForestClassifier(random_state=seed, n_jobs=4, n_estimators=400,
                                      min_samples_leaf=2)),
    ]
    return StackingClassifier(estimators=estimators, stack_method="predict_proba", cv=5,
                              final_estimator=LogisticRegression(C=c, max_iter=2000,
                                                                 random_state=seed), n_jobs=1)


def search_space(trial: Any) -> dict[str, Any]:
    return {"meta_C": trial.suggest_float("meta_C", 0.05, 10, log=True)}
