"""Isotonic-calibrated gradient boosting (Zadrozny & Elkan 2002; Guo et al. 2017 motivation).
Best current rung1 metadata models show ECE 0.09-0.16 -> unusable as confidence signal
(e.g. for triaging review queues). CalibratedClassifierCV(isotonic, cv=3) targets ECE
without touching ranking; macro-F1 should hold, ECE should drop well under 0.05.
"""
from typing import Any

KIND = "calibrated_gbt"


def build(params: dict[str, Any], seed: int) -> Any:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    defaults: dict[str, Any] = {"max_depth": 6, "learning_rate": 0.1, "max_iter": 300}
    defaults.update(params)
    base = HistGradientBoostingClassifier(random_state=seed, **defaults)
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def search_space(trial: Any) -> dict[str, Any]:
    return {"max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500)}
