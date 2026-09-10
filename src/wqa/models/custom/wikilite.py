"""wikilite: distilled micro-model (Hinton et al. 2015 idea, tree flavour a la Bucila 2006).
Teacher = strong HistGB on metadata; student = tiny LightGBM (depth<=4, <=40 trees)
trained on the teacher's soft targets (per-class prob regression via sample weighting).
Goal: quality score in microseconds, exportable to pure JSON/JS -> runs client-side
(browser extension, bots, Toolforge) with zero dependencies. Fidelity to teacher and
accuracy loss are both reported; export happens in fit() -> results/<mode>/wikilite/.
"""
from pathlib import Path
from typing import Any

import numpy as np

KIND = "wikilite"


class Distilled:
    def __init__(self, params: dict[str, Any], seed: int):
        self.params = dict(params)
        self.seed = seed
        self.classes_ = np.arange(6)

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {"params": self.params, "seed": self.seed}

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier
        teacher = HistGradientBoostingClassifier(random_state=self.seed, max_iter=300,
                                                 max_depth=6)
        teacher.fit(X, y)
        soft = teacher.predict_proba(X)
        # soft-target distillation with a hard-label anchor: replicate rows per class,
        # weight = alpha*soft + (1-alpha)*onehot(y); LGBM accepts weighted multiclass fine
        alpha = float(self.params.pop("alpha", 0.7))
        n, k = soft.shape
        onehot = np.zeros_like(soft)
        onehot[np.arange(n), np.asarray(y)] = 1.0
        w = alpha * soft + (1 - alpha) * onehot
        Xrep = np.repeat(np.asarray(X), k, axis=0)
        yrep = np.tile(np.arange(k), n)
        wrep = w.reshape(-1)
        keep = wrep > 1e-3
        student = LGBMClassifier(objective="multiclass", num_class=6, random_state=self.seed,
                                 n_jobs=4, verbose=-1, **self.params)
        student.fit(Xrep[keep], yrep[keep], sample_weight=wrep[keep])
        self.teacher_ = teacher
        self.student_ = student
        self.fidelity_ = float((student.predict(np.asarray(X)) == soft.argmax(1)).mean())
        self.feature_names_ = list(getattr(X, "columns", [f"f{i}" for i in range(np.asarray(X).shape[1])]))
        self._export()
        return self

    def _export(self):
        """Dump the student as portable JSON next to results (any language can score it)."""
        try:
            from wqa.config import ROOT
            out = ROOT / "results" / "small" / "wikilite"
            out.mkdir(parents=True, exist_ok=True)
            model_json = self.student_.booster_.dump_model()
            (out / f"wikilite_s{self.seed}.json").write_text(
                __import__("json").dumps({"features": self.feature_names_,
                                          "n_classes": 6, "fidelity_train": self.fidelity_,
                                          "model": model_json}))
        except Exception:
            pass  # export is a bonus, never fails training

    def predict_proba(self, X):
        return self.student_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def build(params: dict[str, Any], seed: int) -> Any:
    defaults: dict[str, Any] = {"n_estimators": 40, "num_leaves": 15, "max_depth": 4,
                                "learning_rate": 0.15, "alpha": 0.7}
    defaults.update(params)
    return Distilled(defaults, seed)


def search_space(trial: Any) -> dict[str, Any]:
    return {"n_estimators": trial.suggest_int("n_estimators", 20, 60),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
            "alpha": trial.suggest_float("alpha", 0.3, 0.9)}
