"""Ordinal cumulative-link ensemble (Frank & Hall 2001 decomposition; CORAL-style shared idea).
Grades are ordered (Stub<Start<C<B<GA<FA): fit K-1 binary LightGBM models P(y > k),
reconstruct class probs as adjacent differences with monotonicity enforced.
Rationale: multiclass softmax discards order; cumulative decomposition uses it and
should cut ordinal MAE / raise QWK at equal cost (THEORY prediction, viva-defensible).
"""
from typing import Any

import numpy as np

KIND = "ordinal_coral"
N_CLASSES = 6


class CumulativeOrdinal:
    def __init__(self, params: dict[str, Any], seed: int):
        self.params = dict(params)
        self.seed = seed
        self.models_: list[Any] = []
        self.classes_ = np.arange(N_CLASSES)

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {"params": self.params, "seed": self.seed}

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        y = np.asarray(y)
        self.models_ = []
        for k in range(N_CLASSES - 1):
            m = LGBMClassifier(objective="binary", random_state=self.seed, n_jobs=4,
                               verbose=-1, **self.params)
            m.fit(X, (y > k).astype(int))
            self.models_.append(m)
        return self

    def predict_proba(self, X):
        # P(y>k) for k=0..4; enforce monotone non-increasing via cumulative min
        gt = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        gt = np.minimum.accumulate(gt, axis=1)
        ones = np.ones((gt.shape[0], 1))
        zeros = np.zeros((gt.shape[0], 1))
        cum = np.hstack([ones, gt, zeros])            # P(y>-1)..P(y>5)
        probs = cum[:, :-1] - cum[:, 1:]              # P(y=k)
        probs = np.clip(probs, 1e-9, None)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def build(params: dict[str, Any], seed: int) -> Any:
    defaults: dict[str, Any] = {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05}
    defaults.update(params)
    return CumulativeOrdinal(defaults, seed)


def search_space(trial: Any) -> dict[str, Any]:
    return {"num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600)}
