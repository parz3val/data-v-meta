"""Fixed metrics (SPEC §6). In: y_true/y_pred (+probs). Out: dict of metrics + bootstrap CIs.
Usage: from wqa.eval.metrics import all_metrics, bootstrap_ci."""
from typing import Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score


def acc(y: Sequence[int], p: Sequence[int]) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def macro_f1(y: Sequence[int], p: Sequence[int]) -> float:
    return float(f1_score(y, p, average="macro", labels=list(range(6)), zero_division=0))


def per_class_f1(y: Sequence[int], p: Sequence[int]) -> list[float]:
    return [float(v) for v in f1_score(y, p, average=None, labels=list(range(6)), zero_division=0)]


def within_one(y: Sequence[int], p: Sequence[int]) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p)) <= 1))


def ordinal_mae(y: Sequence[int], p: Sequence[int]) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def qwk(y: Sequence[int], p: Sequence[int]) -> float:
    return float(cohen_kappa_score(y, p, weights="quadratic", labels=list(range(6))))


def all_metrics(y: Sequence[int], p: Sequence[int]) -> dict[str, float | list[float]]:
    return {"macro_f1": macro_f1(y, p), "qwk": qwk(y, p), "within_one": within_one(y, p),
            "acc": acc(y, p), "per_class_f1": per_class_f1(y, p), "ordinal_mae": ordinal_mae(y, p)}


def bootstrap_ci(y: Sequence[int], p: Sequence[int], metric: str = "macro_f1",
                 n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% percentile CI over resampled (y, p) pairs."""
    fns = {"macro_f1": macro_f1, "qwk": qwk, "within_one": within_one, "acc": acc,
           "ordinal_mae": ordinal_mae}
    fn = fns[metric]
    rng = np.random.default_rng(seed)
    ya, pa = np.asarray(y), np.asarray(p)
    stats = [fn(ya[idx], pa[idx]) for idx in
             (rng.integers(0, len(ya), len(ya)) for _ in range(n_boot))]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
