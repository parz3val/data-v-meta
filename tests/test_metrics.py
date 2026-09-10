"""SPEC §6: metrics vs hand-computed 12-row fixture (worked by hand, see values inline)."""
import pytest

from wqa.eval.metrics import all_metrics, bootstrap_ci

Y = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
P = [0, 1, 1, 1, 2, 3, 3, 2, 4, 4, 5, 4]  # 8 exact, 4 off-by-one


def test_hand_computed() -> None:
    m = all_metrics(Y, P)
    assert m["acc"] == pytest.approx(8 / 12)
    assert m["within_one"] == pytest.approx(1.0)
    assert m["ordinal_mae"] == pytest.approx(4 / 12)
    assert m["macro_f1"] == pytest.approx((2/3 + 0.8 + 0.5 + 0.5 + 0.8 + 2/3) / 6)
    assert m["qwk"] == pytest.approx(1 - 4 / 62)  # sum wO=4; sum wE=372/6=62 (hand-worked)
    assert m["per_class_f1"][1] == pytest.approx(0.8)


def test_bootstrap_ci_brackets_point() -> None:
    lo, hi = bootstrap_ci(Y, P, "macro_f1", n_boot=200, seed=1)
    assert lo <= all_metrics(Y, P)["macro_f1"] <= hi
