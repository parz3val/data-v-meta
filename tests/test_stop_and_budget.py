"""SPEC §10: stop-file honoured; budget guard raises BudgetExceeded."""
import time

import pytest

from wqa.config import ROOT
from wqa.models import common
from wqa.monitor.budget import BudgetExceeded, budget


def test_stop_file_roundtrip() -> None:
    stop = ROOT / "data" / "STOP"
    stop.parent.mkdir(exist_ok=True)
    try:
        assert not common.stop_requested()
        stop.touch()
        assert common.stop_requested()
    finally:
        stop.unlink(missing_ok=True)


def test_budget_guard_raises() -> None:
    with pytest.raises(BudgetExceeded):
        with budget("test_phase", hours=1e-6):
            time.sleep(0.02)


def test_expected_grade_round() -> None:
    import numpy as np
    probs = np.array([[0.5, 0.5, 0, 0, 0, 0], [0, 0, 0, 0, 0.2, 0.8]])
    assert common.expected_grade_round(probs).tolist() == [0, 5]  # 0.5 rounds to even (banker's)
