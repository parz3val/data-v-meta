"""Budget guard (SPEC §7). Wraps long steps; logs to RUNLOG; raises BudgetExceeded.
Usage: python -m wqa.monitor.budget --check | with budget("P2_clean", hours): ..."""
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from wqa.config import ROOT, load


class BudgetExceeded(RuntimeError):
    pass


@contextmanager
def budget(phase: str, hours: float | None = None) -> Iterator[None]:
    caps = load("budgets")["phases_h"]
    cap_s = (hours if hours is not None else float(caps.get(phase, 1))) * 3600
    t0 = time.monotonic()
    try:
        yield
    finally:
        spent = time.monotonic() - t0
        with (ROOT / "status" / "RUNLOG.md").open("a") as f:
            f.write(f"budget {phase}: {spent:.0f}s of {cap_s:.0f}s\n")
        if spent > cap_s:
            raise BudgetExceeded(f"{phase}: {spent:.0f}s > {cap_s:.0f}s")


if __name__ == "__main__":
    cfg = load("budgets")
    assert cfg["attempts_per_step"] == 3 and cfg["phases_h"], "budgets.yaml malformed"
    print("budget config ok:", ", ".join(f"{k}={v}h" for k, v in list(cfg["phases_h"].items())[:4]), "...")
    sys.exit(0)
