"""CLI: python -m wqa.eda --mode small (Makefile `eda` target)."""
from __future__ import annotations

import argparse

import structlog

from wqa.eda.run import run

structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"),
                                structlog.processors.JSONRenderer()])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="small")
    args = p.parse_args()
    result = run(args.mode)
    print(f"run_id={result['run_id']} rows={result['rows']} flagged={len(result['flagged'])}")


if __name__ == "__main__":
    main()
