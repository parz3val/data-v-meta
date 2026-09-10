"""CLI: python -m wqa.collect --mode small [--smoke] (Makefile `data`/`data-smoke` targets)."""
from __future__ import annotations

import argparse

import structlog

from wqa.collect.run import run

structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"),
                                structlog.processors.JSONRenderer()])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="small")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    result = run(args.mode, smoke=args.smoke)
    print(f"per_grade={result['per_grade']} excluded={result['excluded']} "
          f"total={result['total']} dups={result['dups']} missing_rate={result['missing_rate']}")


if __name__ == "__main__":
    main()
