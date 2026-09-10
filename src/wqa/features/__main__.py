"""CLI: python -m wqa.features --mode small [--splits] (Makefile `features`/`splits` targets)."""
from __future__ import annotations

import argparse

import structlog

from wqa.features import build, splits

structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"),
                                structlog.processors.JSONRenderer()])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="small")
    p.add_argument("--splits", action="store_true")
    args = p.parse_args()
    if args.splits:
        sizes = splits.run(args.mode)
        print(f"splits={sizes}")
    else:
        result = build.run(args.mode)
        print(f"rows={result['rows']} cols={result['cols']}")


if __name__ == "__main__":
    main()
