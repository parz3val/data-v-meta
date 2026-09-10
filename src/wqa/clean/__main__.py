"""CLI: python -m wqa.clean --mode small (Makefile `clean-data` target)."""
from __future__ import annotations

import argparse

import structlog

from wqa.clean.run import run

structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"),
                                structlog.processors.JSONRenderer()])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="small")
    args = p.parse_args()
    result = run(args.mode)
    print(f"rows={result['rows']} cols={result['cols']}")


if __name__ == "__main__":
    main()
