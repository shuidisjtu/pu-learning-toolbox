"""Command-line entry point for native deep-PU benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import SUPPORTED_METHODS, load_config, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trials, summary = run_benchmark(config, args.output, selected_methods=args.methods)
    print(f"Wrote {len(trials)} trials and {len(summary)} summary rows to {args.output}")


if __name__ == "__main__":
    main()
