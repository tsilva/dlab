from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis import compare_runs, summarize_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dlab experiment outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary")
    summary.add_argument("run_dir", type=Path)

    compare = subparsers.add_parser("compare")
    compare.add_argument("run_dirs", nargs="+", type=Path)
    compare.add_argument("--metric", default="val/loss")

    args = parser.parse_args()
    if args.command == "summary":
        print(summarize_run(args.run_dir).to_string())
    elif args.command == "compare":
        print(compare_runs(args.run_dirs, metric=args.metric).to_string(index=False))


if __name__ == "__main__":
    main()
