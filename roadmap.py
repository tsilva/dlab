from __future__ import annotations

import argparse

import pandas as pd

from src.roadmap import find_stage, load_roadmap, roadmap_table, run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and inspect the dlab learning roadmap.")
    parser.add_argument("--roadmap", default="configs/roadmap/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    run = subparsers.add_parser("run")
    run.add_argument("stage")
    run.add_argument("--backend", default="local", choices=["local", "wandb"])
    run.add_argument("--launcher", default=None)
    run.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    stages = load_roadmap(args.roadmap)

    if args.command == "list":
        print(pd.DataFrame(roadmap_table(stages)).to_string(index=False))
        return

    if args.command == "run":
        stage = find_stage(stages, args.stage)
        exit_codes = run_stage(
            stage,
            backend=args.backend,
            launcher=args.launcher,
            dry_run=args.dry_run,
        )
        failures = [code for code in exit_codes if code != 0]
        if failures:
            raise SystemExit(f"Roadmap stage failed: {failures}")


if __name__ == "__main__":
    main()
