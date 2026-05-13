from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.runs import compare_runs, summarize_run
from src.analysis.wandb import summarize_wandb_study, write_wandb_study_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dlab experiment outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary")
    summary.add_argument("run_dir", type=Path)

    compare = subparsers.add_parser("compare")
    compare.add_argument("run_dirs", nargs="+", type=Path)
    compare.add_argument("--metric", default="val/loss")

    wandb_study = subparsers.add_parser("wandb-study")
    wandb_study.add_argument("study")
    wandb_study.add_argument("--project", default="dlab")
    wandb_study.add_argument("--entity", default=None)
    wandb_study.add_argument("--stage", default=None)
    wandb_study.add_argument("--metric", default="val/loss")
    wandb_study.add_argument("--goal", default="minimize", choices=["minimize", "maximize"])
    wandb_study.add_argument("--report", action="store_true")
    wandb_study.add_argument("--reports-dir", default="reports")

    args = parser.parse_args()
    if args.command == "summary":
        print(summarize_run(args.run_dir).to_string())
    elif args.command == "compare":
        print(compare_runs(args.run_dirs, metric=args.metric).to_string(index=False))
    elif args.command == "wandb-study":
        df = summarize_wandb_study(
            project=args.project,
            entity=args.entity,
            stage=args.stage,
            study=args.study,
            metric=args.metric,
            goal=args.goal,
        )
        print(df.to_string(index=False))
        if args.report:
            path = write_wandb_study_report(
                df,
                study=args.study,
                metric=args.metric,
                goal=args.goal,
                output_dir=args.reports_dir,
            )
            print(f"Wrote report: {path}")


if __name__ == "__main__":
    main()
