from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_metrics_csv(run_dir: str | Path) -> pd.DataFrame:
    run_path = Path(run_dir)
    path = run_path / "metrics.csv"
    if not path.exists():
        matches = sorted(run_path.glob("**/metrics.csv"))
        if matches:
            path = matches[0]
    if not path.exists():
        raise FileNotFoundError(f"No metrics.csv found in {run_dir}")
    return pd.read_csv(path)


def summarize_run(run_dir: str | Path) -> pd.Series:
    df = load_metrics_csv(run_dir)
    numeric = df.select_dtypes("number")
    return numeric.ffill().iloc[-1]


def compare_runs(run_dirs: list[str | Path], metric: str = "val/loss") -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        summary = summarize_run(run_dir)
        rows.append({"run_dir": str(run_dir), metric: summary.get(metric)})
    return pd.DataFrame(rows).sort_values(metric)
