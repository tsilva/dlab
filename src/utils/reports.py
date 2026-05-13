from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def write_experiment_report(
    cfg: DictConfig,
    metrics: dict[str, Any],
    run_dir: str | Path,
    reports_dir: str | Path = "reports",
) -> Path:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    name = cfg.get("experiment_name", "experiment")
    report_path = reports_path / f"{name}.md"
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True)

    metric_lines = []
    for key in sorted(metrics):
        value = metrics[key]
        try:
            value = float(value)
        except (TypeError, ValueError):
            pass
        metric_lines.append(f"- `{key}`: {value}")

    report_path.write_text(
        "\n".join(
            [
                f"# {name}",
                "",
                "## Final Metrics",
                "",
                *(metric_lines or ["- No metrics recorded."]),
                "",
                "## Run Artifacts",
                "",
                f"- Run directory: `{Path(run_dir)}`",
                "",
                "## Config",
                "",
                "```yaml",
                config_yaml.rstrip(),
                "```",
                "",
                "## Observations",
                "",
                "- What changed?",
                "- What was surprising?",
                "- What should be tried next?",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report_path

