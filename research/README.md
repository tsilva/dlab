# Research Projects

This directory is the committed research notebook for milestone-level work.
Generated training artifacts stay in `outputs/`, generated per-run reports stay in
`reports/`, and durable remote checkpoints/configs/metrics/reports are stored as
W&B Artifacts.

For manual W&B run triage, use the [Run Debugging Playbook](debugging_playbook.md).

Use this hierarchy:

```text
research/<project_id>/
  project.yaml
  README.md
  results.md
  checkpoints.md
  experiments/
    <nnn_experiment_id>/
      experiment.yaml
      report.md
      runs.csv
      artifact_manifest.md
```

The generated dlab run name is the cross-reference key across local and remote
stores:

- local output directory: `outputs/<run_name>/`
- local generated report: `reports/<run_name>.md`
- W&B run name: `<run_name>`
- W&B run artifact: `<run_name>-run`

Large checkpoint binaries should not be committed to Git. Keep local checkpoint
copies under `checkpoints/` only as a cache/archive; W&B Artifacts are the
durable store.
