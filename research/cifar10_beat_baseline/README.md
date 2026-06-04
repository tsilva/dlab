# CIFAR-10 Beat Baseline

This project tracks the CIFAR-10 milestone work. The committed files here hold
the research decisions and artifact pointers; generated metrics, reports, and
checkpoints remain in `outputs/`, `reports/`, and W&B Artifacts.

Migrated content as of 2026-06-04:

- local ResNet-18 gradient-clipping baseline run
- local ResNet-18 CIFAR-stem run
- downloaded W&B artifact for the current WRN-28-10 CutMix candidate

Use `run.project: cifar10_beat_baseline` on new configs so W&B runs can be
filtered by this milestone. Keep `run.study` aligned to the numbered experiment
folder when practical, while preserving the generated dlab run name as the
artifact cross-reference key.
