# Results

| ID | Experiment | Best Val Acc | Best Val Loss | Test Acc | Decision | Report |
|---|---|---:|---:|---:|---|---|
| 001 | ResNet-18 gradient-clipping baseline | 0.6880 | 1.8596 | - | Baseline only | experiments/001_resnet18_gradclip_baseline/report.md |
| 002 | ResNet-18 CIFAR stem | 0.6948 | 0.8702 | - | Keep as evidence that the CIFAR stem helps early validation behavior | experiments/002_resnet18_cifar_stem/report.md |
| 003 | WRN-28-10 CutMix vs erasing candidate | 0.9640 | 0.4197 | - | Current strongest migrated candidate; test set not used for selection | experiments/003_wrn28_10_cutmix_vs_erasing/report.md |

Notes:

- Metrics above were migrated from existing local CSVs/reports/artifact downloads.
- The WRN row reports the best observed validation accuracy from the downloaded metrics CSV; its selected checkpoint summary in the generated report records `evaluation/selected/val/acc = 0.9624`.
- Test accuracy is intentionally blank until a final candidate is selected.
