# Results

## Incumbent Before Val-Acc Selection Sweep

W&B evidence from the existing Fashion-MNIST CNN studies shows the current
incumbent family is the shallow/wide CNN with `[64, 128, 256]` channels,
cosine schedule, EMA, label smoothing `0.02`, and random affine augmentation.

| Variant | Seeds | Mean Val Acc | Std | Best Val Acc | Mean Val Loss |
|---|---:|---:|---:|---:|---:|
| dropout `0.1`, wd `0.00003` | 3 | 0.951500 | 0.001500 | 0.953000 | 0.266265 |
| dropout `0.2`, wd `0.00003` | 3 | 0.951389 | 0.001858 | 0.953500 | 0.265328 |
| dropout `0.2`, wd `0.00010` | 3 | 0.950111 | 0.002750 | 0.952833 | 0.266821 |

Best observed single run:

- W&B run `qwfq2ihy`
- seed `3`
- dropout `0.2`
- weight decay `0.00003`
- validation accuracy `0.953500`
- validation loss `0.258240`

## Diagnostic Insight

The main failure mode is not lack of raw depth. Plain deeper CNNs and residual
deep variants did not beat the shallow/wide family; the worst no-BN deep runs
collapsed to chance-level accuracy with loss near `log(10)`, which indicates an
optimization or gradient-flow failure rather than ordinary overfitting.

The next useful search neighborhood is therefore selection and regularization
around the proven shallow/wide architecture, not adding more depth.

## Next Decision Gate

Run `001_wide_cnn_val_acc_selection`, then rank by mean validation accuracy
across seeds. Use validation loss only as a tie-breaker. Do not run test
evaluation until one validation winner is selected.
