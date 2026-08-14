# MNIST RNN Validation Accuracy

Maximize MNIST validation accuracy using recurrent sequence models while preserving validation-only model selection.

## Current Best

As of 2026-06-18, the best confirmed configuration is a one-layer GRU over MNIST
columns with hidden size 384, last-state pooling, AdamW at `lr=0.003`, cosine
scheduling, `weight_decay=0.001`, batch size 512, light image augmentation,
dropout 0.2, no label smoothing, and gradient clipping at 2.0.

Across seeds 1337, 2024, and 9001 it reached mean selected validation accuracy
`0.993667` with best single-seed selected validation accuracy `0.994667`.

After explicit test evaluation on 2026-06-18, the same config also had the best
mean test accuracy among the seven confirmed candidates: `0.992700` mean test
accuracy, `0.000748` population standard deviation, and `0.024245` mean test
loss. Because the test set was used after validation selection, treat this as a
final audit result, not as a clean hyperparameter-search signal.
