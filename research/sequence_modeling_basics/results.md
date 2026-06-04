# Results

| ID | Experiment | Best Val Acc | Best Val Loss | Test Acc | Decision | Report |
|---|---|---:|---:|---:|---|---|
| 001 | Vanilla RNN baseline on sequential MNIST | 0.9507 | 0.1783 | - | Keep as first recurrent baseline; durable W&B artifact uploaded from Modal T4 | experiments/001_rnn_baseline/report.md |
| 002 | Pixel-sequence vanilla RNN failure probe | 0.1090 | 2.3009 | - | Keep as failure baseline; compare GRU/LSTM on same 784-step sequence | experiments/002_pixel_rnn_failure_probe/report.md |
| 003 | Pixel-sequence RNN diagnostic rerun | 0.1090 | 2.3009 | - | Keep as diagnostic failure baseline; entropy remains maximal and recurrent signal is weak | experiments/003_pixel_rnn_diagnostic_rerun/report.md |
