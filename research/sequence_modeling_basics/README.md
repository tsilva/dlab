# Sequence Modeling Basics

Learn recurrent sequence modeling by treating MNIST images as row sequences and comparing recurrent classifiers under low compute budgets.

Initial implementation:

- model config: `configs/model/rnn.yaml`
- baseline experiment config: `configs/experiment/mnist_rnn_sequence.yaml`
- W&B project key: `run.project: sequence_modeling_basics`

Filtered W&B saved views:

- Training monitor: https://wandb.ai/tsilva/dlab?nw=v8xhirjrhqo
- Evaluation monitor: https://wandb.ai/tsilva/dlab?nw=qtrxpvmluc2
- Forensics: https://wandb.ai/tsilva/dlab?nw=kly6yon709o
- Minimal gradient debug: https://wandb.ai/tsilva/dlab?nw=9vgagpvrpzk
- Gradient diagnostics: https://wandb.ai/tsilva/dlab?nw=hg8un7imhgn
- Sweep comparison: https://wandb.ai/tsilva/dlab?nw=hyja0yk5qto
