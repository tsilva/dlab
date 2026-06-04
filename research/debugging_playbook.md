# Run Debugging Playbook

Use this checklist when a run disappoints, fails, or behaves strangely. Start
with the general pass before using model-specific sections. The goal is to avoid
jumping to a favorite explanation before ruling out simpler failure classes.

## General Algorithm

### 1. Identify the Run and Baseline

Open the run in W&B and write down:

- run name
- `run.project`, `run.stage`, `run.study`
- seed
- selected checkpoint metric, usually `evaluation/selected/val/acc` or `evaluation/selected/val/loss`
- closest comparable baseline

Ask one question first: what changed relative to the closest run that worked?

Do not diagnose a run in isolation if a controlled baseline exists. Most useful
debugging comes from comparing two runs with only one or two changed variables.

### 2. Check Train and Validation Learning

Dashboard: Training monitor.

Metrics:

- `train/acc_epoch`
- `val/acc`
- `train/loss_epoch`
- `val/loss`

Interpretation:

| Pattern | Likely Meaning | Next Step |
|---|---|---|
| train improves, val improves | run is learning | compare against target/baseline |
| train improves, val flat/worse | overfitting, data mismatch, or validation-time issue | inspect regularization, splits, augmentations |
| train flat, val flat | optimization, architecture, data, or logging issue | go to LR and gradients |
| train loss becomes NaN/inf | numerical instability | inspect LR, precision, clipping, data |
| loss flat near random baseline | model is not extracting signal | compare baseline and diagnostics |

Random-baseline anchors:

- 10-class classification random loss is about `log(10) = 2.3026`.
- 10-class random accuracy is about `0.10`.

### 3. Check the Optimizer Is Actually Moving

Dashboard: Training monitor.

Metrics:

- `train/lr`
- `lr-Adam`, `lr-AdamW`, or `lr-SGD`
- `train/grad_norm`

Interpretation:

| Pattern | Likely Meaning |
|---|---|
| LR is zero or missing | scheduler/config bug |
| LR much larger than expected | unstable training risk |
| LR much smaller than expected | learning too slow |
| grad norm zero/missing | no gradient path, frozen parameters, or logging issue |
| grad norm exists but metrics flat | gradients exist, but updates are not useful |

### 4. Rule Out Gradient Explosion or Clipping-Dominated Training

Dashboard: Minimal gradient debug.

Metrics:

- `train/grad_clip/raw_norm_step`
- `train/grad_clip/threshold_step`
- `train/grad_clip/clip_coef_step`
- `train/grad_clip/was_clipped_step`
- `train/grad_clip/was_clipped_epoch`

Interpretation:

| Pattern | Likely Meaning |
|---|---|
| `clip_coef_step` near `1.0`, `was_clipped` near `0.0` | clipping is not the failure reason |
| `clip_coef_step` often far below `1.0` | exploding or oversized gradients |
| raw norm repeatedly spikes before bad metrics | instability or too-high LR |
| clipping active and learning flat | lower LR, change initialization, or use architecture with better signal flow |

### 5. Check Whether the Head Is Confused or Confident-Wrong

Dashboard: Training monitor or Gradient diagnostics if available.

Metrics:

- `train/pred_entropy_epoch`
- `val/pred_entropy`
- `train/pred_entropy_normalized_epoch`
- `val/pred_entropy_normalized`
- `train/pred_max_prob_epoch`
- `val/pred_max_prob`

Interpretation:

| Pattern | Likely Meaning |
|---|---|
| entropy near maximum, max prob near random | head is confused/random |
| entropy falls, max prob rises, accuracy rises | normal learning |
| entropy falls, max prob rises, accuracy stays low | confident wrong predictions |
| train entropy falls, val entropy stays high | overfitting or validation mismatch |

For 10 classes:

- maximum entropy is `log(10) = 2.3026`
- normalized entropy near `1.0` means almost uniform predictions
- max probability near `0.10` means almost random confidence

### 6. Inspect Generalization and Selection

Dashboard: Evaluation monitor.

Metrics:

- `evaluation/selected/val/acc`
- `evaluation/selected/val/loss`
- `checkpoint/best_score`
- `val/acc`
- `val/loss`
- `test/acc`, only when explicitly approved and relevant

Interpretation:

| Pattern | Likely Meaning |
|---|---|
| selected metrics better than final metrics | checkpoint selection matters; report selected metrics |
| selected and final both poor | training failure or bad setup |
| val good, test poor | possible validation overfitting or distribution mismatch |
| val loss improves while val acc worsens | calibration/regularization tradeoff |

### 7. Inspect Data and Output Forensics

Dashboard: Forensics.

Metrics or media:

- `examples/predictions`
- `errors/val_misclassifications`
- `errors/val_confusion_matrix`
- `errors/val_misclassification_count_total`

Interpretation:

| Pattern | Likely Meaning |
|---|---|
| predictions all one class | class collapse, imbalance, or head bias |
| labels/images mismatched | dataset transform or target bug |
| errors concentrated in similar classes | normal model limitation or insufficient inductive bias |
| errors uniformly random | model learned little usable signal |

### 8. Decide the Failure Class

Use the first matching category:

| Failure Class | Evidence |
|---|---|
| data/config bug | labels wrong, shapes wrong, baseline also fails unexpectedly |
| no learning | train and val flat, loss near random |
| overfitting | train improves, val worsens or stays flat |
| exploding gradients | clipping frequent, raw norm spikes, NaN/inf risk |
| vanishing/weak signal | gradients exist, train flat, architecture-specific signal diagnostics weak |
| confident wrong | entropy drops, max probability rises, accuracy remains low |
| insufficient capacity | train improves slowly and plateaus below target without instability |
| excessive regularization | train accuracy suppressed, loss may improve slowly, confidence stays low |

End every debug pass with a concrete next experiment. Change one thing when
possible.

## RNN and Sequence Models

Use this section after the general pass says the run is not simply overfitting,
not exploding, and not obviously misconfigured.

### RNN Scan Order

1. Compare sequence representation against baseline.
2. Check classification confidence.
3. Check hidden-state behavior by timestep.
4. Check recurrent gradient split.
5. Decide whether to test gating, shorter sequences, or pooling changes.

### Sequence Representation

Metrics/config:

- `model.params.sequence_axis`
- `model.params.rnn_type`
- `model.params.pooling`
- `model.params.hidden_dim`

Interpretation:

| Pattern | Likely Meaning |
|---|---|
| short sequence works, long sequence fails | long-range credit assignment or temporal representation issue |
| rows work, pixels fail | representation length is the likely stressor |
| mean pooling beats last pooling | final state is losing useful early evidence |
| GRU/LSTM beats vanilla RNN | gating helps preserve/update signal |

### Prediction Entropy

Metrics:

- `train/pred_entropy_epoch`
- `val/pred_entropy`
- `train/pred_max_prob_epoch`
- `val/pred_max_prob`

Interpretation:

| Pattern | Meaning |
|---|---|
| entropy near max for all epochs | RNN never forms class evidence |
| entropy drops but accuracy stays low | RNN forms confident but wrong evidence |
| train entropy drops only | overfits training sequences |

### Hidden-State Timestep Diagnostics

Metrics:

- `train/sequence/hidden_norm_t000_step`
- `train/sequence/hidden_norm_t25pct_step`
- `train/sequence/hidden_norm_t50pct_step`
- `train/sequence/hidden_norm_t75pct_step`
- `train/sequence/hidden_norm_tlast_step`
- `train/sequence/hidden_norm_last_to_first_ratio_step`
- `val/sequence/hidden_norm_last_to_first_ratio`

Interpretation:

| Pattern | Likely Meaning |
|---|---|
| late hidden norm collapses below early norm | weak signal persistence |
| hidden norms explode over timesteps | unstable recurrence |
| hidden norms are stable but entropy remains max | state magnitude exists but is not class-informative |
| validation hidden dynamics differ sharply from train | distribution or regularization mismatch |

Hidden-state norm is not accuracy. Treat it as a signal-flow proxy, not proof by
itself.

### Recurrent Gradient Split

Metrics:

- `train/recurrent_grad/input_kernel_norm_step`
- `train/recurrent_grad/recurrent_kernel_norm_step`
- `train/recurrent_grad/classifier_norm_step`
- `train/recurrent_grad/recurrent_to_input_ratio_step`
- `train/recurrent_grad/classifier_to_recurrent_ratio_step`

Interpretation:

| Pattern | Likely Meaning |
|---|---|
| classifier gradient dominates recurrent gradient | head can move, recurrent dynamics are weak |
| recurrent gradient much larger than input/classifier | unstable recurrence risk |
| recurrent-to-input ratio very low | transition is not receiving useful update signal |
| all gradient splits tiny | global signal is weak or loss is saturated |

### RNN Next Experiments

Pick the smallest experiment that tests the diagnosis:

| Diagnosis | Next Experiment |
|---|---|
| long sequence failure | compare GRU on same sequence |
| gating helps | compare LSTM on same sequence |
| final state loses evidence | try mean pooling |
| sequence too long | run 196, 392, 784 step ablation |
| confidence stays random | inspect labels/data, then try gated model |
| recurrent dynamics weak | increase hidden size only after gating/sequence ablation |

## CNN and Vision Classifiers

Use this section when training learns something but validation quality, errors,
or robustness disappoint.

### CNN Scan Order

1. Check train/val gap.
2. Inspect prediction examples and misclassifications.
3. Compare augmentation and regularization changes.
4. Check first-layer filters only when the model exposes them.
5. Decide whether failure is capacity, augmentation, architecture, or data.

### Common CNN Interpretations

| Pattern | Likely Meaning | Next Experiment |
|---|---|---|
| train high, val low | overfitting | stronger augmentation, weight decay, mixup/cutmix |
| train and val both low | undercapacity or optimization issue | larger model, LR change, better stem |
| val loss improves, val acc drops | calibration/regularization tradeoff | compare error counts and confidence |
| errors are mostly transformed/shifted examples | robustness issue | targeted augmentation or TTA probe |
| pretrained model underperforms on small images | stem/geometry mismatch | use CIFAR-sized stem or less aggressive downsampling |

### CNN Metrics and Media

- `train/acc_epoch`
- `val/acc`
- `train/loss_epoch`
- `val/loss`
- `errors/val_misclassifications`
- `errors/val_confusion_matrix`
- `filters`
- `evaluation/tta/val/acc`, if TTA was enabled

## MLP Classifiers

MLPs are useful baselines. Debug them mostly as optimization and capacity probes.

| Pattern | Likely Meaning | Next Experiment |
|---|---|---|
| MLP learns, CNN/RNN fails | model-specific implementation or representation issue |
| MLP flat at random | data, optimizer, labels, or global training bug |
| train high, val lower | capacity without inductive bias, overfitting |
| deeper MLP worse than shallow | optimization, normalization, dropout, or LR issue |

Useful checks:

- compare against simplest MLP baseline before blaming dataset
- inspect `train/pred_entropy_epoch` for confused vs confident-wrong
- use MLP as a data-pipeline sanity check

## Autoencoders, VAE, and VQ-VAE

Use reconstruction models to separate representation quality from classifier
accuracy.

Metrics:

- `train/recon_loss_epoch`
- `val/recon_loss`
- `train/kl_loss_epoch`, `val/kl_loss`
- `train/codebook_perplexity_epoch`, `val/codebook_perplexity`
- `train/codebook_utilization_epoch`, `val/codebook_utilization`
- `examples/reconstructions`
- `latent_traversals`

Interpretation:

| Pattern | Likely Meaning |
|---|---|
| recon loss high on train and val | undercapacity, optimizer, or decoder issue |
| train recon good, val recon poor | overfitting |
| VAE KL near zero | posterior collapse |
| VAE KL huge, recon poor | beta too strong or latent pressure too high |
| VQ codebook utilization low | codebook collapse |
| recon images blurry but loss low | objective mismatch or expected MSE behavior |

## Final Debug Note Template

Use this short structure in every experiment report:

```text
Observed:
- train:
- validation:
- optimizer/gradient:
- confidence:
- artifacts/forensics:

Ruled out:
- 

Leading diagnosis:
- 

Next controlled experiment:
- Change:
- Keep fixed:
- Expected pattern:
```
