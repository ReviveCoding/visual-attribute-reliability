# Evidence Report

## Study question

Can a narrowly scoped, parameter-efficient vision adaptation improve seven-class neckline-attribute verification relative to a matched frozen visual control, while preserving explicit split, selection, and evidence boundaries?

The study uses Fashionpedia category 33 (`neckline`) and seven mutually exclusive neckline attributes under a train-only task-definition rule.

## Immutable task contract

The task configuration fixes:

- Target category: `33`, `neckline`
- Target attributes: `182, 183, 185, 187, 189, 190, 200`
- Eligible instance rule:
  - target category only
  - exactly one target attribute
  - positive bounding-box area
  - exclude zero-target and multi-target instances
- Development grouping key: `image_id`
- Development split: 80% train / 20% development with seed `20260701`

The group-stratified source manifest contained 20,800 eligible instances, split into 16,635 train and 4,165 development instances. There were 16,446 train image groups and 4,111 development image groups, with zero group overlap.

## Model arms

| Arm | Encoder | Trainable components | Selected development epoch |
|---|---|---|---:|
| HOG baseline | HOG image features | Logistic-regression classifier | n/a |
| Frozen control | SigLIP2 base patch16 224 image encoder | Matched classification head, 5,383 parameters | 6 |
| LoRA adaptation | Same SigLIP2 image encoder | Same head plus vision-attention LoRA | 5 |

The audited LoRA scope covers 48 vision-attention projection modules. The adapter contains 589,824 trainable parameters; the matched classifier head contains 5,383 trainable parameters. The pooling-head-only path was excluded from the LoRA target scope.

## Development evidence

| Development result | Accuracy | Balanced accuracy | Macro-F1 | Top-label ECE | Multiclass Brier | Top-2 accuracy |
|---|---:|---:|---:|---:|---:|---:|
| HOG + logistic regression | 0.4545 | 0.4615 | 0.4469 | 0.2760 | 0.7987 | 0.6744 |
| Frozen SigLIP2 + logistic regression | 0.6038 | 0.6130 | 0.5981 | 0.1044 | 0.5384 | 0.8120 |
| Matched frozen head, BF16 development | — | 0.6126 | 0.6049 | 0.0564 | — | 0.8072 |
| Vision-attention LoRA, BF16 development | — | 0.7007 | 0.7014 | 0.0598 | — | 0.8821 |

The LoRA arm improved development Macro-F1 by 0.0965 absolute relative to the matched frozen-head arm. Development results were used only for checkpoint selection. The frozen control selected epoch 6; LoRA selected epoch 5.

## Untouched official validation confirmation

After selecting checkpoints from train-only development results, the fixed official-validation confirmation contained:

- 654 eligible instances
- 644 eligible images
- zero source train/development to validation image overlap
- a source-task reconstruction contract covering 20,800 train/development pairs

| Final raw metric | Matched frozen control | Vision-attention LoRA | LoRA minus frozen |
|---|---:|---:|---:|
| Macro-F1 | 0.579770 | 0.668109 | **+0.088339** |
| Top-label ECE | 0.083956 | 0.065230 | -0.018725 |

The final-confirmation result is an absolute **+8.83 percentage-point** Macro-F1 improvement for the audited LoRA arm over the matched frozen control.

## Paired post-validation transitions

A post-validation analysis read saved prediction artifacts only. It did not load a model, run inference, train, re-score validation, or select a checkpoint.

| Paired transition | Count |
|---|---:|
| Frozen wrong, LoRA correct | 93 |
| Frozen correct, LoRA wrong | 42 |
| Both wrong | 164 |

LoRA corrected 51 more cases than it regressed on this fixed subset. This paired-transition count is descriptive and should be interpreted together with the final Macro-F1 result and stated task boundary.

## What is validated

- Train-only task definition and data-quality audit
- Image-group-disjoint development split
- HOG, frozen encoder, and matched-head LoRA comparisons
- Audited 48-module vision-attention LoRA scope
- Fixed checkpoint selection before final validation access
- Untouched official-validation confirmation
- Local full-release SHA-256 and source-artifact evidence contracts
- Hosted-CI static fixture contracts

## What is not claimed

- Full Fashionpedia benchmark coverage
- Consumer-to-shop retrieval performance
- Production fashion-catalog performance
- Post-calibration LoRA superiority
- Independent external replication
- Hosted-CI model training, inference, or checkpoint verification
