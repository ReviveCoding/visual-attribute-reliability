# Source-Task Forensics

## Why this audit exists

The initial simple final-confirmation eligibility predicate reconstructed 20,829 source pairs, while the canonical v1b train/development manifests contained 20,800 pairs.

A source-only forensic audit was created to identify that discrepancy before proceeding with official validation.

## Forensic boundary

The forensic audit:

- reads source train manifests and source train annotations only
- does **not** open official validation annotations
- does **not** scan official validation images
- does **not** load a model

## Finding

| Forensic field | Value |
|---|---:|
| Source manifest pairs | 20,800 |
| Baseline reconstructed pairs | 20,829 |
| Missing source pairs | 0 |
| Unexpected reconstructed pairs | 29 |

No exact non-data-derived filter reproduced the canonical source pair set. The audit identified a data-derived category filter based on categories observed in the source manifest.

The final confirmation uses the source-rule reconstruction contract rather than assuming that a simpler predicate is equivalent. The official-validation manifest summary records a successful source-train-rule reproduction contract with 20,800 source train/development pairs.

## Operational implication

For dataset-derived tasks, eligibility must be treated as part of the experiment specification. A plausible-looking predicate is not sufficient evidence that a final-validation sample construction matches the source task exactly.
