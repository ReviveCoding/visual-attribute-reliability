# Visual Attribute Reliability

[![Fashionpedia Evidence CI](https://github.com/ReviveCoding/visual-attribute-reliability/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ReviveCoding/visual-attribute-reliability/actions/workflows/ci.yml)

A reproducible, reliability-first framework for **visual attribute verification**. The reference study uses a category-scoped Fashionpedia neckline task to compare a matched frozen SigLIP2 control with an audited vision-attention LoRA adaptation under leakage-aware evaluation and explicit evidence contracts.

## Why this repository exists

Vision adaptation results are easy to overstate when task construction, validation access, control design, or evidence provenance are unclear. This repository makes those boundaries explicit:

- **Train-only task definition** with category 33 (`neckline`), exactly one target attribute, and positive bounding-box area.
- **Image-group-disjoint development split** so an image cannot appear in both train and development.
- **Matched control comparison** between a frozen image encoder with a trainable head and a vision-attention LoRA arm with the same head.
- **Untouched official validation confirmation** after selecting development checkpoints.
- **Separate hosted-CI and local evidence gates** so a green GitHub check is not misrepresented as full model retraining or full-release verification.

## Headline result

On the fixed official Fashionpedia validation subset, the audited LoRA arm achieved **0.6681 Macro-F1** versus **0.5798 Macro-F1** for the matched frozen control: an **absolute improvement of 0.0883, or +8.83 percentage points**.

| Final confirmation metric | Matched frozen control | Vision-attention LoRA | Difference |
|---|---:|---:|---:|
| Macro-F1 | 0.5798 | 0.6681 | **+0.0883** |
| Top-label ECE, raw | 0.0840 | 0.0652 | -0.0187 |
| Selected development epoch | 6 | 5 | — |

The final-confirmation subset contains **654 eligible instances across 644 images**, with **zero image overlap** with source train/development data. The source-task reconstruction contract covers **20,800** source train/development pairs.

## Scope and claim boundaries

This is a category-scoped, seven-class neckline-attribute verification study, not a full Fashionpedia benchmark, consumer-to-shop retrieval benchmark, or production catalog system.

The seven target attributes are:

`round (neck)`, `v-neck`, `oval (neck)`, `sweetheart (neckline)`, `boat (neck)`, `scoop (neck)`, and `straight across (neck)`.

The evidence supports the stated frozen-versus-LoRA comparison under the documented protocol. It does **not** support a post-calibration LoRA-superiority claim, a full-dataset claim, or a claim of deployment performance.

## Evidence and reproducibility

| Evidence layer | What it verifies | What it does not verify |
|---|---|---|
| [Hosted CI fixture contracts](docs/CI_BOUNDARY.md) | Static task, split, metric, claim-boundary, and fixture-hash contracts tracked in Git | Model loading, checkpoints, raw Fashionpedia data, inference, training, or validation rescoring |
| [Local full evidence contracts](docs/LOCAL_EVIDENCE_RELEASE.md) | Checkpoint presence, staged source-artifact hashes, and full release-file SHA-256 manifest | Hosted retraining or a public model-serving workflow |
| [Evidence report](docs/EVIDENCE.md) | Experiment protocol, baselines, final confirmation, and error-transition analysis | A general claim beyond the fixed task and evidence boundary |

### Verify the tracked hosted-CI fixture

```powershell
python -m pip install -r requirements-ci.txt
python -m pytest -q tests/test_ci_release_fixture_contracts.py
python -B scripts/validate_documentation.py
```

### Verify the local full evidence release

The full release bundle is intentionally excluded from Git history. It must be present locally under `dist/` before running the local evidence-contract suite.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q `
  tests\test_release_evidence_contracts.py `
  -p no:cacheprovider
```

See [Local evidence release](docs/LOCAL_EVIDENCE_RELEASE.md) for the release hash, archive structure, and verification boundary.

## Repository layout

```text
.github/workflows/      Hosted static-contract CI
configs/                Immutable task and experiment configurations
scripts/                Dataset, model, audit, and documentation utilities
tests/                  Hosted-CI fixture and local full-release contract tests
docs/                   Evidence, CI, release, and reproducibility documentation
```

## Documentation

- [Evidence report](docs/EVIDENCE.md)
- [CI boundary](docs/CI_BOUNDARY.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Local evidence release](docs/LOCAL_EVIDENCE_RELEASE.md)
- [Source-task forensics](docs/FORENSICS.md)

## Data and model access

Raw Fashionpedia images and annotations, Hugging Face model cache, local checkpoints, and full release evidence archives are intentionally not committed to Git. Follow the applicable dataset and pretrained-model terms before obtaining or using those resources.
