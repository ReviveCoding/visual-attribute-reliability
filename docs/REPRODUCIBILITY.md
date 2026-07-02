# Reproducibility Guide

## Environment used for the local evidence runs

| Component | Local evidence environment |
|---|---|
| Operating system | Windows 11 |
| Python | 3.11.9 |
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU |
| CUDA runtime reported by PyTorch | 12.8 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 4.51.3 |
| PEFT | 0.19.1 |
| Model | `google/siglip2-base-patch16-224` |

The repository’s tracked CI intentionally does not require this GPU environment.

## Local roots

Keep the following responsibilities separate:

| Role | Recommended location |
|---|---|
| Repository source, tests, configs, local venv | repository root |
| Raw Fashionpedia data | external dataset root, outside Git |
| Hugging Face cache | external model-cache root, outside Git |
| Runs, logs, analysis, recovery records | external output root, outside Git |
| Full local evidence release | `dist/` under the repository, ignored by Git |

Do not commit raw dataset files, model caches, checkpoints, release archives, or runtime logs into Git history.

## Reproduction flow

1. Verify the raw Fashionpedia dataset layout and integrity.
2. Run the train-only profile and task-definition audit.
3. Build the image-group-disjoint train/development manifest.
4. Fit the HOG baseline.
5. Run the frozen SigLIP2 baseline.
6. Audit SigLIP2 vision-attention LoRA targets.
7. Run matched frozen-head versus LoRA development experiments.
8. Freeze selected checkpoints from development results.
9. Run untouched official-validation final confirmation.
10. Freeze local evidence and execute local full-release contracts.
11. Run hosted static-fixture contracts and documentation validation.

## Hosted-CI verification

```powershell
python -m pip install -r requirements-ci.txt
python -m pytest -q tests/test_ci_release_fixture_contracts.py
python -B scripts/validate_documentation.py
```

This path validates source-level and static-evidence contracts only.

## Local full-evidence verification

With the local release bundle and project venv present:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q `
  tests\test_release_evidence_contracts.py `
  -p no:cacheprovider
```

The local test reads the evidence release. It does not retrain the model or re-score Fashionpedia.

## Dependency boundaries

- `requirements-ci.txt` is intentionally minimal for hosted static-contract verification.
- HOG, frozen SigLIP2, and LoRA dependencies are separated into dedicated requirements files.
- The full local evidence environment is expected to include the project venv used for original evidence generation and verification.

## Clean-environment caution

Do not delete `.venv` until a fresh environment rebuild has been qualified against the exact local evidence-contract tests. Several local drivers call `.venv\Scripts\python.exe` directly.
