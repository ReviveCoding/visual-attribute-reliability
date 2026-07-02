# CI Boundary

This repository deliberately separates a **hosted static-evidence gate** from a **local full-evidence gate**.

## Hosted GitHub Actions gate

The workflow in `.github/workflows/ci.yml` runs:

1. Python setup and the pinned CI dependency set
2. Source and test compilation
3. Documentation contract validation
4. Static evidence-fixture contracts
5. JUnit XML artifact upload

The workflow checks a small fixture tracked in Git. It validates:

- Category 33 / `neckline` task identity
- Seven fixed target attributes
- Immutable sample-rule fields
- 20,800 source train/development reconstruction count
- Zero source train/development to official-validation image overlap
- 654 eligible validation instances across 644 images
- Frozen-versus-LoRA final metric contract
- Claim-boundary language
- Fixture-file SHA-256 values

A hosted CI pass means that the **tracked static fixture and repository contracts are intact**.

## Hosted CI does not do the following

- Download or open Fashionpedia raw images or annotations
- Load SigLIP2, PyTorch, Transformers, PEFT, or model checkpoints
- Run inference or embedding extraction
- Train a classifier or adapter
- Re-score validation data
- Select a checkpoint
- Verify full local release-file or source-artifact hashes

## Local full-evidence gate

`tests/test_release_evidence_contracts.py` is intentionally local-only. It validates the full local release-evidence bundle, including:

- Full checkpoint presence
- Staged source-artifact SHA-256 values
- Full release-file SHA-256 manifest
- Full release semantic contract
- Final confirmation and task-config consistency

The bundle stays out of Git history because it contains local evidence artifacts and large model/checkpoint material. A local full-evidence gate pass is therefore a stronger but local qualification, not a replacement for independent external replication.

## Interpretation

| Result | Accurate interpretation |
|---|---|
| GitHub Actions green check | The tracked CI fixture, documentation contracts, source syntax, and static evidence assertions passed. |
| Local full-evidence contracts pass | The local hash-verified release bundle, checkpoints, and staged evidence match the documented contract. |
| Both pass | The repository’s tracked static contract and the local full evidence release agree under their respective boundaries. |

Do not represent a hosted CI pass as end-to-end model retraining, full Fashionpedia evaluation, or deployment validation.
