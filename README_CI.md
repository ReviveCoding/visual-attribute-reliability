# Hosted CI Boundary

This repository intentionally separates two evidence gates.

## Hosted CI gate

`.github/workflows/ci.yml` runs only `tests/test_ci_release_fixture_contracts.py` against a small static fixture tracked in Git. It validates the immutable category-33 task rule, seven target attributes, source-rule reconstruction count, source-validation disjointness, frozen official-validation metrics, claim boundaries, and fixture SHA-256 integrity.

The workflow does **not** download Fashionpedia, load SigLIP2, load LoRA checkpoints, run inference, train, or score validation.

## Local evidence-release gate

`tests/test_release_evidence_contracts.py` is intentionally local-only because it validates the full local evidence release, including checkpoint presence and source/release SHA-256 contracts. The full bundle remains excluded from Git to avoid committing model artifacts and the local release archive into repository history.

A hosted CI pass therefore means the tracked static evidence contract is intact. It is not a substitute for the local full evidence-release contract suite.
