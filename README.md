# Fashionpedia Test Hardening v1

This package adds a read-only pytest evidence-contract suite for the local evidence freeze.

It verifies:

1. Final official-validation metrics and selection boundary.
2. Source rule reconstruction count and source/validation image disjointness.
3. Immutable category-33 neckline task configuration.
4. Selected LoRA and frozen checkpoint presence and model-parameter contract.
5. Source-artifact SHA-256 values.
6. File-level SHA-256 evidence manifest integrity.
7. Claim boundaries in frozen release notes.

The tests use only the Python standard library and pytest. They do not import PyTorch,
Transformers, or PEFT and do not load a model, run inference, train, access raw Fashionpedia
images/annotations, or re-score validation.
