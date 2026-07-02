Fashionpedia source-task rule forensic audit v1

Purpose: identify why the simple final-confirmation eligibility predicate reconstructed 20,829 source pairs while the canonical v1b train/dev manifests contain 20,800 pairs.

This package does NOT open official validation annotations or images and does NOT load a model.

Expand the ZIP into the project repo root, then run:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\run_amazon07_fashionpedia_task_rule_forensics_v1.ps1

Read the resulting forensic JSON before touching official validation again.
