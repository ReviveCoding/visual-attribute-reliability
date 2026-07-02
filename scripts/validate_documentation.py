"""Validate public documentation claims and links without model or dataset access."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_TEXT = {
    "README.md": [
        "# Visual Attribute Reliability",
        "+8.83 percentage points",
        "not a full Fashionpedia benchmark",
        "docs/EVIDENCE.md",
        "docs/CI_BOUNDARY.md",
        "docs/REPRODUCIBILITY.md",
        "docs/LOCAL_EVIDENCE_RELEASE.md",
        "docs/FORENSICS.md",
    ],
    "docs/EVIDENCE.md": [
        "20,800",
        "16,635",
        "4,165",
        "589,824",
        "654",
        "644",
        "0.668109",
        "0.579770",
        "93",
        "42",
        "164",
    ],
    "docs/CI_BOUNDARY.md": [
        "hosted static-evidence gate",
        "local full-evidence gate",
        "Do not represent a hosted CI pass",
    ],
    "docs/REPRODUCIBILITY.md": [
        "Python | 3.11.9",
        "NVIDIA GeForce RTX 4090 Laptop GPU",
        "2.11.0+cu128",
        "12.8",
    ],
    "docs/LOCAL_EVIDENCE_RELEASE.md": [
        "00d2e1b056c41373fff32d27471a72d9fe76bf94cde5c53598595fe145853b03",
        "125",
        "LOCAL_EVIDENCE_FROZEN",
    ],
    "docs/FORENSICS.md": [
        "20,829",
        "20,800",
        "29",
    ],
}

LEGACY_ROOT_DOCS = ("README.txt", "README_CI.md")


def main() -> int:
    failures: list[str] = []

    for relative_path, fragments in REQUIRED_TEXT.items():
        path = ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing required documentation file: {relative_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in text:
                failures.append(
                    f"documentation contract missing {fragment!r} in {relative_path}"
                )

    for relative_path in LEGACY_ROOT_DOCS:
        if (ROOT / relative_path).exists():
            failures.append(f"legacy root documentation still exists: {relative_path}")

    if failures:
        print("DOCUMENTATION_CONTRACT=FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("DOCUMENTATION_CONTRACT=PASS")
    print("NO_MODEL_RUNTIME_LOADED=True")
    print("NO_DATASET_ACCESS=True")
    print("NO_INFERENCE_PERFORMED=True")
    print("NO_TRAINING_PERFORMED=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
