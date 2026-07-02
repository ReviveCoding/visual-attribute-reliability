"""Hosted-CI contracts for a lightweight Fashionpedia evidence fixture.

These tests deliberately validate a tracked, static subset of the frozen evidence.
They do not load model checkpoints, import ML runtimes, access raw Fashionpedia data,
run inference, train, re-score validation, or select checkpoints. The separate local
release-evidence suite remains the full checkpoint/hash gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

FIXTURE_ROOT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "fashionpedia_release_ci_fixture_v1"
)

TARGET_ATTRIBUTE_IDS = [182, 183, 185, 187, 189, 190, 200]


def _read_json(name: str) -> dict:
    path = FIXTURE_ROOT / name
    assert path.is_file(), f"Fixture file is missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(actual: float, expected: float, label: str) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (
        f"{label} mismatch: expected={expected!r}, actual={actual!r}"
    )


def test_fixture_manifest_hashes_and_paths() -> None:
    manifest = _read_json("fixture_manifest.json")
    assert manifest["schema_version"] == "fashionpedia_ci_fixture_manifest_v1"
    assert manifest["fixture_role"] == "hosted_ci_static_evidence_contract_fixture"
    assert manifest["file_count"] == len(manifest["files"])
    assert manifest["full_release_bundle_required_for"] == [
        "checkpoint_presence_contract",
        "source_artifact_hash_contract",
        "release_file_manifest_hash_contract",
    ]

    for row in manifest["files"]:
        path = FIXTURE_ROOT / row["relative_path"]
        assert path.is_file(), f"Manifest-listed fixture file is missing: {path}"
        assert path.stat().st_size == row["bytes"]
        assert _sha256(path) == row["sha256"]


def test_immutable_category_scoped_task_definition() -> None:
    config = _read_json("fashionpedia_neckline_attribute_v1.json")
    assert config["schema_version"] == "fashionpedia_neckline_attribute_v1"
    assert config["target_category"] == {"id": 33, "expected_name": "neckline"}
    assert config["target_attribute_ids"] == TARGET_ATTRIBUTE_IDS
    assert config["sample_rule"] == {
        "include_only_target_category": True,
        "require_exactly_one_target_attribute": True,
        "require_positive_bbox_area": True,
    }


def test_source_reconstruction_and_image_disjointness_contract() -> None:
    summary = _read_json("official_validation_manifest_summary.json")
    assert summary["status"] == "PASS"
    assert summary["source_train_rule_reproduction"] == {
        "pass": True,
        "source_train_dev_pairs": 20800,
    }
    official = summary["official_validation"]
    assert official["source_train_dev_image_id_overlap_count"] == 0
    assert official["eligible_instances"] == 654
    assert official["eligible_images"] == 644


def test_final_confirmation_metric_contract() -> None:
    summary = _read_json("final_confirmation_summary.json")
    assert summary["status"] == "PASS"
    assert summary["evaluation_scope"] == "untouched_official_fashionpedia_validation"
    assert summary["selection_scope"] == "fixed_from_train_only_development_before_validation_access"

    data = summary["data"]
    assert data["target_category_id"] == 33
    assert data["target_category_name"] == "neckline"
    assert data["official_validation_samples"] == 654
    assert data["official_validation_images"] == 644
    assert data["source_train_dev_image_id_overlap_count"] == 0

    assert summary["frozen_head"]["selected_development_epoch"] == 6
    assert summary["lora"]["selected_development_epoch"] == 5
    _assert_close(summary["frozen_head"]["raw_metrics"]["macro_f1"], 0.5797703254640079, "frozen Macro-F1")
    _assert_close(summary["lora"]["raw_metrics"]["macro_f1"], 0.6681089306597886, "LoRA Macro-F1")
    _assert_close(summary["lora_minus_frozen_head"]["raw"]["macro_f1"], 0.08833860519578074, "LoRA uplift")
    _assert_close(summary["frozen_head"]["raw_metrics"]["top_label_ece"], 0.08395556030105743, "frozen raw ECE")
    _assert_close(summary["lora"]["raw_metrics"]["top_label_ece"], 0.06523039754742146, "LoRA raw ECE")


def test_claim_boundaries_are_preserved() -> None:
    notes_path = FIXTURE_ROOT / "RELEASE_NOTES.md"
    assert notes_path.is_file()
    notes = notes_path.read_text(encoding="utf-8")
    assert "+8.83 percentage points" in notes
    assert "not a full Fashionpedia benchmark" in notes
    assert "no post-calibration LoRA-superiority claim" in notes
    assert "Full checkpoint-presence" in notes


@pytest.mark.parametrize("module_name", ["torch", "transformers", "peft"])
def test_fixture_suite_does_not_import_model_runtime(module_name: str) -> None:
    # This is a declarative guard: the module source stays stdlib + pytest only.
    source = Path(__file__).read_text(encoding="utf-8")
    assert f"import {module_name}" not in source
    assert f"from {module_name}" not in source
