"""Read-only evidence-contract tests for the Fashionpedia local evidence freeze.

These tests deliberately use only Python's standard library plus pytest. They do
not import torch, transformers, peft, or project training runners; they never
load a model, access raw Fashionpedia data, run inference, or re-score validation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest

RELEASE_NAME = "fashionpedia-neckline-attribute-reliability-v1.0.0-local-evidence"


def _repo_root() -> Path:
    configured = os.environ.get("AMAZON07_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _release_root() -> Path:
    configured = os.environ.get("AMAZON07_RELEASE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return _repo_root() / "dist" / RELEASE_NAME


@pytest.fixture(scope="session")
def release_root() -> Path:
    root = _release_root()
    assert root.is_dir(), f"Evidence release directory is missing: {root}"
    return root


@pytest.fixture(scope="session")
def release_manifest(release_root: Path) -> dict:
    path = release_root / "release_evidence_manifest.json"
    assert path.is_file(), f"Release evidence manifest is missing: {path}"
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


def test_release_manifest_semantic_contract(release_manifest: dict) -> None:
    assert release_manifest["status"] == "PASS"
    assert (
        release_manifest["schema_version"]
        == "fashionpedia_neckline_attribute_reliability_local_evidence_freeze_v1"
    )
    assert release_manifest["release_status"] == "LOCAL_EVIDENCE_FROZEN"
    assert release_manifest["no_training_or_inference_executed"] is True

    official = release_manifest["official_confirmation"]
    assert official["evaluation_scope"] == "untouched_official_fashionpedia_validation"
    assert (
        official["selection_scope"]
        == "fixed_from_train_only_development_before_validation_access"
    )
    assert official["target_category_id"] == 33
    assert official["target_category_name"] == "neckline"
    assert official["official_validation_samples"] == 654
    assert official["official_validation_images"] == 644
    assert official["source_train_dev_image_overlap_count"] == 0
    assert official["frozen_selected_development_epoch"] == 6
    assert official["lora_selected_development_epoch"] == 5

    _assert_close(official["frozen_raw_macro_f1"], 0.5797703254640079, "frozen Macro-F1")
    _assert_close(official["lora_raw_macro_f1"], 0.6681089306597886, "LoRA Macro-F1")
    _assert_close(
        official["lora_minus_frozen_raw_macro_f1"],
        0.08833860519578074,
        "LoRA minus frozen Macro-F1",
    )
    _assert_close(official["frozen_raw_ece"], 0.08395556030105743, "frozen raw ECE")
    _assert_close(official["lora_raw_ece"], 0.06523039754742146, "LoRA raw ECE")


def test_final_confirmation_matches_release_manifest(
    release_root: Path,
    release_manifest: dict,
) -> None:
    summary_path = (
        release_root
        / "evidence"
        / "runs"
        / "fashionpedia_neckline_final_confirmation_v2"
        / "final_confirmation_summary.json"
    )
    assert summary_path.is_file(), f"Final confirmation summary is missing: {summary_path}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "PASS"
    assert summary["evaluation_scope"] == "untouched_official_fashionpedia_validation"
    assert (
        summary["selection_scope"]
        == "fixed_from_train_only_development_before_validation_access"
    )
    assert summary["data"]["target_category_id"] == 33
    assert summary["data"]["target_category_name"] == "neckline"
    assert summary["data"]["official_validation_samples"] == 654
    assert summary["data"]["official_validation_images"] == 644

    qualification_values = list(summary["qualification_checks"].values())
    assert qualification_values
    assert all(bool(value) for value in qualification_values)

    official = release_manifest["official_confirmation"]
    _assert_close(
        summary["frozen_head"]["raw_metrics"]["macro_f1"],
        official["frozen_raw_macro_f1"],
        "frozen summary to release manifest Macro-F1",
    )
    _assert_close(
        summary["lora"]["raw_metrics"]["macro_f1"],
        official["lora_raw_macro_f1"],
        "LoRA summary to release manifest Macro-F1",
    )
    _assert_close(
        summary["lora_minus_frozen_head"]["raw"]["macro_f1"],
        official["lora_minus_frozen_raw_macro_f1"],
        "uplift summary to release manifest Macro-F1",
    )


def test_source_task_reproduction_and_disjoint_validation(release_root: Path) -> None:
    path = (
        release_root
        / "evidence"
        / "runs"
        / "fashionpedia_neckline_final_confirmation_v2"
        / "manifest"
        / "official_validation_manifest_summary.json"
    )
    assert path.is_file(), f"Official validation manifest summary is missing: {path}"
    summary = json.loads(path.read_text(encoding="utf-8"))

    assert summary["status"] == "PASS"
    reconstruction = summary["source_train_rule_reproduction"]
    assert reconstruction["pass"] is True
    assert reconstruction["source_train_dev_pairs"] == 20800
    assert summary["official_validation"]["source_train_dev_image_id_overlap_count"] == 0


def test_immutable_neckline_task_config(release_root: Path) -> None:
    path = (
        release_root
        / "source_snapshot"
        / "configs"
        / "fashionpedia_neckline_attribute_v1.json"
    )
    assert path.is_file(), f"Task config is missing: {path}"
    config = json.loads(path.read_text(encoding="utf-8"))

    target = config["target_category"]
    assert target["id"] == 33
    assert target["expected_name"] == "neckline"

    sample_rule = config["sample_rule"]
    assert sample_rule["include_only_target_category"] is True
    assert sample_rule["require_exactly_one_target_attribute"] is True
    assert sample_rule["require_positive_bbox_area"] is True
    assert sample_rule["exclude_zero_target_attribute_instances"] is True
    assert sample_rule["exclude_multi_target_attribute_instances"] is True

    target_attribute_ids = [
        int(attribute["id"])
        for attribute in config["target_attributes"]
    ]
    assert target_attribute_ids == [182, 183, 185, 187, 189, 190, 200]

def test_lora_checkpoint_and_parameter_contract(
    release_root: Path,
    release_manifest: dict,
) -> None:
    development_root = (
        release_root
        / "evidence"
        / "runs"
        / "siglip2_lora_matched_head_development_v2_bf16"
    )
    adapter_root = development_root / "lora" / "best_lora_adapter"
    assert adapter_root.is_dir(), f"LoRA adapter directory is missing: {adapter_root}"
    assert (adapter_root / "adapter_config.json").is_file()

    adapter_weights = [
        candidate
        for candidate in (adapter_root / "adapter_model.safetensors", adapter_root / "adapter_model.bin")
        if candidate.is_file()
    ]
    assert len(adapter_weights) == 1
    assert (development_root / "lora" / "best_head_state.pt").is_file()
    assert (development_root / "frozen_head" / "best_head_state.pt").is_file()

    model = release_manifest["model"]
    assert model["base_model_id"] == "google/siglip2-base-patch16-224"
    assert model["lora_target_module_count"] == 48
    assert model["lora_adapter_trainable_parameters"] == 589824
    assert model["matched_head_trainable_parameters"] == 5383


def test_source_artifact_hashes_match_release_content(
    release_root: Path,
    release_manifest: dict,
) -> None:
    staged_paths = {
        "forensics": release_root / "evidence" / "forensics" / "fashionpedia_neckline_source_rule_forensics_v1.json",
        "task_audit": release_root / "evidence" / "task_audit" / "fashionpedia_neckline_train_only_audit_v1.json",
        "task_config": release_root / "source_snapshot" / "configs" / "fashionpedia_neckline_attribute_v1.json",
        "development_summary": (
            release_root
            / "evidence"
            / "runs"
            / "siglip2_lora_matched_head_development_v2_bf16"
            / "run_summary.json"
        ),
        "final_confirmation_summary": (
            release_root
            / "evidence"
            / "runs"
            / "fashionpedia_neckline_final_confirmation_v2"
            / "final_confirmation_summary.json"
        ),
        "official_validation_manifest_summary": (
            release_root
            / "evidence"
            / "runs"
            / "fashionpedia_neckline_final_confirmation_v2"
            / "manifest"
            / "official_validation_manifest_summary.json"
        ),
        "lora_adapter_config": (
            release_root
            / "evidence"
            / "runs"
            / "siglip2_lora_matched_head_development_v2_bf16"
            / "lora"
            / "best_lora_adapter"
            / "adapter_config.json"
        ),
        "lora_head_checkpoint": (
            release_root
            / "evidence"
            / "runs"
            / "siglip2_lora_matched_head_development_v2_bf16"
            / "lora"
            / "best_head_state.pt"
        ),
        "frozen_head_checkpoint": (
            release_root
            / "evidence"
            / "runs"
            / "siglip2_lora_matched_head_development_v2_bf16"
            / "frozen_head"
            / "best_head_state.pt"
        ),
    }

    for artifact in release_manifest["source_artifact_hashes"]:
        label = artifact["label"]
        if label == "lora_adapter_weights":
            candidates = [
                release_root
                / "evidence"
                / "runs"
                / "siglip2_lora_matched_head_development_v2_bf16"
                / "lora"
                / "best_lora_adapter"
                / "adapter_model.safetensors",
                release_root
                / "evidence"
                / "runs"
                / "siglip2_lora_matched_head_development_v2_bf16"
                / "lora"
                / "best_lora_adapter"
                / "adapter_model.bin",
            ]
            staged_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            assert staged_path is not None
        else:
            assert label in staged_paths, f"Unmapped source-artifact label: {label}"
            staged_path = staged_paths[label]

        assert staged_path.is_file(), f"Missing staged source artifact for {label}: {staged_path}"
        assert staged_path.stat().st_size == artifact["bytes"]
        assert _sha256(staged_path) == artifact["sha256"]


def test_release_file_manifest_hashes(release_root: Path) -> None:
    path = release_root / "release_file_manifest.json"
    assert path.is_file(), f"Release file manifest is missing: {path}"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "sha256_file_manifest_v1"
    assert manifest["root"] == RELEASE_NAME
    files = manifest["files"]
    assert manifest["file_count"] == len(files)
    assert files

    for row in files:
        relative = Path(*row["relative_path"].split("/"))
        file_path = release_root / relative
        assert file_path.is_file(), f"Manifest-listed file is absent: {relative}"
        assert file_path.stat().st_size == row["bytes"]
        assert _sha256(file_path) == row["sha256"]


def test_release_notes_preserve_claim_boundaries(release_root: Path) -> None:
    path = release_root / "RELEASE_NOTES.md"
    assert path.is_file(), f"Release notes are missing: {path}"
    notes = path.read_text(encoding="utf-8")

    assert "LOCAL_EVIDENCE_FROZEN" in notes
    assert "+8.83 percentage points" in notes
    assert "not a full Fashionpedia benchmark" in notes
    assert "no post-calibration LoRA-superiority claim" in notes
