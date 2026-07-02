from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize_scalar
import sklearn
import torch
from PIL import Image
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from transformers import AutoImageProcessor, AutoModel
import peft
import transformers


EPS = 1e-12
TARGET_IDS_DEFAULT = [182, 183, 185, 187, 189, 190, 200]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def required_columns(rows: list[dict[str, str]], required: set[str], artifact: Path) -> None:
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"{artifact} is missing required columns: {missing}")


def canonical_pair(annotation_id: Any, attribute_id: Any) -> tuple[int, int]:
    return int(annotation_id), int(attribute_id)


def target_hits(annotation: dict[str, Any], target_ids: set[int]) -> list[int]:
    raw = annotation.get("attribute_ids", [])
    return sorted({int(value) for value in raw if int(value) in target_ids})


def build_train_rule_pairs(
    annotation_payload: dict[str, Any],
    target_ids: set[int],
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for annotation in annotation_payload.get("annotations", []):
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
            continue
        hits = target_hits(annotation, target_ids)
        if len(hits) == 1:
            pairs.append(canonical_pair(annotation["id"], hits[0]))
    return pairs


def index_images(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    images: dict[int, dict[str, Any]] = {}
    for image in payload.get("images", []):
        image_id = int(image["id"])
        if image_id in images:
            raise RuntimeError(f"Duplicate image ID in annotation payload: {image_id}")
        images[image_id] = image
    return images


def index_file_names(image_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    duplicate_names: set[str] = set()
    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        key = path.name
        if key in mapping and mapping[key] != path:
            duplicate_names.add(key)
        else:
            mapping[key] = path
    if duplicate_names:
        preview = sorted(duplicate_names)[:5]
        raise RuntimeError(
            "Ambiguous basename resolution under validation image root; "
            f"examples={preview}"
        )
    return mapping


def resolve_image_path(
    image_root: Path,
    image_record: dict[str, Any],
    basename_index: dict[str, Path],
) -> Path:
    file_name = str(image_record.get("file_name", ""))
    if not file_name:
        raise RuntimeError(f"Image record has no file_name: {image_record}")
    direct = image_root / file_name
    if direct.is_file():
        return direct.resolve()
    basename = Path(file_name).name
    candidate = basename_index.get(basename)
    if candidate is None or not candidate.is_file():
        raise FileNotFoundError(
            f"Unable to resolve validation image file_name={file_name!r} "
            f"under root={image_root}"
        )
    return candidate.resolve()


def build_validation_rows(
    payload: dict[str, Any],
    image_root: Path,
    target_ids: set[int],
    target_names: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images = index_images(payload)
    basename_index = index_file_names(image_root)
    rows: list[dict[str, Any]] = []
    missing_image_rows: list[dict[str, Any]] = []
    excluded_multiple_target = 0
    excluded_no_target = 0
    excluded_invalid_bbox = 0

    for annotation in payload.get("annotations", []):
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            excluded_invalid_bbox += 1
            continue
        if float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
            excluded_invalid_bbox += 1
            continue
        hits = target_hits(annotation, target_ids)
        if len(hits) == 0:
            excluded_no_target += 1
            continue
        if len(hits) > 1:
            excluded_multiple_target += 1
            continue
        image_id = int(annotation["image_id"])
        image = images.get(image_id)
        if image is None:
            raise RuntimeError(
                f"Eligible validation annotation {annotation['id']} references missing image ID {image_id}"
            )
        try:
            image_path = resolve_image_path(image_root, image, basename_index)
        except FileNotFoundError as exc:
            missing_image_rows.append(
                {
                    "annotation_id": int(annotation["id"]),
                    "image_id": image_id,
                    "file_name": str(image.get("file_name", "")),
                    "error": str(exc),
                }
            )
            continue
        attribute_id = hits[0]
        rows.append(
            {
                "annotation_id": int(annotation["id"]),
                "image_id": image_id,
                "split_group_id": str(image_id),
                "image_path": str(image_path),
                "attribute_id": attribute_id,
                "attribute_name": target_names[attribute_id],
                "bbox_x": float(bbox[0]),
                "bbox_y": float(bbox[1]),
                "bbox_width": float(bbox[2]),
                "bbox_height": float(bbox[3]),
                "source_category_id": int(annotation.get("category_id", -1)),
                "source_image_file_name": str(image.get("file_name", "")),
            }
        )

    if missing_image_rows:
        preview = missing_image_rows[:3]
        raise FileNotFoundError(
            "Eligible validation samples have missing images. "
            f"count={len(missing_image_rows)} preview={preview}"
        )

    rows.sort(key=lambda row: (int(row["image_id"]), int(row["annotation_id"])))
    stats = {
        "annotation_count": len(payload.get("annotations", [])),
        "eligible_samples": len(rows),
        "eligible_images": len({row["image_id"] for row in rows}),
        "excluded_no_target": excluded_no_target,
        "excluded_multiple_target": excluded_multiple_target,
        "excluded_invalid_bbox": excluded_invalid_bbox,
        "validation_image_file_count": len(basename_index),
    }
    return rows, stats


def crop_from_record(record: dict[str, Any], padding_fraction: float) -> Image.Image:
    image_path = Path(str(record["image_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"Image missing: {image_path}")
    x = float(record["bbox_x"])
    y = float(record["bbox_y"])
    width = float(record["bbox_width"])
    height = float(record["bbox_height"])
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"Non-positive bbox for annotation_id={record['annotation_id']}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image_width, image_height = image.size
        pad_x = width * padding_fraction
        pad_y = height * padding_fraction
        left = max(0, int(math.floor(x - pad_x)))
        top = max(0, int(math.floor(y - pad_y)))
        right = min(image_width, int(math.ceil(x + width + pad_x)))
        bottom = min(image_height, int(math.ceil(y + height + pad_y)))
        if right <= left or bottom <= top:
            raise ValueError(f"Degenerate crop for annotation_id={record['annotation_id']}")
        return image.crop((left, top, right, bottom))


def make_inputs(
    records: list[dict[str, Any]],
    image_processor: Any,
    device: torch.device,
    padding_fraction: float,
) -> dict[str, torch.Tensor]:
    crops = [crop_from_record(record, padding_fraction) for record in records]
    processor_outputs = image_processor(images=crops, return_tensors="pt")
    return {
        name: value.to(device, non_blocking=True)
        for name, value in processor_outputs.items()
        if hasattr(value, "to")
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Non-finite probabilities.")
    return probabilities


def top_label_ece(probabilities: np.ndarray, targets: np.ndarray, bins: int) -> tuple[float, list[dict[str, Any]]]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        average_confidence = float(confidence[mask].mean())
        average_accuracy = float(correct[mask].mean())
        gap = abs(average_confidence - average_accuracy)
        ece += (count / len(targets)) * gap
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_confidence": average_confidence,
                "accuracy": average_accuracy,
                "absolute_gap": gap,
            }
        )
    return float(ece), rows


def metric_bundle(
    probabilities: np.ndarray,
    labels: np.ndarray,
    target_ids: list[int],
    target_names: dict[int, str],
    ece_bins: int,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = probabilities.argmax(axis=1)
    n_classes = len(target_ids)
    accuracy = float(accuracy_score(labels, predictions))
    balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, predictions, average="weighted", zero_division=0))
    nll = float(log_loss(labels, probabilities, labels=list(range(n_classes))))
    one_hot = np.eye(n_classes, dtype=np.float64)[labels]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    ece, ece_rows = top_label_ece(probabilities, labels, ece_bins)
    top_2 = np.argsort(-probabilities, axis=1)[:, :2]
    top2_accuracy = float(np.mean(np.any(top_2 == labels[:, None], axis=1)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(n_classes)),
        zero_division=0,
    )
    per_class = []
    for class_index, attribute_id in enumerate(target_ids):
        per_class.append(
            {
                "attribute_id": int(attribute_id),
                "attribute_name": target_names[int(attribute_id)],
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
            }
        )
    return (
        {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "negative_log_likelihood": nll,
            "multiclass_brier": brier,
            "top_label_ece": float(ece),
            "top_2_accuracy": top2_accuracy,
        },
        per_class,
        ece_rows,
    )


def probabilities_from_saved_dev_predictions(
    path: Path,
    target_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    rows = load_csv(path)
    required = {"true_attribute_id"} | {f"probability_attribute_{attr_id}" for attr_id in target_ids}
    required_columns(rows, required, path)
    target_to_index = {attribute_id: index for index, attribute_id in enumerate(target_ids)}
    labels = np.asarray([target_to_index[int(row["true_attribute_id"])] for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row[f"probability_attribute_{attr_id}"]) for attr_id in target_ids] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise RuntimeError(f"Invalid probabilities in {path}")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise RuntimeError(f"Non-positive probability sum in {path}")
    probabilities /= row_sums
    return probabilities, labels


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    base_logits = np.log(np.clip(probabilities, EPS, 1.0))
    before = float(log_loss(labels, softmax(base_logits), labels=list(range(probabilities.shape[1]))))

    def objective(temperature: float) -> float:
        if not np.isfinite(temperature) or temperature <= 0.0:
            return float("inf")
        calibrated = softmax(base_logits / float(temperature))
        return float(log_loss(labels, calibrated, labels=list(range(probabilities.shape[1]))))

    result = minimize_scalar(
        objective,
        bounds=(0.05, 5.0),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.x) or result.x <= 0.0:
        raise RuntimeError(f"Temperature optimization failed: {result}")
    temperature = float(result.x)
    after = objective(temperature)
    calibrated = softmax(base_logits / temperature)
    predictions_unchanged = bool(np.array_equal(probabilities.argmax(axis=1), calibrated.argmax(axis=1)))
    return {
        "method": "single_scalar_temperature_scaling",
        "optimization_scope": "selected_development_predictions_only",
        "temperature": temperature,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "development_nll_before": before,
        "development_nll_after": after,
        "development_argmax_unchanged": predictions_unchanged,
    }


def apply_temperature_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"Invalid temperature: {temperature}")
    return softmax(logits / float(temperature))


def predict_logits(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    records: list[dict[str, Any]],
    image_processor: Any,
    device: torch.device,
    padding_fraction: float,
    batch_size: int,
) -> np.ndarray:
    encoder.eval()
    head.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))
        inputs = make_inputs(records[start:end], image_processor, device, padding_fraction)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = encoder.get_image_features(**inputs)
                logits = head(features)
        chunk = logits.detach().float().cpu().numpy()
        if not np.isfinite(chunk).all():
            raise FloatingPointError(f"Non-finite logits for batch {start}:{end}")
        chunks.append(chunk)
        print(f"inference_progress completed={end}/{len(records)}", flush=True)
    logits = np.concatenate(chunks, axis=0)
    if logits.shape[0] != len(records):
        raise RuntimeError("Prediction count mismatch")
    return logits


def load_head(checkpoint_path: Path, input_dimension: int, class_count: int, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "head_state_dict" not in checkpoint:
        raise RuntimeError(f"Missing head_state_dict in {checkpoint_path}")
    head = torch.nn.Linear(input_dimension, class_count).to(device)
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    head.eval()
    return head


def count_lora_parameters(model: torch.nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        )
    )


def load_frozen_arm(
    model_id: str,
    cache_dir: Path,
    checkpoint_path: Path,
    input_dimension: int,
    class_count: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    model = AutoModel.from_pretrained(model_id, cache_dir=str(cache_dir))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = load_head(checkpoint_path, input_dimension, class_count, device)
    return model, head, {
        "model_class": model.__class__.__name__,
        "adapter_parameter_count": 0,
        "checkpoint_path": str(checkpoint_path),
    }


def load_lora_arm(
    model_id: str,
    cache_dir: Path,
    adapter_dir: Path,
    checkpoint_path: Path,
    input_dimension: int,
    class_count: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    base_model = AutoModel.from_pretrained(model_id, cache_dir=str(cache_dir))
    lora_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=False,
        autocast_adapter_dtype=False,
    )
    lora_model.to(device)
    lora_model.eval()
    for parameter in lora_model.parameters():
        parameter.requires_grad = False
    encoder = lora_model.get_base_model()
    head = load_head(checkpoint_path, input_dimension, class_count, device)
    return encoder, head, {
        "model_class": encoder.__class__.__name__,
        "adapter_model_class": lora_model.__class__.__name__,
        "adapter_parameter_count": count_lora_parameters(lora_model),
        "checkpoint_path": str(checkpoint_path),
        "adapter_directory": str(adapter_dir),
    }


def prediction_rows(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    target_ids: list[int],
    target_names: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_indices = raw_probabilities.argmax(axis=1)
    calibrated_indices = calibrated_probabilities.argmax(axis=1)
    for index, record in enumerate(records):
        raw_attribute_id = target_ids[int(raw_indices[index])]
        calibrated_attribute_id = target_ids[int(calibrated_indices[index])]
        row: dict[str, Any] = {
            "annotation_id": record["annotation_id"],
            "image_id": record["image_id"],
            "split_group_id": record["split_group_id"],
            "image_path": record["image_path"],
            "true_attribute_id": target_ids[int(labels[index])],
            "true_attribute_name": target_names[target_ids[int(labels[index])]],
            "raw_predicted_attribute_id": raw_attribute_id,
            "raw_predicted_attribute_name": target_names[raw_attribute_id],
            "raw_confidence": float(raw_probabilities[index].max()),
            "raw_correct": bool(raw_indices[index] == labels[index]),
            "calibrated_predicted_attribute_id": calibrated_attribute_id,
            "calibrated_predicted_attribute_name": target_names[calibrated_attribute_id],
            "calibrated_confidence": float(calibrated_probabilities[index].max()),
            "calibrated_correct": bool(calibrated_indices[index] == labels[index]),
        }
        for class_index, attribute_id in enumerate(target_ids):
            row[f"raw_probability_attribute_{attribute_id}"] = float(raw_probabilities[index, class_index])
            row[f"calibrated_probability_attribute_{attribute_id}"] = float(calibrated_probabilities[index, class_index])
        rows.append(row)
    return rows


def confirmation_arm(
    arm_name: str,
    records: list[dict[str, Any]],
    labels: np.ndarray,
    target_ids: list[int],
    target_names: dict[int, str],
    development_arm_summary: dict[str, Any],
    run_root: Path,
    model_id: str,
    cache_dir: Path,
    image_processor: Any,
    device: torch.device,
    padding_fraction: float,
    batch_size: int,
    ece_bins: int,
    output_dir: Path,
) -> dict[str, Any]:
    arm_dir = output_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=False)
    artifacts = development_arm_summary.get("artifacts", {})
    head_path = Path(str(artifacts.get("head_state_path", "")))
    dev_predictions_path = Path(str(artifacts.get("best_dev_predictions_csv", "")))
    if not head_path.is_file():
        raise FileNotFoundError(f"Selected head checkpoint is missing for {arm_name}: {head_path}")
    if not dev_predictions_path.is_file():
        raise FileNotFoundError(f"Selected development predictions are missing for {arm_name}: {dev_predictions_path}")

    input_dimension = 768
    class_count = len(target_ids)
    if arm_name == "frozen_head":
        encoder, head, load_info = load_frozen_arm(
            model_id, cache_dir, head_path, input_dimension, class_count, device
        )
    elif arm_name == "lora":
        adapter_dir = Path(str(artifacts.get("adapter_directory", "")))
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Selected LoRA adapter is missing: {adapter_dir}")
        encoder, head, load_info = load_lora_arm(
            model_id, cache_dir, adapter_dir, head_path, input_dimension, class_count, device
        )
    else:
        raise ValueError(f"Unknown arm: {arm_name}")

    dev_probabilities, dev_labels = probabilities_from_saved_dev_predictions(dev_predictions_path, target_ids)
    temperature_info = fit_temperature(dev_probabilities, dev_labels)
    temperature = float(temperature_info["temperature"])

    print(f"===== CONFIRMATION ARM START: {arm_name} =====", flush=True)
    raw_logits = predict_logits(
        encoder, head, records, image_processor, device, padding_fraction, batch_size
    )
    raw_probabilities = softmax(raw_logits)
    calibrated_probabilities = apply_temperature_from_logits(raw_logits, temperature)
    raw_metrics, per_class, raw_ece_rows = metric_bundle(
        raw_probabilities, labels, target_ids, target_names, ece_bins
    )
    calibrated_metrics, _, calibrated_ece_rows = metric_bundle(
        calibrated_probabilities, labels, target_ids, target_names, ece_bins
    )

    raw_prediction_indices = raw_probabilities.argmax(axis=1)
    calibrated_prediction_indices = calibrated_probabilities.argmax(axis=1)
    calibration_argmax_unchanged = bool(np.array_equal(raw_prediction_indices, calibrated_prediction_indices))
    if not calibration_argmax_unchanged:
        raise RuntimeError("Temperature scaling unexpectedly changed final argmax predictions.")

    rows = prediction_rows(
        records, labels, raw_probabilities, calibrated_probabilities, target_ids, target_names
    )
    prediction_fields = list(rows[0].keys())
    predictions_path = arm_dir / "confirmation_predictions.csv"
    per_class_path = arm_dir / "raw_per_class_metrics.csv"
    raw_ece_path = arm_dir / "raw_ece_bins.json"
    calibrated_ece_path = arm_dir / "calibrated_ece_bins.json"
    temperature_path = arm_dir / "temperature_scaling.json"
    logits_path = arm_dir / "raw_logits.npy"
    summary_path = arm_dir / "arm_summary.json"

    write_csv(predictions_path, prediction_fields, rows)
    write_csv(
        per_class_path,
        ["attribute_id", "attribute_name", "precision", "recall", "f1", "support"],
        per_class,
    )
    write_json(raw_ece_path, {"top_label_ece": raw_metrics["top_label_ece"], "bin_count": ece_bins, "bins": raw_ece_rows})
    write_json(calibrated_ece_path, {"top_label_ece": calibrated_metrics["top_label_ece"], "bin_count": ece_bins, "bins": calibrated_ece_rows})
    write_json(temperature_path, temperature_info)
    np.save(logits_path, raw_logits.astype(np.float32, copy=False))

    qualification_checks = {
        "selected_checkpoint_exists": head_path.is_file(),
        "development_predictions_exist": dev_predictions_path.is_file(),
        "finite_raw_logits": bool(np.isfinite(raw_logits).all()),
        "raw_prediction_count_matches_confirmation": len(rows) == len(records),
        "temperature_optimizer_success": bool(temperature_info["optimizer_success"]),
        "temperature_positive": temperature > 0.0,
        "temperature_preserves_argmax_on_development": bool(temperature_info["development_argmax_unchanged"]),
        "temperature_preserves_argmax_on_confirmation": calibration_argmax_unchanged,
        "finite_raw_metrics": all(np.isfinite(value) for value in raw_metrics.values()),
        "finite_calibrated_metrics": all(np.isfinite(value) for value in calibrated_metrics.values()),
    }
    if arm_name == "lora":
        qualification_checks["adapter_parameter_count_matches_development"] = (
            int(load_info["adapter_parameter_count"])
            == int(development_arm_summary["parameter_counts"]["lora_adapter"])
        )

    status = "PASS" if all(qualification_checks.values()) else "REVIEW"
    summary = {
        "status": status,
        "arm_name": arm_name,
        "evaluation_scope": "official_fashionpedia_validation_confirmation",
        "selected_development_epoch": int(development_arm_summary["selected_epoch"]),
        "development_selected_metrics": development_arm_summary["selected_metrics"],
        "model_load": load_info,
        "raw_metrics": raw_metrics,
        "temperature_calibrated_metrics": calibrated_metrics,
        "temperature_scaling": temperature_info,
        "qualification_checks": qualification_checks,
        "artifacts": {
            "confirmation_predictions_csv": str(predictions_path),
            "raw_per_class_metrics_csv": str(per_class_path),
            "raw_ece_bins_json": str(raw_ece_path),
            "calibrated_ece_bins_json": str(calibrated_ece_path),
            "temperature_scaling_json": str(temperature_path),
            "raw_logits_npy": str(logits_path),
        },
    }
    write_json(summary_path, summary)
    print(
        f"confirmation_summary arm={arm_name} status={status} "
        f"raw_macro_f1={raw_metrics['macro_f1']:.6f} "
        f"raw_ece={raw_metrics['top_label_ece']:.6f} "
        f"calibrated_ece={calibrated_metrics['top_label_ece']:.6f}",
        flush=True,
    )

    del encoder
    del head
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Final untouched Fashionpedia validation confirmation.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--source-train-manifest", required=True)
    parser.add_argument("--source-dev-manifest", required=True)
    parser.add_argument("--source-train-annotation", required=True)
    parser.add_argument("--validation-annotation", required=True)
    parser.add_argument("--validation-image-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    run_root = Path(args.run_root).resolve()
    train_manifest_path = Path(args.source_train_manifest).resolve()
    dev_manifest_path = Path(args.source_dev_manifest).resolve()
    train_annotation_path = Path(args.source_train_annotation).resolve()
    validation_annotation_path = Path(args.validation_annotation).resolve()
    validation_image_root = Path(args.validation_image_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    required_paths = [
        run_root / "run_summary.json",
        run_root / "config.json",
        train_manifest_path,
        dev_manifest_path,
        train_annotation_path,
        validation_annotation_path,
        validation_image_root,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path is missing: {path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active environment.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not report bf16 support.")

    run_summary = load_json(run_root / "run_summary.json")
    config = load_json(run_root / "config.json")
    if run_summary.get("status") != "PASS":
        raise RuntimeError("Development run summary is not PASS.")
    for arm_name in ("frozen_head_arm", "lora_arm"):
        if run_summary.get(arm_name, {}).get("status") != "PASS":
            raise RuntimeError(f"Development arm is not PASS: {arm_name}")

    target_ids = [int(value) for value in config.get("target_attribute_ids", TARGET_IDS_DEFAULT)]
    if target_ids != TARGET_IDS_DEFAULT:
        raise RuntimeError(f"Unexpected target class list: {target_ids}")
    target_set = set(target_ids)
    target_names = {int(key): str(value) for key, value in config["target_attribute_names"].items()}
    if set(target_names) != target_set:
        raise RuntimeError("Target attribute name map does not match target IDs.")
    padding_fraction = float(config["crop"]["padding_fraction"])
    ece_bins = int(config["evaluation"]["ece_bins"])
    batch_size = int(config["training"]["physical_batch_size"])
    model_id = str(config["model"]["model_id"])

    train_manifest = load_csv(train_manifest_path)
    dev_manifest = load_csv(dev_manifest_path)
    source_required = {"annotation_id", "image_id", "attribute_id", "split_group_id"}
    required_columns(train_manifest, source_required, train_manifest_path)
    required_columns(dev_manifest, source_required, dev_manifest_path)
    source_rows = train_manifest + dev_manifest
    source_pairs = [canonical_pair(row["annotation_id"], row["attribute_id"]) for row in source_rows]
    if len(source_pairs) != len(set(source_pairs)):
        raise RuntimeError("Source train/dev manifests contain duplicate annotation/attribute pairs.")

    train_payload = load_json(train_annotation_path)
    reconstructed_pairs = build_train_rule_pairs(train_payload, target_set)
    reconstructed_pair_set = set(reconstructed_pairs)
    source_pair_set = set(source_pairs)
    train_rule_reproduction_pass = source_pair_set == reconstructed_pair_set
    if not train_rule_reproduction_pass:
        missing = sorted(source_pair_set - reconstructed_pair_set)[:10]
        unexpected = sorted(reconstructed_pair_set - source_pair_set)[:10]
        raise RuntimeError(
            "Validation eligibility rule could not reproduce the source train/dev task exactly. "
            f"source_count={len(source_pair_set)} reconstructed_count={len(reconstructed_pair_set)} "
            f"missing_examples={missing} unexpected_examples={unexpected}"
        )

    validation_payload = load_json(validation_annotation_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_dir = output_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=False)
    validation_rows, val_stats = build_validation_rows(
        validation_payload,
        validation_image_root,
        target_set,
        target_names,
    )
    if not validation_rows:
        raise RuntimeError("No eligible official validation samples were produced.")
    confirmation_ids = {int(row["image_id"]) for row in validation_rows}
    source_ids = {int(row["image_id"]) for row in source_rows}
    image_overlap = source_ids & confirmation_ids
    if image_overlap:
        raise RuntimeError(
            "Official validation image IDs overlap source train/dev image IDs. "
            f"count={len(image_overlap)} examples={sorted(image_overlap)[:10]}"
        )
    class_counts = Counter(int(row["attribute_id"]) for row in validation_rows)
    missing_classes = sorted(target_set - set(class_counts))
    if missing_classes:
        raise RuntimeError(f"Official validation lacks target classes: {missing_classes}")
    if len({canonical_pair(row["annotation_id"], row["attribute_id"]) for row in validation_rows}) != len(validation_rows):
        raise RuntimeError("Official validation manifest contains duplicate annotation/attribute pairs.")

    manifest_path = manifest_dir / "official_validation_manifest.csv"
    write_csv(
        manifest_path,
        list(validation_rows[0].keys()),
        validation_rows,
    )
    manifest_summary = {
        "status": "PASS",
        "schema_version": "fashionpedia_neckline_official_validation_manifest_v1",
        "eligibility_rule": "An annotation is eligible iff its positive target-neckline attribute intersection has exactly one member and its bbox has positive width and height.",
        "source_train_rule_reproduction": {
            "pass": train_rule_reproduction_pass,
            "source_train_dev_pairs": len(source_pair_set),
            "reconstructed_train_pairs": len(reconstructed_pair_set),
        },
        "official_validation": {
            **val_stats,
            "class_counts": {str(attribute_id): int(class_counts[attribute_id]) for attribute_id in target_ids},
            "source_train_dev_image_id_overlap_count": len(image_overlap),
            "all_images_resolved": True,
        },
        "source_artifacts": {
            "development_run_summary": str(run_root / "run_summary.json"),
            "development_config": str(run_root / "config.json"),
            "source_train_manifest": str(train_manifest_path),
            "source_dev_manifest": str(dev_manifest_path),
            "source_train_annotation": str(train_annotation_path),
            "validation_annotation": str(validation_annotation_path),
            "validation_image_root": str(validation_image_root),
            "sha256": {
                "development_run_summary": sha256_file(run_root / "run_summary.json"),
                "development_config": sha256_file(run_root / "config.json"),
                "source_train_manifest": sha256_file(train_manifest_path),
                "source_dev_manifest": sha256_file(dev_manifest_path),
                "source_train_annotation": sha256_file(train_annotation_path),
                "validation_annotation": sha256_file(validation_annotation_path),
            },
        },
    }
    write_json(manifest_dir / "official_validation_manifest_summary.json", manifest_summary)

    print("===== FINAL CONFIRMATION MANIFEST AUDIT =====", flush=True)
    print("STATUS: PASS", flush=True)
    print(f"SOURCE_TASK_REPRODUCTION_PAIRS: {len(source_pair_set)}", flush=True)
    print(f"OFFICIAL_VALIDATION_SAMPLES: {len(validation_rows)}", flush=True)
    print(f"OFFICIAL_VALIDATION_IMAGES: {len(confirmation_ids)}", flush=True)
    print(f"IMAGE_ID_OVERLAP_COUNT: {len(image_overlap)}", flush=True)
    print(f"OFFICIAL_VALIDATION_MANIFEST: {manifest_path}", flush=True)

    labels = np.asarray(
        [{attribute_id: index for index, attribute_id in enumerate(target_ids)}[int(row["attribute_id"])] for row in validation_rows],
        dtype=np.int64,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    image_processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=str(cache_dir), use_fast=False)

    frozen_summary = confirmation_arm(
        "frozen_head",
        validation_rows,
        labels,
        target_ids,
        target_names,
        run_summary["frozen_head_arm"],
        run_root,
        model_id,
        cache_dir,
        image_processor,
        device,
        padding_fraction,
        batch_size,
        ece_bins,
        output_dir,
    )
    lora_summary = confirmation_arm(
        "lora",
        validation_rows,
        labels,
        target_ids,
        target_names,
        run_summary["lora_arm"],
        run_root,
        model_id,
        cache_dir,
        image_processor,
        device,
        padding_fraction,
        batch_size,
        ece_bins,
        output_dir,
    )

    comparisons: dict[str, dict[str, float]] = {}
    for scope, frozen_metrics, lora_metrics in (
        ("raw", frozen_summary["raw_metrics"], lora_summary["raw_metrics"]),
        ("temperature_calibrated", frozen_summary["temperature_calibrated_metrics"], lora_summary["temperature_calibrated_metrics"]),
    ):
        comparisons[scope] = {
            metric: float(lora_metrics[metric] - frozen_metrics[metric])
            for metric in frozen_metrics
        }

    qualification_checks = {
        "development_run_pass": run_summary.get("status") == "PASS",
        "source_task_rule_reproduced": train_rule_reproduction_pass,
        "official_validation_no_source_image_overlap": len(image_overlap) == 0,
        "official_validation_all_target_classes_present": len(missing_classes) == 0,
        "frozen_confirmation_pass": frozen_summary["status"] == "PASS",
        "lora_confirmation_pass": lora_summary["status"] == "PASS",
        "lora_adapter_parameter_count_matches_development": lora_summary["qualification_checks"].get("adapter_parameter_count_matches_development", False),
    }
    overall_status = "PASS" if all(qualification_checks.values()) else "REVIEW"
    summary = {
        "status": overall_status,
        "schema_version": "fashionpedia_neckline_final_confirmation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "untouched_official_fashionpedia_validation",
        "selection_scope": "fixed_from_train_only_development_before_validation_access",
        "model_id": model_id,
        "data": {
            "official_validation_samples": len(validation_rows),
            "official_validation_images": len(confirmation_ids),
            "class_counts": {str(attribute_id): int(class_counts[attribute_id]) for attribute_id in target_ids},
        },
        "manifest_summary": manifest_summary,
        "frozen_head": frozen_summary,
        "lora": lora_summary,
        "lora_minus_frozen_head": comparisons,
        "qualification_checks": qualification_checks,
        "gpu_peak_memory_allocated_mib": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "claim_boundary": "External confirmation is limited to Fashionpedia validation instances eligible under the source task rule. No post-validation checkpoint, hyperparameter, crop, rank, or target-module changes were performed by this runner.",
    }
    summary_path = output_dir / "final_confirmation_summary.json"
    write_json(summary_path, summary)

    print("===== FINAL FASHIONPEDIA CONFIRMATION SUMMARY =====", flush=True)
    print(f"STATUS: {overall_status}", flush=True)
    print(f"FROZEN_RAW_MACRO_F1: {frozen_summary['raw_metrics']['macro_f1']:.6f}", flush=True)
    print(f"LORA_RAW_MACRO_F1: {lora_summary['raw_metrics']['macro_f1']:.6f}", flush=True)
    print(f"LORA_MINUS_FROZEN_RAW_MACRO_F1: {comparisons['raw']['macro_f1']:.6f}", flush=True)
    print(f"FROZEN_RAW_ECE: {frozen_summary['raw_metrics']['top_label_ece']:.6f}", flush=True)
    print(f"LORA_RAW_ECE: {lora_summary['raw_metrics']['top_label_ece']:.6f}", flush=True)
    print(f"FROZEN_CALIBRATED_ECE: {frozen_summary['temperature_calibrated_metrics']['top_label_ece']:.6f}", flush=True)
    print(f"LORA_CALIBRATED_ECE: {lora_summary['temperature_calibrated_metrics']['top_label_ece']:.6f}", flush=True)
    print(f"SUMMARY_JSON: {summary_path}", flush=True)
    if overall_status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
