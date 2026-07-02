from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import PIL
import sklearn
import torch
import transformers
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"Manifest contains no rows: {path}")

    required_columns = {
        "annotation_id",
        "image_id",
        "split_group_id",
        "image_path",
        "attribute_id",
        "attribute_name",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
    }

    missing = sorted(required_columns - set(rows[0]))

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {missing}"
        )

    return rows


def crop_from_record(
    record: dict[str, str],
    padding_fraction: float,
) -> Image.Image:
    image_path = Path(record["image_path"])

    if not image_path.is_file():
        raise FileNotFoundError(
            "Missing image for annotation_id="
            f"{record['annotation_id']}: {image_path}"
        )

    x = float(record["bbox_x"])
    y = float(record["bbox_y"])
    width = float(record["bbox_width"])
    height = float(record["bbox_height"])

    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            "Non-positive bbox for annotation_id="
            f"{record['annotation_id']}"
        )

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image_width, image_height = image.size

        pad_x = width * padding_fraction
        pad_y = height * padding_fraction

        left = max(0, int(math.floor(x - pad_x)))
        top = max(0, int(math.floor(y - pad_y)))
        right = min(
            image_width,
            int(math.ceil(x + width + pad_x)),
        )
        bottom = min(
            image_height,
            int(math.ceil(y + height + pad_y)),
        )

        if right <= left or bottom <= top:
            raise ValueError(
                "Degenerate crop for annotation_id="
                f"{record['annotation_id']}"
            )

        return image.crop((left, top, right, bottom))


def encode_one_batch(
    records: list[dict[str, str]],
    image_processor,
    model,
    device: torch.device,
    padding_fraction: float,
) -> np.ndarray:
    crops = [
        crop_from_record(record, padding_fraction)
        for record in records
    ]

    processor_outputs = image_processor(
        images=crops,
        return_tensors="pt",
    )

    model_inputs = {
        name: value.to(device)
        for name, value in processor_outputs.items()
        if hasattr(value, "to")
    }

    with torch.inference_mode():
        image_features = model.get_image_features(
            **model_inputs,
        )

    torch.cuda.synchronize(device)

    features = (
        image_features.detach()
        .float()
        .cpu()
        .numpy()
    )

    if features.ndim != 2:
        raise RuntimeError(
            f"Expected rank-2 embeddings, observed {features.shape}"
        )

    if features.shape[0] != len(records):
        raise RuntimeError(
            "Embedding batch size mismatch: "
            f"{features.shape[0]} vs {len(records)}"
        )

    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite values in frozen embeddings.")

    return np.asarray(features, dtype=np.float32)


def extract_embeddings(
    records: list[dict[str, str]],
    split_name: str,
    image_processor,
    model,
    device: torch.device,
    padding_fraction: float,
    initial_batch_size: int,
    allow_oom_batch_halving: bool,
) -> tuple[np.ndarray, dict]:
    if not records:
        raise RuntimeError(f"No records supplied for split={split_name}")

    active_batch_size = initial_batch_size
    index = 0
    chunks: list[np.ndarray] = []
    fallback_events: list[dict] = []
    next_progress = max(500, len(records) // 20)

    print(
        f"embedding_extraction_start split={split_name} "
        f"n={len(records)} initial_batch_size={initial_batch_size}",
        flush=True,
    )

    while index < len(records):
        batch_end = min(
            index + active_batch_size,
            len(records),
        )

        batch_records = records[index:batch_end]

        try:
            batch_embeddings = encode_one_batch(
                records=batch_records,
                image_processor=image_processor,
                model=model,
                device=device,
                padding_fraction=padding_fraction,
            )
        except RuntimeError as exc:
            message = str(exc).lower()

            is_oom = (
                "out of memory" in message
                or "cuda error: out of memory" in message
            )

            if (
                is_oom
                and allow_oom_batch_halving
                and active_batch_size > 1
            ):
                old_size = active_batch_size
                active_batch_size = max(
                    1,
                    active_batch_size // 2,
                )

                torch.cuda.empty_cache()

                fallback_events.append(
                    {
                        "at_index": index,
                        "old_batch_size": old_size,
                        "new_batch_size": active_batch_size,
                        "error": str(exc),
                    }
                )

                print(
                    "embedding_batch_oom_fallback "
                    f"split={split_name} index={index} "
                    f"old_batch_size={old_size} "
                    f"new_batch_size={active_batch_size}",
                    flush=True,
                )

                continue

            raise

        chunks.append(batch_embeddings)
        index = batch_end

        if index >= next_progress or index == len(records):
            print(
                f"embedding_progress split={split_name} "
                f"completed={index}/{len(records)} "
                f"batch_size={active_batch_size}",
                flush=True,
            )

            next_progress += max(500, len(records) // 20)

    embeddings = np.concatenate(chunks, axis=0)

    if embeddings.shape[0] != len(records):
        raise RuntimeError(
            "Final embedding count mismatch for split="
            f"{split_name}: {embeddings.shape[0]} vs {len(records)}"
        )

    return embeddings, {
        "split_name": split_name,
        "sample_count": len(records),
        "initial_batch_size": initial_batch_size,
        "final_batch_size": active_batch_size,
        "fallback_events": fallback_events,
        "fallback_count": len(fallback_events),
        "embedding_dimension": int(embeddings.shape[1]),
    }


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True,
    )

    if not np.isfinite(norms).all():
        raise RuntimeError("Non-finite embedding norms.")

    if np.any(norms <= 0.0):
        raise RuntimeError("Zero-norm frozen embedding found.")

    normalized = features / norms

    if not np.isfinite(normalized).all():
        raise RuntimeError("Non-finite normalized embedding found.")

    return np.asarray(normalized, dtype=np.float32)


def top_label_ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bin_count: int,
) -> tuple[float, list[dict]]:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == targets).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bin_count + 1)
    ece = 0.0
    rows: list[dict] = []

    for index in range(bin_count):
        lower = edges[index]
        upper = edges[index + 1]

        if index == bin_count - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        weight = float(count) / float(len(targets))
        gap = abs(mean_confidence - accuracy)

        ece += weight * gap

        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )

    return float(ece), rows


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--manifest-summary", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()

    started = time.perf_counter()

    train_manifest_path = Path(args.train_manifest).resolve()
    dev_manifest_path = Path(args.dev_manifest).resolve()
    manifest_summary_path = Path(args.manifest_summary).resolve()
    config_path = Path(args.config).resolve()
    run_dir = Path(args.run_dir).resolve()
    cache_root = Path(args.cache_dir).resolve()

    for path in (
        train_manifest_path,
        dev_manifest_path,
        manifest_summary_path,
        config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact missing: {path}"
            )

    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite run directory: {run_dir}"
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest_summary = load_json(manifest_summary_path)
    config = load_json(config_path)

    if manifest_summary.get("status") != "PASS":
        raise RuntimeError(
            "Source manifest summary is not PASS: "
            f"{manifest_summary.get('status')!r}"
        )

    if not manifest_summary.get(
        "checks",
        {},
    ).get(
        "group_disjoint_pass",
        False,
    ):
        raise RuntimeError(
            "Source manifest did not pass group-disjoint validation."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in the active environment."
        )

    device = torch.device("cuda")

    train_records = load_manifest(train_manifest_path)
    dev_records = load_manifest(dev_manifest_path)

    train_groups = {
        record["split_group_id"]
        for record in train_records
    }

    dev_groups = {
        record["split_group_id"]
        for record in dev_records
    }

    group_overlap = train_groups & dev_groups

    if group_overlap:
        raise RuntimeError(
            "Group overlap detected between train and dev: "
            f"{len(group_overlap)}"
        )

    target_attribute_ids = [
        int(value)
        for value in config["target_attribute_ids"]
    ]

    target_attribute_set = set(target_attribute_ids)

    if len(target_attribute_set) != len(target_attribute_ids):
        raise RuntimeError(
            "Configured target attribute IDs are not unique."
        )

    target_attribute_names = {
        int(attribute_id): str(name)
        for attribute_id, name in config[
            "target_attribute_names"
        ].items()
    }

    train_attribute_ids = {
        int(record["attribute_id"])
        for record in train_records
    }

    dev_attribute_ids = {
        int(record["attribute_id"])
        for record in dev_records
    }

    if train_attribute_ids != target_attribute_set:
        raise RuntimeError(
            "Train label set mismatch: "
            f"{sorted(train_attribute_ids)}"
        )

    if dev_attribute_ids != target_attribute_set:
        raise RuntimeError(
            "Dev label set mismatch: "
            f"{sorted(dev_attribute_ids)}"
        )

    class_index = {
        attribute_id: index
        for index, attribute_id in enumerate(
            target_attribute_ids
        )
    }

    y_train = np.asarray(
        [
            class_index[int(record["attribute_id"])]
            for record in train_records
        ],
        dtype=np.int64,
    )

    y_dev = np.asarray(
        [
            class_index[int(record["attribute_id"])]
            for record in dev_records
        ],
        dtype=np.int64,
    )

    train_class_counts = Counter(
        int(record["attribute_id"])
        for record in train_records
    )

    dev_class_counts = Counter(
        int(record["attribute_id"])
        for record in dev_records
    )

    padding_fraction = float(
        config["crop"]["padding_fraction"]
    )

    initial_batch_size = int(
        config["embedding_extraction"][
            "initial_batch_size"
        ]
    )

    allow_oom_batch_halving = bool(
        config["embedding_extraction"][
            "allow_oom_batch_halving"
        ]
    )

    print("===== SIGLIP2 FROZEN NECKLINE PRE-FLIGHT =====")
    print(f"TRAIN_SAMPLES: {len(train_records)}")
    print(f"DEV_SAMPLES: {len(dev_records)}")
    print(f"TRAIN_GROUPS: {len(train_groups)}")
    print(f"DEV_GROUPS: {len(dev_groups)}")
    print(f"GROUP_OVERLAP_COUNT: {len(group_overlap)}")
    print(f"DEVICE: {device}")
    print(f"MODEL_ID: {config['model']['model_id']}")
    print(f"INITIAL_BATCH_SIZE: {initial_batch_size}")
    print(
        "TARGET_ATTRIBUTE_IDS: "
        f"{target_attribute_ids}"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    image_processor = AutoImageProcessor.from_pretrained(
        config["model"]["model_id"],
        cache_dir=str(cache_root),
        use_fast=False,
    )

    model = AutoModel.from_pretrained(
        config["model"]["model_id"],
        cache_dir=str(cache_root),
    )

    model.eval()
    model.to(device)

    train_raw, train_extract_info = extract_embeddings(
        records=train_records,
        split_name="train",
        image_processor=image_processor,
        model=model,
        device=device,
        padding_fraction=padding_fraction,
        initial_batch_size=initial_batch_size,
        allow_oom_batch_halving=allow_oom_batch_halving,
    )

    dev_raw, dev_extract_info = extract_embeddings(
        records=dev_records,
        split_name="dev",
        image_processor=image_processor,
        model=model,
        device=device,
        padding_fraction=padding_fraction,
        initial_batch_size=initial_batch_size,
        allow_oom_batch_halving=allow_oom_batch_halving,
    )

    train_features = l2_normalize(train_raw)
    dev_features = l2_normalize(dev_raw)

    if train_features.shape[1] != dev_features.shape[1]:
        raise RuntimeError(
            "Train/dev embedding dimension mismatch: "
            f"{train_features.shape[1]} vs "
            f"{dev_features.shape[1]}"
        )

    classifier_config = config["classifier"]

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(
                    with_mean=True,
                    with_std=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver=str(
                        classifier_config["solver"]
                    ),
                    C=float(classifier_config["C"]),
                    class_weight=str(
                        classifier_config["class_weight"]
                    ),
                    max_iter=int(
                        classifier_config["max_iter"]
                    ),
                    tol=float(classifier_config["tol"]),
                    random_state=int(
                        classifier_config["random_state"]
                    ),
                ),
            ),
        ],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(train_features, y_train)

    warning_rows = [
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in caught
    ]

    convergence_warning_count = sum(
        1
        for warning in caught
        if issubclass(
            warning.category,
            ConvergenceWarning,
        )
    )

    classifier = pipeline.named_steps["classifier"]

    expected_classes = np.arange(
        len(target_attribute_ids),
        dtype=np.int64,
    )

    if not np.array_equal(
        classifier.classes_,
        expected_classes,
    ):
        raise RuntimeError(
            "Unexpected classifier class order: "
            f"{classifier.classes_.tolist()}"
        )

    probabilities = pipeline.predict_proba(dev_features)

    expected_probability_shape = (
        len(dev_records),
        len(target_attribute_ids),
    )

    if probabilities.shape != expected_probability_shape:
        raise RuntimeError(
            "Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Probability matrix contains non-finite values."
        )

    predicted_indices = probabilities.argmax(axis=1)

    accuracy = float(
        accuracy_score(y_dev, predicted_indices)
    )

    balanced_accuracy = float(
        balanced_accuracy_score(y_dev, predicted_indices)
    )

    macro_f1 = float(
        f1_score(
            y_dev,
            predicted_indices,
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = float(
        f1_score(
            y_dev,
            predicted_indices,
            average="weighted",
            zero_division=0,
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_dev,
            predicted_indices,
            labels=list(
                range(len(target_attribute_ids))
            ),
            zero_division=0,
        )
    )

    negative_log_likelihood = float(
        log_loss(
            y_dev,
            probabilities,
            labels=list(
                range(len(target_attribute_ids))
            ),
        )
    )

    one_hot = np.eye(
        len(target_attribute_ids),
        dtype=np.float64,
    )[y_dev]

    multiclass_brier = float(
        np.mean(
            np.sum(
                (probabilities - one_hot) ** 2,
                axis=1,
            )
        )
    )

    ece, ece_bins = top_label_ece(
        probabilities,
        y_dev,
        int(config["evaluation"]["ece_bins"]),
    )

    top_2 = np.argsort(
        -probabilities,
        axis=1,
    )[:, :2]

    top_2_accuracy = float(
        np.mean(
            np.any(
                top_2 == y_dev[:, None],
                axis=1,
            )
        )
    )

    confusion = confusion_matrix(
        y_dev,
        predicted_indices,
        labels=list(
            range(len(target_attribute_ids))
        ),
    )

    per_class_rows: list[dict] = []

    for index, attribute_id in enumerate(
        target_attribute_ids
    ):
        per_class_rows.append(
            {
                "attribute_id": attribute_id,
                "attribute_name": target_attribute_names[
                    attribute_id
                ],
                "train_count": int(
                    train_class_counts[attribute_id]
                ),
                "dev_count": int(
                    dev_class_counts[attribute_id]
                ),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )

    prediction_rows: list[dict] = []

    for index, record in enumerate(dev_records):
        predicted_attribute_id = target_attribute_ids[
            int(predicted_indices[index])
        ]

        row = {
            "annotation_id": record["annotation_id"],
            "image_id": record["image_id"],
            "split_group_id": record["split_group_id"],
            "image_path": record["image_path"],
            "true_attribute_id": int(
                record["attribute_id"]
            ),
            "true_attribute_name": record[
                "attribute_name"
            ],
            "predicted_attribute_id": int(
                predicted_attribute_id
            ),
            "predicted_attribute_name": (
                target_attribute_names[
                    predicted_attribute_id
                ]
            ),
            "confidence": float(
                probabilities[index].max()
            ),
            "correct": bool(
                predicted_attribute_id
                == int(record["attribute_id"])
            ),
        }

        for class_position, attribute_id in enumerate(
            target_attribute_ids
        ):
            row[
                f"probability_attribute_{attribute_id}"
            ] = float(
                probabilities[index, class_position]
            )

        prediction_rows.append(row)

    train_embeddings_path = run_dir / "train_embeddings_l2.npy"
    dev_embeddings_path = run_dir / "dev_embeddings_l2.npy"
    train_labels_path = run_dir / "train_labels.npy"
    dev_labels_path = run_dir / "dev_labels.npy"
    model_path = run_dir / "siglip2_frozen_logreg.joblib"
    per_class_path = run_dir / "per_class_metrics.csv"
    predictions_path = run_dir / "dev_predictions.csv"
    confusion_path = run_dir / "confusion_matrix.csv"
    ece_path = run_dir / "ece_bins.json"
    warnings_path = run_dir / "training_warnings.json"
    extraction_path = run_dir / "embedding_extraction.json"
    config_copy_path = run_dir / "config.json"
    summary_path = run_dir / "run_summary.json"

    np.save(train_embeddings_path, train_features)
    np.save(dev_embeddings_path, dev_features)
    np.save(train_labels_path, y_train)
    np.save(dev_labels_path, y_dev)

    joblib.dump(
        pipeline,
        model_path,
        compress=3,
    )

    write_csv(
        per_class_path,
        [
            "attribute_id",
            "attribute_name",
            "train_count",
            "dev_count",
            "precision",
            "recall",
            "f1",
            "support",
        ],
        per_class_rows,
    )

    prediction_fields = [
        "annotation_id",
        "image_id",
        "split_group_id",
        "image_path",
        "true_attribute_id",
        "true_attribute_name",
        "predicted_attribute_id",
        "predicted_attribute_name",
        "confidence",
        "correct",
    ] + [
        f"probability_attribute_{attribute_id}"
        for attribute_id in target_attribute_ids
    ]

    write_csv(
        predictions_path,
        prediction_fields,
        prediction_rows,
    )

    confusion_rows: list[dict] = []

    for true_index, true_attribute_id in enumerate(
        target_attribute_ids
    ):
        row = {
            "true_attribute_id": true_attribute_id,
            "true_attribute_name": target_attribute_names[
                true_attribute_id
            ],
        }

        for predicted_index, predicted_attribute_id in enumerate(
            target_attribute_ids
        ):
            row[
                f"predicted_attribute_{predicted_attribute_id}"
            ] = int(
                confusion[
                    true_index,
                    predicted_index,
                ]
            )

        confusion_rows.append(row)

    write_csv(
        confusion_path,
        [
            "true_attribute_id",
            "true_attribute_name",
        ] + [
            f"predicted_attribute_{attribute_id}"
            for attribute_id in target_attribute_ids
        ],
        confusion_rows,
    )

    ece_path.write_text(
        json.dumps(
            {
                "top_label_ece": ece,
                "bin_count": int(
                    config["evaluation"]["ece_bins"]
                ),
                "bins": ece_bins,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    warnings_path.write_text(
        json.dumps(
            {
                "warning_count": len(warning_rows),
                "convergence_warning_count": (
                    convergence_warning_count
                ),
                "warnings": warning_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    extraction = {
        "train": train_extract_info,
        "dev": dev_extract_info,
        "embedding_dimension": int(
            train_features.shape[1]
        ),
        "embedding_normalization": (
            config["model"]["embedding_normalization"]
        ),
        "gpu_peak_memory_allocated_mib": round(
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2),
            2,
        ),
    }

    extraction_path.write_text(
        json.dumps(extraction, indent=2),
        encoding="utf-8",
    )

    config_copy_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    qualification_checks = {
        "source_manifest_pass": (
            manifest_summary.get("status") == "PASS"
        ),
        "source_group_disjoint_pass": bool(
            manifest_summary.get(
                "checks",
                {},
            ).get(
                "group_disjoint_pass",
                False,
            )
        ),
        "rechecked_group_disjoint_pass": (
            len(group_overlap) == 0
        ),
        "cuda_available": bool(
            torch.cuda.is_available()
        ),
        "all_target_classes_in_train": (
            train_attribute_ids
            == target_attribute_set
        ),
        "all_target_classes_in_dev": (
            dev_attribute_ids
            == target_attribute_set
        ),
        "train_embedding_count_matches": (
            train_features.shape[0]
            == len(train_records)
        ),
        "dev_embedding_count_matches": (
            dev_features.shape[0]
            == len(dev_records)
        ),
        "matching_embedding_dimension": (
            train_features.shape[1]
            == dev_features.shape[1]
        ),
        "finite_probability_matrix": bool(
            np.isfinite(probabilities).all()
        ),
        "prediction_count_matches_dev": (
            len(prediction_rows)
            == len(dev_records)
        ),
        "no_convergence_warning": (
            convergence_warning_count == 0
        ),
    }

    status = (
        "PASS"
        if all(qualification_checks.values())
        else "REVIEW"
    )

    summary = {
        "status": status,
        "schema_version": (
            "siglip2_frozen_neckline_baseline_run_v1"
        ),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "run_role": config["run_role"],
        "task_name": config["task_name"],
        "selection_scope": "train_only_development",
        "source_artifacts": {
            "train_manifest": str(train_manifest_path),
            "dev_manifest": str(dev_manifest_path),
            "manifest_summary": str(
                manifest_summary_path
            ),
            "train_manifest_sha256": sha256_file(
                train_manifest_path
            ),
            "dev_manifest_sha256": sha256_file(
                dev_manifest_path
            ),
            "manifest_summary_sha256": sha256_file(
                manifest_summary_path
            ),
            "config_sha256": sha256_file(
                config_path
            ),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "scikit_learn": sklearn.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "model": {
            "model_id": config["model"]["model_id"],
            "model_class": model.__class__.__name__,
            "image_processor_class": (
                image_processor.__class__.__name__
            ),
            "model_commit_hash": getattr(
                model.config,
                "_commit_hash",
                None,
            ),
            "model_parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                )
            ),
            "encoder_trainable": False,
        },
        "data": {
            "train_samples": len(train_records),
            "dev_samples": len(dev_records),
            "train_image_groups": len(train_groups),
            "dev_image_groups": len(dev_groups),
            "group_overlap_count": len(group_overlap),
            "target_attribute_ids": target_attribute_ids,
            "train_class_counts": {
                str(attribute_id): int(
                    train_class_counts[attribute_id]
                )
                for attribute_id in target_attribute_ids
            },
            "dev_class_counts": {
                str(attribute_id): int(
                    dev_class_counts[attribute_id]
                )
                for attribute_id in target_attribute_ids
            },
        },
        "embedding_extraction": extraction,
        "classifier_specification": {
            **config["classifier"],
            "observed_max_iterations": int(
                np.max(classifier.n_iter_)
            ),
        },
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "negative_log_likelihood": (
                negative_log_likelihood
            ),
            "multiclass_brier": multiclass_brier,
            "top_label_ece": ece,
            "top_2_accuracy": top_2_accuracy,
        },
        "per_class_metrics": per_class_rows,
        "qualification_checks": qualification_checks,
        "warnings": {
            "warning_count": len(warning_rows),
            "convergence_warning_count": (
                convergence_warning_count
            ),
        },
        "artifacts": {
            "train_embeddings_l2_npy": str(
                train_embeddings_path
            ),
            "dev_embeddings_l2_npy": str(
                dev_embeddings_path
            ),
            "train_labels_npy": str(train_labels_path),
            "dev_labels_npy": str(dev_labels_path),
            "model_joblib": str(model_path),
            "per_class_metrics_csv": str(
                per_class_path
            ),
            "dev_predictions_csv": str(
                predictions_path
            ),
            "confusion_matrix_csv": str(
                confusion_path
            ),
            "ece_bins_json": str(ece_path),
            "training_warnings_json": str(
                warnings_path
            ),
            "embedding_extraction_json": str(
                extraction_path
            ),
            "config_copy_json": str(config_copy_path),
        },
        "elapsed_seconds": round(
            time.perf_counter() - started,
            3,
        ),
        "claim_boundary": config["claim_boundary"],
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("")
    print("===== SIGLIP2 FROZEN NECKLINE BASELINE SUMMARY =====")
    print(f"STATUS: {status}")
    print(f"TRAIN_SAMPLES: {len(train_records)}")
    print(f"DEV_SAMPLES: {len(dev_records)}")
    print(f"GROUP_OVERLAP_COUNT: {len(group_overlap)}")
    print(
        "EMBEDDING_DIMENSION: "
        f"{train_features.shape[1]}"
    )
    print(f"ACCURACY: {accuracy:.6f}")
    print(
        f"BALANCED_ACCURACY: {balanced_accuracy:.6f}"
    )
    print(f"MACRO_F1: {macro_f1:.6f}")
    print(f"WEIGHTED_F1: {weighted_f1:.6f}")
    print(
        "NEGATIVE_LOG_LIKELIHOOD: "
        f"{negative_log_likelihood:.6f}"
    )
    print(
        f"MULTICLASS_BRIER: {multiclass_brier:.6f}"
    )
    print(f"TOP_LABEL_ECE: {ece:.6f}")
    print(f"TOP_2_ACCURACY: {top_2_accuracy:.6f}")
    print(
        "CONVERGENCE_WARNING_COUNT: "
        f"{convergence_warning_count}"
    )
    print(
        "GPU_PEAK_MEMORY_ALLOCATED_MIB: "
        f"{extraction['gpu_peak_memory_allocated_mib']:.2f}"
    )

    print("")
    print("===== PER-CLASS METRICS =====")
    print(
        "ID   DEV_SUPPORT  PRECISION  RECALL  F1  ATTRIBUTE"
    )

    for row in per_class_rows:
        print(
            f'{row["attribute_id"]:>3}  '
            f'{row["support"]:>11}  '
            f'{row["precision"]:>9.4f}  '
            f'{row["recall"]:>6.4f}  '
            f'{row["f1"]:>6.4f}  '
            f'{row["attribute_name"]}'
        )

    print("")
    print(f"RUN_SUMMARY_JSON: {summary_path}")
    print(f"EMBEDDING_EXTRACTION_JSON: {extraction_path}")
    print(f"MODEL: {model_path}")
    print(f"DEV_PREDICTIONS: {predictions_path}")

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()