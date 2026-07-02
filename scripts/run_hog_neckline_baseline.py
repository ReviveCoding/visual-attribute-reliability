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
import skimage
from PIL import Image
from skimage.feature import hog
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

    missing_columns = sorted(required_columns - set(rows[0]))

    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {missing_columns}"
        )

    return rows


def crop_and_resize(
    record: dict[str, str],
    crop_config: dict,
) -> Image.Image:
    image_path = Path(record["image_path"])

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image missing for annotation_id={record['annotation_id']}: "
            f"{image_path}"
        )

    try:
        x = float(record["bbox_x"])
        y = float(record["bbox_y"])
        width = float(record["bbox_width"])
        height = float(record["bbox_height"])
    except ValueError as exc:
        raise ValueError(
            f"Invalid bbox values for annotation_id={record['annotation_id']}"
        ) from exc

    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            f"Non-positive bbox area for annotation_id={record['annotation_id']}"
        )

    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
        image_width, image_height = image.size

        pad_fraction = float(crop_config["padding_fraction"])
        pad_x = width * pad_fraction
        pad_y = height * pad_fraction

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
                f"Degenerate clamped crop for annotation_id={record['annotation_id']}; "
                f"bbox=({x}, {y}, {width}, {height}), "
                f"image_size=({image_width}, {image_height})"
            )

        crop = image.crop((left, top, right, bottom))

    resize_size = (
        int(crop_config["resize_width"]),
        int(crop_config["resize_height"]),
    )

    return crop.resize(
        resize_size,
        resample=Image.Resampling.BILINEAR,
    )


def extract_hog_vector(
    record: dict[str, str],
    config: dict,
) -> np.ndarray:
    crop = crop_and_resize(record, config["crop"])
    gray = np.asarray(
        crop.convert("L"),
        dtype=np.float32,
    ) / 255.0

    hog_config = config["hog"]

    vector = hog(
        gray,
        orientations=int(hog_config["orientations"]),
        pixels_per_cell=tuple(hog_config["pixels_per_cell"]),
        cells_per_block=tuple(hog_config["cells_per_block"]),
        block_norm=str(hog_config["block_norm"]),
        feature_vector=True,
    )

    return np.asarray(vector, dtype=np.float64)


def build_feature_matrix(
    records: list[dict[str, str]],
    config: dict,
    split_name: str,
) -> np.ndarray:
    if not records:
        raise RuntimeError(f"No records supplied for split={split_name}")

    print(f"Extracting HOG features for split={split_name}, n={len(records)}")

    first_vector = extract_hog_vector(records[0], config)
    features = np.empty(
        (len(records), first_vector.shape[0]),
        dtype=np.float64,
    )
    features[0] = first_vector

    progress_interval = max(250, len(records) // 20)

    for index, record in enumerate(records[1:], start=1):
        vector = extract_hog_vector(record, config)

        if vector.shape[0] != features.shape[1]:
            raise RuntimeError(
                f"Inconsistent HOG vector size for annotation_id="
                f"{record['annotation_id']}: "
                f"{vector.shape[0]} vs expected {features.shape[1]}"
            )

        features[index] = vector

        if (
            index % progress_interval == 0
            or index == len(records) - 1
        ):
            print(
                f"feature_progress split={split_name} "
                f"completed={index + 1}/{len(records)}"
            )

    return features


def top_label_ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bin_count: int,
) -> tuple[float, list[dict]]:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == targets).astype(np.float64)

    bins = np.linspace(0.0, 1.0, bin_count + 1)
    ece = 0.0
    rows = []

    for index in range(bin_count):
        lower = bins[index]
        upper = bins[index + 1]

        if index == bin_count - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = int(mask.sum())

        if count == 0:
            continue

        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        weight = float(count) / float(len(targets))
        ece += weight * abs(bin_accuracy - bin_confidence)

        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "mean_confidence": bin_confidence,
                "accuracy": bin_accuracy,
                "absolute_gap": abs(bin_accuracy - bin_confidence),
            }
        )

    return float(ece), rows


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--manifest-summary", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    start_time = time.perf_counter()

    train_manifest_path = Path(args.train_manifest).resolve()
    dev_manifest_path = Path(args.dev_manifest).resolve()
    manifest_summary_path = Path(args.manifest_summary).resolve()
    config_path = Path(args.config).resolve()
    run_dir = Path(args.run_dir).resolve()

    for path in (
        train_manifest_path,
        dev_manifest_path,
        manifest_summary_path,
        config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run directory: {run_dir}"
        )

    run_dir.mkdir(parents=True, exist_ok=False)

    manifest_summary = load_json(manifest_summary_path)
    config = load_json(config_path)

    if manifest_summary.get("status") != "PASS":
        raise RuntimeError(
            "Source manifest is not qualified: "
            f"{manifest_summary.get('status')!r}"
        )

    checks = manifest_summary.get("checks", {})

    if not checks.get("group_disjoint_pass", False):
        raise RuntimeError("Source manifest did not pass group-disjoint check.")

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
            f"Group leakage detected: {len(group_overlap)} overlapping image groups."
        )

    target_attribute_ids = [
        int(value)
        for value in config["target_attribute_ids"]
    ]

    target_attribute_set = set(target_attribute_ids)

    if len(target_attribute_set) != len(target_attribute_ids):
        raise ValueError("Target attribute IDs are not unique.")

    train_attribute_ids = {
        int(record["attribute_id"])
        for record in train_records
    }

    dev_attribute_ids = {
        int(record["attribute_id"])
        for record in dev_records
    }

    if train_attribute_ids != target_attribute_set:
        raise ValueError(
            "Train manifest label set does not match configured target labels. "
            f"Observed={sorted(train_attribute_ids)}"
        )

    if dev_attribute_ids != target_attribute_set:
        raise ValueError(
            "Dev manifest label set does not match configured target labels. "
            f"Observed={sorted(dev_attribute_ids)}"
        )

    attribute_names = {
        int(attribute_id): str(name)
        for attribute_id, name in config["target_attribute_names"].items()
    }

    class_index = {
        attribute_id: index
        for index, attribute_id in enumerate(target_attribute_ids)
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

    print("===== HOG NECKLINE BASELINE PRE-FLIGHT =====")
    print(f"TRAIN_SAMPLES: {len(train_records)}")
    print(f"DEV_SAMPLES: {len(dev_records)}")
    print(f"TRAIN_GROUPS: {len(train_groups)}")
    print(f"DEV_GROUPS: {len(dev_groups)}")
    print(f"GROUP_OVERLAP_COUNT: {len(group_overlap)}")
    print(f"TARGET_ATTRIBUTE_IDS: {target_attribute_ids}")

    X_train = build_feature_matrix(
        train_records,
        config,
        split_name="train",
    )

    X_dev = build_feature_matrix(
        dev_records,
        config,
        split_name="dev",
    )

    print(f"HOG_FEATURE_DIMENSION: {X_train.shape[1]}")

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
                    solver=str(classifier_config["solver"]),
                    C=float(classifier_config["C"]),
                    class_weight=str(classifier_config["class_weight"]),
                    max_iter=int(classifier_config["max_iter"]),
                    tol=float(classifier_config["tol"]),
                    random_state=int(classifier_config["random_state"]),
                ),
            ),
        ],
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        pipeline.fit(X_train, y_train)

    warning_rows = [
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in caught_warnings
    ]

    convergence_warning_count = sum(
        1
        for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    )

    probabilities = pipeline.predict_proba(X_dev)

    if probabilities.shape != (
        len(dev_records),
        len(target_attribute_ids),
    ):
        raise RuntimeError(
            "Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Probability matrix contains non-finite values.")

    predicted_indices = probabilities.argmax(axis=1)
    predicted_attribute_ids = [
        target_attribute_ids[int(index)]
        for index in predicted_indices
    ]

    accuracy = float(accuracy_score(y_dev, predicted_indices))
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

    precision, recall, f1, support = precision_recall_fscore_support(
        y_dev,
        predicted_indices,
        labels=list(range(len(target_attribute_ids))),
        zero_division=0,
    )

    negative_log_likelihood = float(
        log_loss(
            y_dev,
            probabilities,
            labels=list(range(len(target_attribute_ids))),
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
        labels=list(range(len(target_attribute_ids))),
    )

    per_class_rows = []

    for index, attribute_id in enumerate(target_attribute_ids):
        per_class_rows.append(
            {
                "attribute_id": attribute_id,
                "attribute_name": attribute_names[attribute_id],
                "train_count": int(train_class_counts[attribute_id]),
                "dev_count": int(dev_class_counts[attribute_id]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )

    prediction_rows = []

    for row_index, record in enumerate(dev_records):
        prediction_row = {
            "annotation_id": record["annotation_id"],
            "image_id": record["image_id"],
            "split_group_id": record["split_group_id"],
            "image_path": record["image_path"],
            "true_attribute_id": int(record["attribute_id"]),
            "true_attribute_name": record["attribute_name"],
            "predicted_attribute_id": int(
                predicted_attribute_ids[row_index]
            ),
            "predicted_attribute_name": attribute_names[
                predicted_attribute_ids[row_index]
            ],
            "confidence": float(probabilities[row_index].max()),
            "correct": bool(
                int(predicted_attribute_ids[row_index])
                == int(record["attribute_id"])
            ),
        }

        for class_index_value, attribute_id in enumerate(
            target_attribute_ids
        ):
            prediction_row[
                f"probability_attribute_{attribute_id}"
            ] = float(
                probabilities[row_index, class_index_value]
            )

        prediction_rows.append(prediction_row)

    model_path = run_dir / "hog_logreg_model.joblib"
    per_class_path = run_dir / "per_class_metrics.csv"
    predictions_path = run_dir / "dev_predictions.csv"
    confusion_path = run_dir / "confusion_matrix.csv"
    ece_path = run_dir / "ece_bins.json"
    warnings_path = run_dir / "training_warnings.json"
    summary_path = run_dir / "run_summary.json"
    config_copy_path = run_dir / "config.json"

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

    confusion_rows = []

    for true_index, true_attribute_id in enumerate(
        target_attribute_ids
    ):
        row = {
            "true_attribute_id": true_attribute_id,
            "true_attribute_name": attribute_names[true_attribute_id],
        }

        for predicted_index, predicted_attribute_id in enumerate(
            target_attribute_ids
        ):
            row[
                f"predicted_attribute_{predicted_attribute_id}"
            ] = int(confusion[true_index, predicted_index])

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
                "ece": ece,
                "bin_count": int(config["evaluation"]["ece_bins"]),
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
                "convergence_warning_count": convergence_warning_count,
                "warnings": warning_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    config_copy_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    classifier = pipeline.named_steps["classifier"]

    qualification_checks = {
        "source_manifest_pass": (
            manifest_summary.get("status") == "PASS"
        ),
        "source_group_disjoint_pass": bool(
            checks.get("group_disjoint_pass", False)
        ),
        "rechecked_group_disjoint_pass": len(group_overlap) == 0,
        "all_target_classes_in_train": (
            train_attribute_ids == target_attribute_set
        ),
        "all_target_classes_in_dev": (
            dev_attribute_ids == target_attribute_set
        ),
        "finite_probability_matrix": bool(
            np.isfinite(probabilities).all()
        ),
        "prediction_count_matches_dev": (
            len(prediction_rows) == len(dev_records)
        ),
        "no_convergence_warning": convergence_warning_count == 0,
    }

    status = (
        "PASS"
        if all(qualification_checks.values())
        else "REVIEW"
    )

    summary = {
        "status": status,
        "schema_version": "hog_neckline_baseline_run_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_role": config["run_role"],
        "task_name": config["task_name"],
        "selection_scope": "train_only_development",
        "source_artifacts": {
            "train_manifest": str(train_manifest_path),
            "dev_manifest": str(dev_manifest_path),
            "manifest_summary": str(manifest_summary_path),
            "train_manifest_sha256": sha256_file(train_manifest_path),
            "dev_manifest_sha256": sha256_file(dev_manifest_path),
            "manifest_summary_sha256": sha256_file(
                manifest_summary_path
            ),
            "config_sha256": sha256_file(config_path),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "scikit_image": skimage.__version__,
            "scikit_learn": sklearn.__version__,
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
        "feature_specification": {
            "crop": config["crop"],
            "hog": config["hog"],
            "hog_feature_dimension": int(X_train.shape[1]),
        },
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
            "negative_log_likelihood": negative_log_likelihood,
            "multiclass_brier": multiclass_brier,
            "top_label_ece": ece,
            "top_2_accuracy": top_2_accuracy,
        },
        "per_class_metrics": per_class_rows,
        "qualification_checks": qualification_checks,
        "warnings": {
            "warning_count": len(warning_rows),
            "convergence_warning_count": convergence_warning_count,
        },
        "artifacts": {
            "model": str(model_path),
            "per_class_metrics_csv": str(per_class_path),
            "dev_predictions_csv": str(predictions_path),
            "confusion_matrix_csv": str(confusion_path),
            "ece_bins_json": str(ece_path),
            "training_warnings_json": str(warnings_path),
            "config_copy_json": str(config_copy_path),
        },
        "elapsed_seconds": round(
            time.perf_counter() - start_time,
            3,
        ),
        "claim_boundary": config["claim_boundary"],
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("")
    print("===== HOG NECKLINE BASELINE SUMMARY =====")
    print(f"STATUS: {status}")
    print(f"TRAIN_SAMPLES: {len(train_records)}")
    print(f"DEV_SAMPLES: {len(dev_records)}")
    print(f"GROUP_OVERLAP_COUNT: {len(group_overlap)}")
    print(f"HOG_FEATURE_DIMENSION: {X_train.shape[1]}")
    print(f"ACCURACY: {accuracy:.6f}")
    print(f"BALANCED_ACCURACY: {balanced_accuracy:.6f}")
    print(f"MACRO_F1: {macro_f1:.6f}")
    print(f"WEIGHTED_F1: {weighted_f1:.6f}")
    print(f"NEGATIVE_LOG_LIKELIHOOD: {negative_log_likelihood:.6f}")
    print(f"MULTICLASS_BRIER: {multiclass_brier:.6f}")
    print(f"TOP_LABEL_ECE: {ece:.6f}")
    print(f"TOP_2_ACCURACY: {top_2_accuracy:.6f}")
    print(f"CONVERGENCE_WARNING_COUNT: {convergence_warning_count}")

    print("")
    print("===== PER-CLASS METRICS =====")
    print("ID   DEV_SUPPORT  PRECISION  RECALL  F1  ATTRIBUTE")

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
    print(f"MODEL: {model_path}")
    print(f"DEV_PREDICTIONS: {predictions_path}")

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()