from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


RAW_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "negative_log_likelihood",
    "multiclass_brier",
    "top_label_ece",
    "top_2_accuracy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected Boolean string, got {value!r}")


def find_artifact(summary: dict[str, Any], arm: str, key: str) -> Path:
    candidate = summary.get(arm, {}).get("artifacts", {}).get(key)
    if not candidate:
        raise KeyError(f"Missing final-confirmation artifact for {arm}.{key}")
    path = Path(str(candidate)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing referenced artifact: {path}")
    return path


def crop_image(image_path: Path, bbox: dict[str, str], padding: float) -> Image.Image:
    with Image.open(image_path) as raw:
        image = raw.convert("RGB")
        width, height = image.size
        x = float(bbox["bbox_x"])
        y = float(bbox["bbox_y"])
        box_width = float(bbox["bbox_width"])
        box_height = float(bbox["bbox_height"])
        left = max(0, int(math.floor(x - box_width * padding)))
        top = max(0, int(math.floor(y - box_height * padding)))
        right = min(width, int(math.ceil(x + box_width + box_width * padding)))
        bottom = min(height, int(math.ceil(y + box_height + box_height * padding)))
        if right <= left or bottom <= top:
            raise ValueError(f"Degenerate crop for {image_path}")
        return image.crop((left, top, right, bottom))


def plot_per_class_f1(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [str(row["attribute_name"]) for row in rows]
    frozen = np.asarray([float(row["frozen_f1"]) for row in rows])
    lora = np.asarray([float(row["lora_f1"]) for row in rows])
    y = np.arange(len(labels))
    height = 0.37
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(y - height / 2, frozen, height, label="Frozen")
    ax.barh(y + height / 2, lora, height, label="LoRA")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Official validation F1")
    ax.set_title("Per-class F1 on official Fashionpedia validation")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_confusion(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted neckline")
    ax.set_ylabel("True neckline")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=38, ha="right")
    ax.set_yticklabels(labels)
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_reliability(
    frozen_bins: list[dict[str, Any]],
    lora_bins: list[dict[str, Any]],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    for name, rows in (("Frozen raw", frozen_bins), ("LoRA raw", lora_bins)):
        x = [float(row["mean_confidence"]) for row in rows]
        y = [float(row["accuracy"]) for row in rows]
        counts = np.asarray([int(row["count"]) for row in rows])
        sizes = 20 + 1.6 * np.sqrt(counts)
        ax.plot(x, y, marker="o", label=name)
        ax.scatter(x, y, s=sizes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean top-label confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Raw top-label reliability on official validation")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def make_contact_sheet(
    title: str,
    examples: list[dict[str, Any]],
    manifest_by_annotation: dict[str, dict[str, str]],
    output: Path,
    padding: float,
) -> list[dict[str, Any]]:
    tile_width = 300
    tile_height = 315
    columns = 3
    rows = max(1, math.ceil(max(1, len(examples)) / columns))
    header_height = 44
    sheet = Image.new("RGB", (columns * tile_width, header_height + rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((10, 12), title, fill="black", font=font)
    index_rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        col = index % columns
        row = index // columns
        x0 = col * tile_width
        y0 = header_height + row * tile_height
        record = manifest_by_annotation.get(str(example["annotation_id"]))
        if record is None:
            continue
        source = Path(record["image_path"])
        try:
            crop = crop_image(source, record, padding)
            crop.thumbnail((tile_width - 16, 220))
            x = x0 + (tile_width - crop.width) // 2
            y = y0 + 5
            sheet.paste(crop, (x, y))
            image_status = "resolved"
        except Exception as exc:  # noqa: BLE001
            draw.rectangle((x0 + 5, y0 + 5, x0 + tile_width - 5, y0 + 220), outline="black")
            draw.text((x0 + 10, y0 + 15), f"crop unavailable: {type(exc).__name__}", fill="black", font=font)
            image_status = f"unavailable:{type(exc).__name__}"
        lines = [
            f"true: {example['true_attribute_name']}",
            f"frozen: {example['frozen_predicted_attribute_name']} ({float(example['frozen_confidence']):.2f})",
            f"lora: {example['lora_predicted_attribute_name']} ({float(example['lora_confidence']):.2f})",
            f"ann: {example['annotation_id']} image: {example['image_id']}",
        ]
        for line_index, line in enumerate(lines):
            draw.text((x0 + 8, y0 + 230 + 18 * line_index), line, fill="black", font=font)
        index_rows.append(
            {
                "contact_sheet": output.name,
                "selection_group": title,
                "image_status": image_status,
                **example,
            }
        )
    sheet.save(output)
    return index_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only post-validation analysis for Fashionpedia neckline confirmation."
    )
    parser.add_argument("--confirmation-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples-per-group", type=int, default=6)
    args = parser.parse_args()

    confirmation_summary_path = Path(args.confirmation_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not confirmation_summary_path.is_file():
        raise FileNotFoundError(f"Missing confirmation summary: {confirmation_summary_path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis directory: {output_dir}")

    summary = read_json(confirmation_summary_path)
    if summary.get("status") != "PASS":
        raise RuntimeError(f"Final confirmation status is not PASS: {summary.get('status')!r}")
    if summary.get("evaluation_scope") != "untouched_official_fashionpedia_validation":
        raise RuntimeError("Unexpected confirmation scope.")

    output_dir.mkdir(parents=True, exist_ok=False)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    crops_dir = output_dir / "representative_crops"
    figures_dir.mkdir()
    tables_dir.mkdir()
    crops_dir.mkdir()

    frozen_predictions_path = find_artifact(summary, "frozen_head", "confirmation_predictions_csv")
    lora_predictions_path = find_artifact(summary, "lora", "confirmation_predictions_csv")
    frozen_per_class_path = find_artifact(summary, "frozen_head", "raw_per_class_metrics_csv")
    lora_per_class_path = find_artifact(summary, "lora", "raw_per_class_metrics_csv")
    frozen_ece_path = find_artifact(summary, "frozen_head", "raw_ece_bins_json")
    lora_ece_path = find_artifact(summary, "lora", "raw_ece_bins_json")

    validation_manifest_path = confirmation_summary_path.parent / "manifest" / "official_validation_manifest.csv"
    if not validation_manifest_path.is_file():
        raise FileNotFoundError(f"Missing official validation manifest: {validation_manifest_path}")

    frozen_predictions = read_csv(frozen_predictions_path)
    lora_predictions = read_csv(lora_predictions_path)
    frozen_per_class = read_csv(frozen_per_class_path)
    lora_per_class = read_csv(lora_per_class_path)
    validation_manifest = read_csv(validation_manifest_path)
    frozen_ece = read_json(frozen_ece_path).get("bins", [])
    lora_ece = read_json(lora_ece_path).get("bins", [])

    if len(frozen_predictions) != len(lora_predictions):
        raise RuntimeError("Frozen and LoRA prediction counts differ.")

    frozen_by_annotation = {row["annotation_id"]: row for row in frozen_predictions}
    lora_by_annotation = {row["annotation_id"]: row for row in lora_predictions}
    manifest_by_annotation = {row["annotation_id"]: row for row in validation_manifest}
    if set(frozen_by_annotation) != set(lora_by_annotation):
        raise RuntimeError("Frozen and LoRA annotation IDs differ.")
    if set(frozen_by_annotation) != set(manifest_by_annotation):
        raise RuntimeError("Prediction and validation-manifest annotation IDs differ.")

    target_ids = [int(value) for value in summary["data"]["class_counts"].keys()]
    # Preserve fixed original task order from prediction probability headers.
    probability_headers = [key for key in frozen_predictions[0] if key.startswith("raw_probability_attribute_")]
    target_ids = [int(key.rsplit("_", 1)[1]) for key in probability_headers]
    target_names = {int(row["attribute_id"]): row["attribute_name"] for row in frozen_per_class}
    labels = [target_names[attribute_id] for attribute_id in target_ids]
    index_by_id = {attribute_id: index for index, attribute_id in enumerate(target_ids)}

    frozen_matrix = np.zeros((len(target_ids), len(target_ids)), dtype=int)
    lora_matrix = np.zeros_like(frozen_matrix)
    transitions: Counter[str] = Counter()
    combined_rows: list[dict[str, Any]] = []
    for annotation_id in sorted(frozen_by_annotation, key=lambda x: int(x)):
        frozen = frozen_by_annotation[annotation_id]
        lora = lora_by_annotation[annotation_id]
        if frozen["true_attribute_id"] != lora["true_attribute_id"]:
            raise RuntimeError(f"True-label mismatch for annotation={annotation_id}")
        true_id = int(frozen["true_attribute_id"])
        frozen_id = int(frozen["raw_predicted_attribute_id"])
        lora_id = int(lora["raw_predicted_attribute_id"])
        frozen_correct = as_bool(frozen["raw_correct"])
        lora_correct = as_bool(lora["raw_correct"])
        frozen_matrix[index_by_id[true_id], index_by_id[frozen_id]] += 1
        lora_matrix[index_by_id[true_id], index_by_id[lora_id]] += 1
        transition = (
            "both_correct" if frozen_correct and lora_correct else
            "lora_corrected" if (not frozen_correct and lora_correct) else
            "lora_regressed" if (frozen_correct and not lora_correct) else
            "both_wrong"
        )
        transitions[transition] += 1
        combined_rows.append(
            {
                "annotation_id": annotation_id,
                "image_id": frozen["image_id"],
                "image_path": frozen["image_path"],
                "true_attribute_id": true_id,
                "true_attribute_name": frozen["true_attribute_name"],
                "frozen_predicted_attribute_id": frozen_id,
                "frozen_predicted_attribute_name": frozen["raw_predicted_attribute_name"],
                "frozen_confidence": float(frozen["raw_confidence"]),
                "frozen_correct": frozen_correct,
                "lora_predicted_attribute_id": lora_id,
                "lora_predicted_attribute_name": lora["raw_predicted_attribute_name"],
                "lora_confidence": float(lora["raw_confidence"]),
                "lora_correct": lora_correct,
                "transition": transition,
            }
        )

    per_class_by_id_frozen = {int(row["attribute_id"]): row for row in frozen_per_class}
    per_class_by_id_lora = {int(row["attribute_id"]): row for row in lora_per_class}
    per_class_rows: list[dict[str, Any]] = []
    for attribute_id in target_ids:
        frozen_row = per_class_by_id_frozen[attribute_id]
        lora_row = per_class_by_id_lora[attribute_id]
        per_class_rows.append(
            {
                "attribute_id": attribute_id,
                "attribute_name": frozen_row["attribute_name"],
                "support": int(frozen_row["support"]),
                "frozen_precision": float(frozen_row["precision"]),
                "frozen_recall": float(frozen_row["recall"]),
                "frozen_f1": float(frozen_row["f1"]),
                "lora_precision": float(lora_row["precision"]),
                "lora_recall": float(lora_row["recall"]),
                "lora_f1": float(lora_row["f1"]),
                "lora_minus_frozen_f1": float(lora_row["f1"]) - float(frozen_row["f1"]),
            }
        )

    raw_metric_rows = []
    frozen_metrics = summary["frozen_head"]["raw_metrics"]
    lora_metrics = summary["lora"]["raw_metrics"]
    for metric in RAW_METRICS:
        raw_metric_rows.append(
            {
                "metric": metric,
                "frozen": float(frozen_metrics[metric]),
                "lora": float(lora_metrics[metric]),
                "lora_minus_frozen": float(lora_metrics[metric]) - float(frozen_metrics[metric]),
            }
        )
    write_csv(tables_dir / "official_validation_metric_comparison.csv", list(raw_metric_rows[0]), raw_metric_rows)
    write_csv(tables_dir / "official_validation_per_class_comparison.csv", list(per_class_rows[0]), per_class_rows)
    transition_rows = [{"transition": key, "count": value, "fraction": value / len(combined_rows)} for key, value in sorted(transitions.items())]
    write_csv(tables_dir / "prediction_transition_summary.csv", ["transition", "count", "fraction"], transition_rows)
    write_csv(tables_dir / "paired_official_validation_predictions.csv", list(combined_rows[0]), combined_rows)

    plot_per_class_f1(per_class_rows, figures_dir / "official_validation_per_class_f1.png")
    plot_confusion(frozen_matrix, labels, "Frozen SigLIP2: raw confusion matrix", figures_dir / "official_validation_frozen_confusion_matrix.png")
    plot_confusion(lora_matrix, labels, "LoRA SigLIP2: raw confusion matrix", figures_dir / "official_validation_lora_confusion_matrix.png")
    plot_reliability(frozen_ece, lora_ece, figures_dir / "official_validation_raw_reliability.png")

    max_examples = max(1, int(args.max_examples_per_group))
    examples_by_transition: dict[str, list[dict[str, Any]]] = {}
    for transition in ("lora_corrected", "lora_regressed", "both_wrong"):
        candidates = [row for row in combined_rows if row["transition"] == transition]
        if transition == "lora_corrected":
            candidates.sort(key=lambda row: (-row["lora_confidence"], row["frozen_confidence"], int(row["annotation_id"])))
        elif transition == "lora_regressed":
            candidates.sort(key=lambda row: (-row["frozen_confidence"], row["lora_confidence"], int(row["annotation_id"])))
        else:
            candidates.sort(key=lambda row: (-row["lora_confidence"], -row["frozen_confidence"], int(row["annotation_id"])))
        examples_by_transition[transition] = candidates[:max_examples]

    crop_index_rows: list[dict[str, Any]] = []
    for transition, label in (
        ("lora_corrected", "LoRA corrected errors"),
        ("lora_regressed", "LoRA regressions"),
        ("both_wrong", "Both models wrong"),
    ):
        crop_index_rows.extend(
            make_contact_sheet(
                label,
                examples_by_transition[transition],
                manifest_by_annotation,
                crops_dir / f"{transition}_contact_sheet.png",
                padding=0.15,
            )
        )
    if crop_index_rows:
        write_csv(crops_dir / "representative_case_index.csv", list(crop_index_rows[0]), crop_index_rows)

    metric_lookup = {row["metric"]: row for row in raw_metric_rows}
    report_lines = [
        "# Fashionpedia Neckline Post-Validation Analysis",
        "",
        "## Scope and boundary",
        "",
        "This is a read-only, post-hoc descriptive analysis of the already-qualified official Fashionpedia validation confirmation. It did not load SigLIP2, PEFT, PyTorch, model checkpoints, or run inference. It did not select a new checkpoint, crop, LoRA rank, training setting, calibration temperature, or decision threshold.",
        "",
        "## Official validation result",
        "",
        f"- Frozen raw Macro-F1: {metric_lookup['macro_f1']['frozen']:.6f}",
        f"- LoRA raw Macro-F1: {metric_lookup['macro_f1']['lora']:.6f}",
        f"- LoRA minus frozen Macro-F1: {metric_lookup['macro_f1']['lora_minus_frozen']:+.6f}",
        f"- Frozen raw ECE: {metric_lookup['top_label_ece']['frozen']:.6f}",
        f"- LoRA raw ECE: {metric_lookup['top_label_ece']['lora']:.6f}",
        "",
        "## Paired prediction transitions",
        "",
    ]
    for row in transition_rows:
        report_lines.append(f"- {row['transition']}: {row['count']} ({row['fraction']:.1%})")
    report_lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- The primary official-validation claim remains the matched-control LoRA Macro-F1 uplift.",
        "- Representative crops are illustrative post-hoc examples, not a new benchmark or selection criterion.",
        "- The raw reliability figure describes stored top-label confidence. It does not establish post-temperature-calibration superiority.",
        "",
        "## Artifacts",
        "",
        "- `tables/official_validation_metric_comparison.csv`",
        "- `tables/official_validation_per_class_comparison.csv`",
        "- `tables/prediction_transition_summary.csv`",
        "- `figures/official_validation_per_class_f1.png`",
        "- `figures/official_validation_frozen_confusion_matrix.png`",
        "- `figures/official_validation_lora_confusion_matrix.png`",
        "- `figures/official_validation_raw_reliability.png`",
        "- `representative_crops/`",
    ]
    (output_dir / "POST_VALIDATION_ANALYSIS_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    artifact_manifest = {
        "status": "PASS",
        "schema_version": "fashionpedia_post_validation_analysis_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "read_only_post_hoc_descriptive_analysis",
        "no_model_runtime_loaded": True,
        "no_inference_performed": True,
        "no_training_performed": True,
        "no_checkpoint_or_hyperparameter_selection_performed": True,
        "source_confirmation_summary": str(confirmation_summary_path),
        "source_confirmation_summary_sha256": sha256_file(confirmation_summary_path),
        "source_artifacts": {
            "frozen_predictions": str(frozen_predictions_path),
            "lora_predictions": str(lora_predictions_path),
            "frozen_per_class": str(frozen_per_class_path),
            "lora_per_class": str(lora_per_class_path),
            "validation_manifest": str(validation_manifest_path),
        },
        "official_validation_samples": len(combined_rows),
        "prediction_transition_counts": dict(transitions),
        "files": [
            {"relative_path": str(path.relative_to(output_dir)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
    }
    write_json(output_dir / "analysis_artifact_manifest.json", artifact_manifest)

    print("===== FASHIONPEDIA POST-VALIDATION ANALYSIS =====", flush=True)
    print("STATUS: PASS", flush=True)
    print(f"OFFICIAL_VALIDATION_SAMPLES: {len(combined_rows)}", flush=True)
    print(f"LORA_CORRECTED: {transitions['lora_corrected']}", flush=True)
    print(f"LORA_REGRESSED: {transitions['lora_regressed']}", flush=True)
    print(f"BOTH_WRONG: {transitions['both_wrong']}", flush=True)
    print(f"ANALYSIS_REPORT: {output_dir / 'POST_VALIDATION_ANALYSIS_REPORT.md'}", flush=True)
    print(f"ANALYSIS_MANIFEST: {output_dir / 'analysis_artifact_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
