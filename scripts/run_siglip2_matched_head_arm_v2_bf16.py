from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import peft
import sklearn
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from transformers import AutoImageProcessor, AutoModel


ENCODER_SELF_ATTN_PATTERN = re.compile(
    r"^vision_model\.encoder\.layers\.\d+\.self_attn\."
    r"(q_proj|k_proj|v_proj|out_proj)$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")

    required = {
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

    missing = sorted(required - set(rows[0]))

    if missing:
        raise RuntimeError(
            f"Manifest missing required columns: {missing}"
        )

    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


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


def prepare_inputs(
    records: list[dict[str, str]],
    image_processor,
    device: torch.device,
    padding_fraction: float,
) -> dict[str, torch.Tensor]:
    crops = [
        crop_from_record(record, padding_fraction)
        for record in records
    ]

    processor_outputs = image_processor(
        images=crops,
        return_tensors="pt",
    )

    return {
        name: value.to(device, non_blocking=True)
        for name, value in processor_outputs.items()
        if hasattr(value, "to")
    }


def freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def cpu_state_dict(module: torch.nn.Module) -> dict:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def count_parameters(module: torch.nn.Module) -> int:
    return int(
        sum(parameter.numel() for parameter in module.parameters())
    )


def trainable_named_parameters(
    head: torch.nn.Module,
    lora_model: torch.nn.Module | None,
) -> list[tuple[str, torch.nn.Parameter]]:
    rows = [
        (f"head.{name}", parameter)
        for name, parameter in head.named_parameters()
        if parameter.requires_grad
    ]

    if lora_model is not None:
        rows.extend(
            (f"lora.{name}", parameter)
            for name, parameter in lora_model.named_parameters()
            if parameter.requires_grad
        )

    return rows


def gradient_health(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict:
    missing = []
    nonfinite = []
    max_abs_gradient = 0.0

    for name, parameter in named_parameters:
        gradient = parameter.grad

        if gradient is None:
            missing.append(name)
            continue

        if not torch.isfinite(gradient).all().item():
            nonfinite.append(name)
            continue

        max_abs_gradient = max(
            max_abs_gradient,
            float(gradient.detach().abs().max().cpu().item()),
        )

    return {
        "trainable_tensor_count": len(named_parameters),
        "missing_gradient_names": missing,
        "nonfinite_gradient_names": nonfinite,
        "max_abs_gradient": max_abs_gradient,
        "all_present_and_finite": (
            len(missing) == 0
            and len(nonfinite) == 0
        ),
    }


def top_label_ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bin_count: int,
) -> tuple[float, list[dict]]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == targets).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bin_count + 1)
    ece = 0.0
    rows = []

    for index in range(bin_count):
        lower = float(edges[index])
        upper = float(edges[index + 1])

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
        mean_accuracy = float(correct[mask].mean())
        gap = abs(mean_confidence - mean_accuracy)

        ece += float(count) / float(len(targets)) * gap

        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": mean_accuracy,
                "absolute_gap": gap,
            }
        )

    return float(ece), rows


def build_class_weights(
    labels: np.ndarray,
    class_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    counts = np.bincount(labels, minlength=class_count)

    if np.any(counts <= 0):
        raise RuntimeError(
            "Every target class must appear in train labels."
        )

    total = float(labels.shape[0])
    raw = total / (
        float(class_count)
        * counts.astype(np.float64)
    )

    return (
        torch.tensor(raw, dtype=torch.float32, device=device),
        {
            str(index): float(raw[index])
            for index in range(class_count)
        },
    )


def evaluate(
    encoder: torch.nn.Module,
    mode_module: torch.nn.Module,
    head: torch.nn.Module,
    records: list[dict[str, str]],
    labels: np.ndarray,
    class_weights: torch.Tensor,
    image_processor,
    device: torch.device,
    padding_fraction: float,
    batch_size: int,
    target_attribute_ids: list[int],
    target_attribute_names: dict[int, str],
    ece_bins: int,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    mode_module.eval()
    head.eval()

    logits_chunks = []
    target_chunks = []
    weighted_loss_sum = 0.0
    observed_count = 0

    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            end = min(start + batch_size, len(records))

            batch_records = records[start:end]
            batch_targets = torch.tensor(
                labels[start:end],
                dtype=torch.long,
                device=device,
            )

            model_inputs = prepare_inputs(
                records=batch_records,
                image_processor=image_processor,
                device=device,
                padding_fraction=padding_fraction,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                features = encoder.get_image_features(
                    **model_inputs,
                )
                logits = head(features)

            loss = F.cross_entropy(
                logits.float(),
                batch_targets,
                weight=class_weights,
            )

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    "Non-finite evaluation loss."
                )

            logits_chunks.append(
                logits.detach().float().cpu().numpy()
            )

            target_chunks.append(
                batch_targets.detach().cpu().numpy()
            )

            weighted_loss_sum += (
                float(loss.detach().cpu().item())
                * len(batch_records)
            )

            observed_count += len(batch_records)

    logits_all = np.concatenate(logits_chunks, axis=0)
    y_true = np.concatenate(target_chunks, axis=0)

    if logits_all.shape != (
        len(records),
        len(target_attribute_ids),
    ):
        raise RuntimeError(
            "Unexpected evaluation logit shape: "
            f"{logits_all.shape}"
        )

    shifted = logits_all - logits_all.max(
        axis=1,
        keepdims=True,
    )

    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(
        axis=1,
        keepdims=True,
    )

    if not np.isfinite(probabilities).all():
        raise FloatingPointError(
            "Non-finite evaluation probabilities."
        )

    predicted_indices = probabilities.argmax(axis=1)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            predicted_indices,
            labels=list(range(len(target_attribute_ids))),
            zero_division=0,
        )
    )

    confusion = confusion_matrix(
        y_true,
        predicted_indices,
        labels=list(range(len(target_attribute_ids))),
    )

    ece, ece_rows = top_label_ece(
        probabilities=probabilities,
        targets=y_true,
        bin_count=ece_bins,
    )

    top_2 = np.argsort(-probabilities, axis=1)[:, :2]

    metrics = {
        "evaluation_weighted_cross_entropy": (
            weighted_loss_sum / float(observed_count)
        ),
        "accuracy": float(
            accuracy_score(y_true, predicted_indices)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predicted_indices)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predicted_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                predicted_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "negative_log_likelihood": float(
            log_loss(
                y_true,
                probabilities,
                labels=list(range(len(target_attribute_ids))),
            )
        ),
        "multiclass_brier": float(
            np.mean(
                np.sum(
                    (
                        probabilities
                        - np.eye(
                            len(target_attribute_ids),
                            dtype=np.float64,
                        )[y_true]
                    ) ** 2,
                    axis=1,
                )
            )
        ),
        "top_label_ece": float(ece),
        "top_2_accuracy": float(
            np.mean(
                np.any(
                    top_2 == y_true[:, None],
                    axis=1,
                )
            )
        ),
    }

    per_class_rows = []

    for class_index, attribute_id in enumerate(
        target_attribute_ids
    ):
        per_class_rows.append(
            {
                "attribute_id": int(attribute_id),
                "attribute_name": target_attribute_names[
                    int(attribute_id)
                ],
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
            }
        )

    prediction_rows = []

    for row_index, record in enumerate(records):
        predicted_attribute_id = target_attribute_ids[
            int(predicted_indices[row_index])
        ]

        row = {
            "annotation_id": record["annotation_id"],
            "image_id": record["image_id"],
            "split_group_id": record["split_group_id"],
            "image_path": record["image_path"],
            "true_attribute_id": int(record["attribute_id"]),
            "true_attribute_name": record["attribute_name"],
            "predicted_attribute_id": int(
                predicted_attribute_id
            ),
            "predicted_attribute_name": (
                target_attribute_names[
                    predicted_attribute_id
                ]
            ),
            "confidence": float(
                probabilities[row_index].max()
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
                probabilities[row_index, class_position]
            )

        prediction_rows.append(row)

    confusion_rows = []

    for true_index, true_attribute_id in enumerate(
        target_attribute_ids
    ):
        row = {
            "true_attribute_id": int(true_attribute_id),
            "true_attribute_name": target_attribute_names[
                int(true_attribute_id)
            ],
        }

        for predicted_index, predicted_attribute_id in enumerate(
            target_attribute_ids
        ):
            row[
                f"predicted_attribute_{predicted_attribute_id}"
            ] = int(
                confusion[true_index, predicted_index]
            )

        confusion_rows.append(row)

    return (
        metrics,
        per_class_rows,
        prediction_rows,
        confusion_rows,
        ece_rows,
    )


def train_epoch(
    arm_name: str,
    encoder: torch.nn.Module,
    mode_module: torch.nn.Module,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_records: list[dict[str, str]],
    train_labels: np.ndarray,
    class_weights: torch.Tensor,
    image_processor,
    device: torch.device,
    padding_fraction: float,
    physical_batch_size: int,
    accumulation_steps: int,
    epoch: int,
    shuffle_seed: int,
    max_grad_norm: float,
) -> dict:
    if arm_name == "frozen_head":
        mode_module.eval()
        encoder.eval()
    else:
        mode_module.train()

    head.train()

    order = np.random.default_rng(
        shuffle_seed + epoch
    ).permutation(len(train_records))

    total_loss = 0.0
    total_samples = 0
    optimizer_steps = 0
    max_abs_gradient = 0.0

    effective_group_size = (
        physical_batch_size * accumulation_steps
    )

    for group_start in range(
        0,
        len(train_records),
        effective_group_size,
    ):
        group_indices = order[
            group_start:min(
                group_start + effective_group_size,
                len(train_records),
            )
        ]

        micro_batches = [
            group_indices[
                start:min(
                    start + physical_batch_size,
                    len(group_indices),
                )
            ]
            for start in range(
                0,
                len(group_indices),
                physical_batch_size,
            )
        ]

        optimizer.zero_grad(set_to_none=True)

        for micro_indices in micro_batches:
            batch_records = [
                train_records[int(index)]
                for index in micro_indices
            ]

            batch_targets = torch.tensor(
                train_labels[micro_indices],
                dtype=torch.long,
                device=device,
            )

            model_inputs = prepare_inputs(
                records=batch_records,
                image_processor=image_processor,
                device=device,
                padding_fraction=padding_fraction,
            )

            if arm_name == "frozen_head":
                with torch.no_grad():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                    ):
                        features = encoder.get_image_features(
                            **model_inputs,
                        )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    logits = head(features)
            else:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    features = encoder.get_image_features(
                        **model_inputs,
                    )
                    logits = head(features)

            raw_loss = F.cross_entropy(
                logits.float(),
                batch_targets,
                weight=class_weights,
            )

            if not torch.isfinite(raw_loss).item():
                raise FloatingPointError(
                    "Non-finite loss "
                    f"arm={arm_name} epoch={epoch} "
                    f"group_start={group_start}"
                )

            (
                raw_loss / float(len(micro_batches))
            ).backward()

            total_loss += (
                float(raw_loss.detach().cpu().item())
                * len(batch_records)
            )

            total_samples += len(batch_records)

        trainable = trainable_named_parameters(
            head=head,
            lora_model=(
                None
                if arm_name == "frozen_head"
                else mode_module
            ),
        )

        health = gradient_health(trainable)

        if not health["all_present_and_finite"]:
            raise FloatingPointError(
                json.dumps(
                    {
                        "reason": "gradient_health_failure",
                        "arm": arm_name,
                        "epoch": epoch,
                        "group_start": int(group_start),
                        "details": health,
                    },
                    indent=2,
                )
            )

        max_abs_gradient = max(
            max_abs_gradient,
            float(health["max_abs_gradient"]),
        )

        clip_value = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable],
            max_norm=max_grad_norm,
            error_if_nonfinite=True,
        )

        if not torch.isfinite(clip_value).item():
            raise FloatingPointError(
                "Non-finite gradient norm "
                f"arm={arm_name} epoch={epoch}"
            )

        optimizer.step()
        optimizer_steps += 1

    return {
        "train_weighted_cross_entropy": (
            total_loss / float(total_samples)
        ),
        "optimizer_steps": int(optimizer_steps),
        "max_abs_gradient": float(max_abs_gradient),
    }


def better_candidate(
    candidate: dict,
    current_best: dict | None,
) -> bool:
    if current_best is None:
        return True

    if (
        candidate["macro_f1"]
        > current_best["macro_f1"] + 1e-12
    ):
        return True

    return (
        abs(
            candidate["macro_f1"]
            - current_best["macro_f1"]
        )
        <= 1e-12
        and candidate["negative_log_likelihood"]
        < current_best["negative_log_likelihood"]
    )


def save_checkpoint(
    arm_name: str,
    arm_dir: Path,
    mode_module: torch.nn.Module,
    head: torch.nn.Module,
    epoch: int,
    metrics: dict,
) -> dict:
    head_path = arm_dir / "best_head_state.pt"

    torch.save(
        {
            "epoch": int(epoch),
            "head_state_dict": cpu_state_dict(head),
            "selected_metrics": metrics,
        },
        head_path,
    )

    adapter_dir = None

    if arm_name == "lora":
        adapter_dir = arm_dir / "best_lora_adapter"

        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)

        mode_module.save_pretrained(
            adapter_dir,
            safe_serialization=True,
        )

    metadata_path = arm_dir / "best_checkpoint_metadata.json"

    metadata = {
        "arm_name": arm_name,
        "epoch": int(epoch),
        "head_state_path": str(head_path),
        "adapter_directory": (
            None
            if adapter_dir is None
            else str(adapter_dir)
        ),
        "selected_metrics": metrics,
    }

    write_json(metadata_path, metadata)

    return {
        "head_state_path": str(head_path),
        "adapter_directory": (
            None
            if adapter_dir is None
            else str(adapter_dir)
        ),
        "checkpoint_metadata_json": str(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        required=True,
        choices=["frozen_head", "lora"],
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--manifest-summary", required=True)
    parser.add_argument("--hog-config", required=True)
    parser.add_argument("--lora-audit", required=True)
    parser.add_argument("--bf16-preflight", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    started = time.perf_counter()

    train_manifest_path = Path(args.train_manifest).resolve()
    dev_manifest_path = Path(args.dev_manifest).resolve()
    manifest_summary_path = Path(
        args.manifest_summary
    ).resolve()
    hog_config_path = Path(args.hog_config).resolve()
    audit_path = Path(args.lora_audit).resolve()
    bf16_preflight_path = Path(args.bf16_preflight).resolve()
    config_path = Path(args.config).resolve()
    cache_root = Path(args.cache_dir).resolve()
    run_root = Path(args.run_dir).resolve()

    for path in (
        train_manifest_path,
        dev_manifest_path,
        manifest_summary_path,
        hog_config_path,
        audit_path,
        bf16_preflight_path,
        config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact missing: {path}"
            )

    if not run_root.is_dir():
        raise FileNotFoundError(
            f"Run root does not exist: {run_root}"
        )

    arm_dir = run_root / args.arm

    if arm_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite arm directory: {arm_dir}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "CUDA device does not report bf16 support."
        )

    manifest_summary = load_json(manifest_summary_path)
    hog_config = load_json(hog_config_path)
    lora_audit = load_json(audit_path)
    bf16_preflight = load_json(bf16_preflight_path)
    config = load_json(config_path)

    if manifest_summary.get("status") != "PASS":
        raise RuntimeError("Manifest summary is not PASS.")

    if not manifest_summary.get(
        "checks",
        {},
    ).get("group_disjoint_pass", False):
        raise RuntimeError("Manifest is not group-disjoint.")

    if lora_audit.get("status") != "PASS":
        raise RuntimeError("LoRA target audit is not PASS.")

    if bf16_preflight.get("status") != "PASS":
        raise RuntimeError("BF16 one-update preflight is not PASS.")

    if not bf16_preflight.get(
        "checks",
        {},
    ).get(
        "frozen_gradients_finite_and_present",
        False,
    ):
        raise RuntimeError(
            "Frozen BF16 gradient preflight was not qualified."
        )

    if not bf16_preflight.get(
        "checks",
        {},
    ).get(
        "lora_gradients_finite_and_present",
        False,
    ):
        raise RuntimeError(
            "LoRA BF16 gradient preflight was not qualified."
        )

    train_records = load_manifest(train_manifest_path)
    dev_records = load_manifest(dev_manifest_path)

    train_groups = {
        row["split_group_id"]
        for row in train_records
    }

    dev_groups = {
        row["split_group_id"]
        for row in dev_records
    }

    group_overlap = train_groups & dev_groups

    if group_overlap:
        raise RuntimeError(
            "Train/dev group overlap detected: "
            f"{len(group_overlap)}"
        )

    target_attribute_ids = [
        int(value)
        for value in config["target_attribute_ids"]
    ]

    target_attribute_names = {
        int(attribute_id): str(name)
        for attribute_id, name in config[
            "target_attribute_names"
        ].items()
    }

    target_attribute_set = set(target_attribute_ids)

    train_attributes = {
        int(row["attribute_id"])
        for row in train_records
    }

    dev_attributes = {
        int(row["attribute_id"])
        for row in dev_records
    }

    if train_attributes != target_attribute_set:
        raise RuntimeError("Train labels do not match task labels.")

    if dev_attributes != target_attribute_set:
        raise RuntimeError("Dev labels do not match task labels.")

    class_index = {
        attribute_id: index
        for index, attribute_id in enumerate(
            target_attribute_ids
        )
    }

    y_train = np.asarray(
        [
            class_index[int(row["attribute_id"])]
            for row in train_records
        ],
        dtype=np.int64,
    )

    y_dev = np.asarray(
        [
            class_index[int(row["attribute_id"])]
            for row in dev_records
        ],
        dtype=np.int64,
    )

    padding_fraction = float(
        config["crop"]["padding_fraction"]
    )

    if abs(
        padding_fraction
        - float(hog_config["crop"]["padding_fraction"])
    ) > 1e-12:
        raise RuntimeError(
            "HOG and matched-head crop padding do not match."
        )

    raw_targets = lora_audit[
        "recommended_lora_scope"
    ]["full_target_module_paths"]

    lora_targets = sorted(
        path
        for path in raw_targets
        if ENCODER_SELF_ATTN_PATTERN.fullmatch(path)
    )

    excluded_targets = sorted(
        set(raw_targets) - set(lora_targets)
    )

    if len(lora_targets) != int(
        config["lora"]["expected_target_module_count"]
    ):
        raise RuntimeError(
            "LoRA target count does not match config."
        )

    if excluded_targets != [
        str(config["lora"]["excluded_module"])
    ]:
        raise RuntimeError(
            "Unexpected LoRA excluded target list: "
            f"{excluded_targets}"
        )

    device = torch.device("cuda")
    set_seed(int(config["training"]["seed"]))

    image_processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        cache_dir=str(cache_root),
        use_fast=False,
    )

    base_model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=str(cache_root),
    )

    freeze_module(base_model)

    set_seed(
        int(config["head"]["initialization_seed"])
    )

    head = torch.nn.Linear(
        int(config["head"]["input_dimension"]),
        int(config["head"]["output_classes"]),
    ).to(device)

    lora_model = None
    encoder = base_model
    mode_module = base_model
    adapter_parameter_count = 0
    unexpected_base_trainables = []

    if args.arm == "frozen_head":
        base_model.to(device)
        base_model.eval()

        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=float(
                config["training"]["head_learning_rate"]
            ),
            weight_decay=float(
                config["training"]["weight_decay"]
            ),
        )
    else:
        lora_config = LoraConfig(
            r=int(config["lora"]["rank"]),
            lora_alpha=int(config["lora"]["alpha"]),
            lora_dropout=float(config["lora"]["dropout"]),
            bias=str(config["lora"]["bias"]),
            target_modules=lora_targets,
        )

        lora_model = get_peft_model(
            base_model,
            lora_config,
        )

        lora_model.to(device)
        encoder = lora_model.get_base_model()
        mode_module = lora_model

        adapter_parameter_count = int(
            sum(
                parameter.numel()
                for parameter in lora_model.parameters()
                if parameter.requires_grad
            )
        )

        unexpected_base_trainables = [
            name
            for name, parameter in lora_model.named_parameters()
            if parameter.requires_grad
            and ".lora_A." not in name
            and ".lora_B." not in name
        ]

        if adapter_parameter_count != int(
            config["lora"]["expected_adapter_parameter_count"]
        ):
            raise RuntimeError(
                "LoRA adapter parameter count mismatch: "
                f"{adapter_parameter_count}"
            )

        if unexpected_base_trainables:
            raise RuntimeError(
                "Unexpected base trainables: "
                f"{unexpected_base_trainables}"
            )

        adapter_parameters = [
            parameter
            for parameter in lora_model.parameters()
            if parameter.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": adapter_parameters,
                    "lr": float(
                        config["training"][
                            "lora_learning_rate"
                        ]
                    ),
                    "weight_decay": float(
                        config["training"]["weight_decay"]
                    ),
                },
                {
                    "params": list(head.parameters()),
                    "lr": float(
                        config["training"][
                            "head_learning_rate"
                        ]
                    ),
                    "weight_decay": float(
                        config["training"]["weight_decay"]
                    ),
                },
            ]
        )

    class_weights, class_weight_map = build_class_weights(
        labels=y_train,
        class_count=len(target_attribute_ids),
        device=device,
    )

    arm_dir.mkdir(parents=True, exist_ok=False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print(
        "===== SIGLIP2 MATCHED-HEAD BF16 ARM PRE-FLIGHT =====",
        flush=True,
    )
    print(f"ARM: {args.arm}", flush=True)
    print(f"TRAIN_SAMPLES: {len(train_records)}", flush=True)
    print(f"DEV_SAMPLES: {len(dev_records)}", flush=True)
    print(
        f"GROUP_OVERLAP_COUNT: {len(group_overlap)}",
        flush=True,
    )
    print(f"DEVICE: {device}", flush=True)
    print("AUTOCAST_DTYPE: bfloat16", flush=True)
    print("GRAD_SCALER: False", flush=True)
    print(
        "PHYSICAL_BATCH_SIZE: "
        f"{config['training']['physical_batch_size']}",
        flush=True,
    )
    print(
        "GRADIENT_ACCUMULATION_STEPS: "
        f"{config['training']['gradient_accumulation_steps']}",
        flush=True,
    )
    print(
        "EFFECTIVE_BATCH_SIZE: "
        f"{config['training']['effective_batch_size']}",
        flush=True,
    )
    print(f"EPOCHS: {config['training']['epochs']}", flush=True)
    print(
        f"HEAD_PARAMETER_COUNT: {count_parameters(head)}",
        flush=True,
    )
    print(
        f"ADAPTER_PARAMETER_COUNT: {adapter_parameter_count}",
        flush=True,
    )

    epochs = int(config["training"]["epochs"])
    batch_size = int(
        config["training"]["physical_batch_size"]
    )
    accumulation_steps = int(
        config["training"]["gradient_accumulation_steps"]
    )

    epoch_rows = []
    best_metrics = None
    best_epoch = None
    best_per_class = None
    best_predictions = None
    best_confusion = None
    best_ece = None
    best_checkpoint = None

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_result = train_epoch(
            arm_name=args.arm,
            encoder=encoder,
            mode_module=mode_module,
            head=head,
            optimizer=optimizer,
            train_records=train_records,
            train_labels=y_train,
            class_weights=class_weights,
            image_processor=image_processor,
            device=device,
            padding_fraction=padding_fraction,
            physical_batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            epoch=epoch,
            shuffle_seed=int(config["training"]["seed"]),
            max_grad_norm=float(
                config["training"]["max_grad_norm"]
            ),
        )

        (
            dev_metrics,
            per_class_rows,
            prediction_rows,
            confusion_rows,
            ece_rows,
        ) = evaluate(
            encoder=encoder,
            mode_module=mode_module,
            head=head,
            records=dev_records,
            labels=y_dev,
            class_weights=class_weights,
            image_processor=image_processor,
            device=device,
            padding_fraction=padding_fraction,
            batch_size=batch_size,
            target_attribute_ids=target_attribute_ids,
            target_attribute_names=target_attribute_names,
            ece_bins=int(config["evaluation"]["ece_bins"]),
        )

        elapsed = time.perf_counter() - epoch_start

        epoch_row = {
            "epoch": int(epoch),
            "elapsed_seconds": round(elapsed, 3),
            **train_result,
            **dev_metrics,
        }

        epoch_rows.append(epoch_row)

        print(
            "epoch_summary "
            f"arm={args.arm} "
            f"epoch={epoch} "
            f"train_loss={train_result['train_weighted_cross_entropy']:.6f} "
            f"dev_macro_f1={dev_metrics['macro_f1']:.6f} "
            f"dev_balanced_accuracy={dev_metrics['balanced_accuracy']:.6f} "
            f"dev_ece={dev_metrics['top_label_ece']:.6f} "
            f"dev_nll={dev_metrics['negative_log_likelihood']:.6f}",
            flush=True,
        )

        if better_candidate(dev_metrics, best_metrics):
            best_epoch = int(epoch)
            best_metrics = dict(dev_metrics)
            best_per_class = per_class_rows
            best_predictions = prediction_rows
            best_confusion = confusion_rows
            best_ece = ece_rows

            best_checkpoint = save_checkpoint(
                arm_name=args.arm,
                arm_dir=arm_dir,
                mode_module=mode_module,
                head=head,
                epoch=epoch,
                metrics=dev_metrics,
            )

            print(
                "best_checkpoint_updated "
                f"arm={args.arm} "
                f"epoch={epoch} "
                f"macro_f1={dev_metrics['macro_f1']:.6f}",
                flush=True,
            )

    if best_epoch is None:
        raise RuntimeError("No best epoch was selected.")

    epoch_metrics_path = arm_dir / "epoch_metrics.csv"
    per_class_path = arm_dir / "best_dev_per_class_metrics.csv"
    predictions_path = arm_dir / "best_dev_predictions.csv"
    confusion_path = arm_dir / "best_dev_confusion_matrix.csv"
    ece_path = arm_dir / "best_dev_ece_bins.json"
    summary_path = arm_dir / "arm_summary.json"

    write_csv(
        epoch_metrics_path,
        fieldnames=list(epoch_rows[0].keys()),
        rows=epoch_rows,
    )

    write_csv(
        per_class_path,
        fieldnames=[
            "attribute_id",
            "attribute_name",
            "precision",
            "recall",
            "f1",
            "support",
        ],
        rows=best_per_class,
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
        fieldnames=prediction_fields,
        rows=best_predictions,
    )

    write_csv(
        confusion_path,
        fieldnames=[
            "true_attribute_id",
            "true_attribute_name",
        ] + [
            f"predicted_attribute_{attribute_id}"
            for attribute_id in target_attribute_ids
        ],
        rows=best_confusion,
    )

    write_json(
        ece_path,
        {
            "top_label_ece": best_metrics["top_label_ece"],
            "bin_count": int(
                config["evaluation"]["ece_bins"]
            ),
            "bins": best_ece,
        },
    )

    expected_head_parameters = (
        int(config["head"]["input_dimension"])
        * int(config["head"]["output_classes"])
        + int(config["head"]["output_classes"])
    )

    checks = {
        "source_manifest_pass": (
            manifest_summary.get("status") == "PASS"
        ),
        "group_disjoint_pass": len(group_overlap) == 0,
        "bf16_supported": bool(
            torch.cuda.is_bf16_supported()
        ),
        "bf16_preflight_pass": (
            bf16_preflight.get("status") == "PASS"
        ),
        "best_epoch_selected": best_epoch is not None,
        "all_epoch_optimizer_steps_positive": all(
            row["optimizer_steps"] > 0
            for row in epoch_rows
        ),
        "finite_selected_macro_f1": bool(
            np.isfinite(best_metrics["macro_f1"])
        ),
        "finite_selected_nll": bool(
            np.isfinite(
                best_metrics["negative_log_likelihood"]
            )
        ),
        "head_parameter_count_expected": (
            count_parameters(head)
            == expected_head_parameters
        ),
        "best_prediction_count_matches_dev": (
            len(best_predictions) == len(dev_records)
        ),
    }

    if args.arm == "frozen_head":
        checks["frozen_adapter_parameter_count_zero"] = (
            adapter_parameter_count == 0
        )
    else:
        checks["lora_target_count_matches"] = (
            len(lora_targets)
            == int(
                config["lora"][
                    "expected_target_module_count"
                ]
            )
        )

        checks["lora_adapter_parameter_count_matches"] = (
            adapter_parameter_count
            == int(
                config["lora"][
                    "expected_adapter_parameter_count"
                ]
            )
        )

        checks["no_unexpected_base_trainables"] = (
            len(unexpected_base_trainables) == 0
        )

    status = "PASS" if all(checks.values()) else "REVIEW"

    summary = {
        "status": status,
        "schema_version": (
            "siglip2_matched_head_arm_v2_bf16"
        ),
        "arm_name": args.arm,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "selection_scope": "train_only_development",
        "selected_epoch": int(best_epoch),
        "selected_metrics": best_metrics,
        "selection_policy": config["selection"],
        "precision_policy": {
            "autocast_dtype": "bfloat16",
            "grad_scaler_used": False,
            "loss_dtype": "float32"
        },
        "source_artifacts": {
            "train_manifest": str(train_manifest_path),
            "dev_manifest": str(dev_manifest_path),
            "manifest_summary": str(manifest_summary_path),
            "hog_config": str(hog_config_path),
            "lora_audit": str(audit_path),
            "bf16_preflight": str(bf16_preflight_path),
            "config": str(config_path),
            "train_manifest_sha256": sha256_file(
                train_manifest_path
            ),
            "dev_manifest_sha256": sha256_file(
                dev_manifest_path
            ),
            "config_sha256": sha256_file(config_path),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "scikit_learn": sklearn.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_peak_memory_allocated_mib": round(
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 2),
                2,
            ),
        },
        "data": {
            "train_samples": len(train_records),
            "dev_samples": len(dev_records),
            "train_image_groups": len(train_groups),
            "dev_image_groups": len(dev_groups),
            "group_overlap_count": len(group_overlap),
            "target_attribute_ids": target_attribute_ids,
            "class_weight_map": class_weight_map,
        },
        "parameter_counts": {
            "classification_head": count_parameters(head),
            "lora_adapter": adapter_parameter_count,
            "total_trainable": (
                count_parameters(head)
                + adapter_parameter_count
            ),
        },
        "lora_scope": {
            "selected_target_modules": (
                []
                if args.arm == "frozen_head"
                else lora_targets
            ),
            "unexpected_base_trainables": (
                unexpected_base_trainables
            ),
        },
        "qualification_checks": checks,
        "artifacts": {
            "epoch_metrics_csv": str(epoch_metrics_path),
            "best_per_class_metrics_csv": str(per_class_path),
            "best_dev_predictions_csv": str(predictions_path),
            "best_dev_confusion_matrix_csv": str(
                confusion_path
            ),
            "best_dev_ece_bins_json": str(ece_path),
            **best_checkpoint,
        },
        "elapsed_seconds": round(
            time.perf_counter() - started,
            3,
        ),
        "claim_boundary": config["claim_boundary"],
    }

    write_json(summary_path, summary)

    print("")
    print("===== SIGLIP2 MATCHED-HEAD BF16 ARM SUMMARY =====")
    print(f"STATUS: {status}")
    print(f"ARM: {args.arm}")
    print(f"SELECTED_EPOCH: {best_epoch}")
    print(
        f"MACRO_F1: {best_metrics['macro_f1']:.6f}"
    )
    print(
        "BALANCED_ACCURACY: "
        f"{best_metrics['balanced_accuracy']:.6f}"
    )
    print(
        f"TOP_LABEL_ECE: {best_metrics['top_label_ece']:.6f}"
    )
    print(
        "NEGATIVE_LOG_LIKELIHOOD: "
        f"{best_metrics['negative_log_likelihood']:.6f}"
    )
    print(
        "TOP_2_ACCURACY: "
        f"{best_metrics['top_2_accuracy']:.6f}"
    )
    print(
        "GPU_PEAK_MEMORY_ALLOCATED_MIB: "
        f"{summary['environment']['gpu_peak_memory_allocated_mib']:.2f}"
    )
    print(f"ARM_SUMMARY_JSON: {summary_path}")

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
