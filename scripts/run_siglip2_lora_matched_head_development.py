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

import joblib
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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def make_model_inputs(
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
        bin_accuracy = float(correct[mask].mean())
        absolute_gap = abs(mean_confidence - bin_accuracy)

        ece += (
            float(count)
            / float(len(targets))
            * absolute_gap
        )

        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": bin_accuracy,
                "absolute_gap": absolute_gap,
            }
        )

    return float(ece), rows


def count_parameters(module: torch.nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
        )
    )


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    )


def freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def cpu_state_dict(module: torch.nn.Module) -> dict:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def all_trainable_gradients_finite(
    modules: list[torch.nn.Module],
) -> bool:
    trainable_parameters = []

    for module in modules:
        trainable_parameters.extend(
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    if not trainable_parameters:
        return False

    return all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        for parameter in trainable_parameters
    )


def build_class_weights(
    labels: np.ndarray,
    class_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    counts = np.bincount(labels, minlength=class_count)

    if np.any(counts <= 0):
        raise RuntimeError(
            "Every target class must appear in training labels."
        )

    total = float(labels.shape[0])
    raw_weights = total / (
        float(class_count)
        * counts.astype(np.float64)
    )

    weights = torch.tensor(
        raw_weights,
        dtype=torch.float32,
        device=device,
    )

    return weights, {
        str(index): float(raw_weights[index])
        for index in range(class_count)
    }


def evaluate(
    encoder: torch.nn.Module,
    mode_module: torch.nn.Module,
    head: torch.nn.Module,
    records: list[dict[str, str]],
    labels: np.ndarray,
    class_weights: torch.Tensor,
    target_attribute_ids: list[int],
    target_attribute_names: dict[int, str],
    image_processor,
    device: torch.device,
    padding_fraction: float,
    batch_size: int,
    ece_bins: int,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    mode_module.eval()
    head.eval()

    logits_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    weighted_loss_sum = 0.0
    sample_count = 0

    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            end = min(start + batch_size, len(records))
            batch_records = records[start:end]

            model_inputs = make_model_inputs(
                records=batch_records,
                image_processor=image_processor,
                device=device,
                padding_fraction=padding_fraction,
            )

            batch_targets = torch.tensor(
                labels[start:end],
                dtype=torch.long,
                device=device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                features = encoder.get_image_features(
                    **model_inputs,
                )
                logits = head(features)
                loss = F.cross_entropy(
                    logits,
                    batch_targets,
                    weight=class_weights,
                )

            logits_chunks.append(
                logits.detach().float().cpu().numpy()
            )

            target_chunks.append(
                batch_targets.detach().cpu().numpy()
            )

            weighted_loss_sum += (
                float(loss.detach().float().cpu().item())
                * len(batch_records)
            )

            sample_count += len(batch_records)

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

    if not np.isfinite(logits_all).all():
        raise RuntimeError(
            "Non-finite logits encountered during evaluation."
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
        raise RuntimeError(
            "Non-finite probabilities encountered."
        )

    predicted_indices = probabilities.argmax(axis=1)

    accuracy = float(
        accuracy_score(y_true, predicted_indices)
    )

    balanced_accuracy = float(
        balanced_accuracy_score(y_true, predicted_indices)
    )

    macro_f1 = float(
        f1_score(
            y_true,
            predicted_indices,
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = float(
        f1_score(
            y_true,
            predicted_indices,
            average="weighted",
            zero_division=0,
        )
    )

    negative_log_likelihood = float(
        log_loss(
            y_true,
            probabilities,
            labels=list(range(len(target_attribute_ids))),
        )
    )

    one_hot = np.eye(
        len(target_attribute_ids),
        dtype=np.float64,
    )[y_true]

    multiclass_brier = float(
        np.mean(
            np.sum(
                (probabilities - one_hot) ** 2,
                axis=1,
            )
        )
    )

    ece, ece_rows = top_label_ece(
        probabilities=probabilities,
        targets=y_true,
        bin_count=ece_bins,
    )

    top_2 = np.argsort(
        -probabilities,
        axis=1,
    )[:, :2]

    top_2_accuracy = float(
        np.mean(
            np.any(
                top_2 == y_true[:, None],
                axis=1,
            )
        )
    )

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

    per_class_rows: list[dict] = []

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

    prediction_rows: list[dict] = []

    for row_index, record in enumerate(records):
        predicted_attribute_id = target_attribute_ids[
            int(predicted_indices[row_index])
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

    confusion_rows: list[dict] = []

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

    metrics = {
        "evaluation_weighted_cross_entropy": (
            weighted_loss_sum / float(sample_count)
        ),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "negative_log_likelihood": negative_log_likelihood,
        "multiclass_brier": multiclass_brier,
        "top_label_ece": float(ece),
        "top_2_accuracy": top_2_accuracy,
    }

    return metrics, per_class_rows, prediction_rows, {
        "ece_rows": ece_rows,
        "confusion_rows": confusion_rows,
    }


def save_best_checkpoint(
    arm_name: str,
    arm_dir: Path,
    mode_module: torch.nn.Module,
    head: torch.nn.Module,
    epoch: int,
    metrics: dict,
) -> dict:
    head_path = arm_dir / "best_head_state.pt"
    metadata_path = arm_dir / "best_checkpoint_metadata.json"

    torch.save(
        {
            "epoch": int(epoch),
            "head_state_dict": cpu_state_dict(head),
            "metrics": metrics,
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

    metadata = {
        "arm_name": arm_name,
        "epoch": int(epoch),
        "head_state_path": str(head_path),
        "adapter_directory": (
            None
            if adapter_dir is None
            else str(adapter_dir)
        ),
        "selection_metrics": metrics,
    }

    write_json(metadata_path, metadata)

    return {
        "head_state_path": str(head_path),
        "adapter_directory": (
            None
            if adapter_dir is None
            else str(adapter_dir)
        ),
        "metadata_path": str(metadata_path),
    }


def build_arm(
    arm_name: str,
    model_id: str,
    cache_root: Path,
    initial_head_state: dict,
    config: dict,
    target_modules: list[str],
    device: torch.device,
) -> dict:
    base_model = AutoModel.from_pretrained(
        model_id,
        cache_dir=str(cache_root),
    )

    freeze_module(base_model)

    head = torch.nn.Linear(
        int(config["head"]["input_dimension"]),
        int(config["head"]["output_classes"]),
    )

    head.load_state_dict(
        copy.deepcopy(initial_head_state)
    )

    head.to(device)

    if arm_name == "frozen_head":
        base_model.to(device)
        base_model.eval()

        return {
            "arm_name": arm_name,
            "encoder": base_model,
            "mode_module": base_model,
            "head": head,
            "adapter_parameter_count": 0,
            "adapter_trainable_names": [],
            "unexpected_base_trainables": [],
            "selected_target_modules": [],
        }

    if arm_name != "lora":
        raise ValueError(f"Unknown arm: {arm_name}")

    lora_config = LoraConfig(
        r=int(config["lora"]["rank"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        bias=str(config["lora"]["bias"]),
        target_modules=target_modules,
    )

    lora_model = get_peft_model(
        base_model,
        lora_config,
    )

    lora_model.to(device)

    lora_base_model = lora_model.get_base_model()

    adapter_trainable = [
        (name, parameter)
        for name, parameter in lora_model.named_parameters()
        if parameter.requires_grad
    ]

    adapter_parameter_count = int(
        sum(
            parameter.numel()
            for _, parameter in adapter_trainable
        )
    )

    unexpected_base_trainables = [
        name
        for name, _ in adapter_trainable
        if ".lora_A." not in name
        and ".lora_B." not in name
    ]

    return {
        "arm_name": arm_name,
        "encoder": lora_base_model,
        "mode_module": lora_model,
        "head": head,
        "adapter_parameter_count": adapter_parameter_count,
        "adapter_trainable_names": [
            name
            for name, _ in adapter_trainable
        ],
        "unexpected_base_trainables": (
            unexpected_base_trainables
        ),
        "selected_target_modules": target_modules,
    }


def train_one_epoch(
    arm: dict,
    train_records: list[dict[str, str]],
    train_labels: np.ndarray,
    class_weights: torch.Tensor,
    image_processor,
    device: torch.device,
    padding_fraction: float,
    physical_batch_size: int,
    gradient_accumulation_steps: int,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    seed: int,
    max_grad_norm: float,
) -> dict:
    arm_name = arm["arm_name"]
    encoder = arm["encoder"]
    mode_module = arm["mode_module"]
    head = arm["head"]

    if arm_name == "frozen_head":
        encoder.eval()
    else:
        mode_module.train()

    head.train()
    optimizer.zero_grad(set_to_none=True)

    order = np.random.default_rng(
        seed + epoch
    ).permutation(len(train_records))

    total_loss = 0.0
    total_samples = 0
    optimizer_steps = 0
    nonfinite_gradient_detected = False

    batch_counter = 0

    for start in range(
        0,
        len(train_records),
        physical_batch_size,
    ):
        end = min(
            start + physical_batch_size,
            len(train_records),
        )

        batch_indices = order[start:end]
        batch_records = [
            train_records[int(index)]
            for index in batch_indices
        ]

        batch_targets = torch.tensor(
            train_labels[batch_indices],
            dtype=torch.long,
            device=device,
        )

        model_inputs = make_model_inputs(
            records=batch_records,
            image_processor=image_processor,
            device=device,
            padding_fraction=padding_fraction,
        )

        if arm_name == "frozen_head":
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    features = encoder.get_image_features(
                        **model_inputs,
                    )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = head(features)
                raw_loss = F.cross_entropy(
                    logits,
                    batch_targets,
                    weight=class_weights,
                )
        else:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                features = encoder.get_image_features(
                    **model_inputs,
                )

                logits = head(features)

                raw_loss = F.cross_entropy(
                    logits,
                    batch_targets,
                    weight=class_weights,
                )

        scaled_loss = (
            raw_loss
            / float(gradient_accumulation_steps)
        )

        scaler.scale(scaled_loss).backward()

        total_loss += (
            float(raw_loss.detach().float().cpu().item())
            * len(batch_records)
        )

        total_samples += len(batch_records)
        batch_counter += 1

        is_final_batch = end == len(train_records)
        should_step = (
            batch_counter % gradient_accumulation_steps == 0
            or is_final_batch
        )

        if should_step:
            scaler.unscale_(optimizer)

            trainable_modules = [head]

            if arm_name == "lora":
                trainable_modules.append(mode_module)

            gradients_finite = all_trainable_gradients_finite(
                trainable_modules
            )

            if not gradients_finite:
                nonfinite_gradient_detected = True
                raise FloatingPointError(
                    "Non-finite or missing trainable gradient "
                    f"detected in arm={arm_name}, epoch={epoch}."
                )

            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.requires_grad
                ],
                max_norm=max_grad_norm,
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    return {
        "train_weighted_cross_entropy": (
            total_loss / float(total_samples)
        ),
        "optimizer_steps": optimizer_steps,
        "nonfinite_gradient_detected": (
            nonfinite_gradient_detected
        ),
    }


def run_arm(
    arm_name: str,
    run_dir: Path,
    config: dict,
    model_id: str,
    cache_root: Path,
    target_modules: list[str],
    initial_head_state: dict,
    train_records: list[dict[str, str]],
    dev_records: list[dict[str, str]],
    y_train: np.ndarray,
    y_dev: np.ndarray,
    class_weights: torch.Tensor,
    class_weight_map: dict[str, float],
    image_processor,
    device: torch.device,
    target_attribute_ids: list[int],
    target_attribute_names: dict[int, str],
) -> dict:
    arm_dir = run_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=False)

    arm = build_arm(
        arm_name=arm_name,
        model_id=model_id,
        cache_root=cache_root,
        initial_head_state=initial_head_state,
        config=config,
        target_modules=target_modules,
        device=device,
    )

    head = arm["head"]
    mode_module = arm["mode_module"]

    if arm_name == "frozen_head":
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(head.parameters()),
                    "lr": float(
                        config["training"]["head_learning_rate"]
                    ),
                    "weight_decay": float(
                        config["training"]["weight_decay"]
                    ),
                }
            ]
        )
    else:
        adapter_parameters = [
            parameter
            for parameter in mode_module.parameters()
            if parameter.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": adapter_parameters,
                    "lr": float(
                        config["training"]["lora_learning_rate"]
                    ),
                    "weight_decay": float(
                        config["training"]["weight_decay"]
                    ),
                },
                {
                    "params": list(head.parameters()),
                    "lr": float(
                        config["training"]["head_learning_rate"]
                    ),
                    "weight_decay": float(
                        config["training"]["weight_decay"]
                    ),
                },
            ]
        )

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=True,
        )

    epochs = int(config["training"]["epochs"])
    batch_size = int(
        config["training"]["physical_batch_size"]
    )
    gradient_accumulation_steps = int(
        config["training"]["gradient_accumulation_steps"]
    )
    seed = int(config["training"]["seed"])
    padding_fraction = float(
        config["crop"]["padding_fraction"]
    )
    max_grad_norm = float(
        config["training"]["max_grad_norm"]
    )
    ece_bins = int(config["evaluation"]["ece_bins"])

    epoch_rows: list[dict] = []
    best_metrics = None
    best_epoch = None
    best_artifacts = None
    best_per_class = None
    best_predictions = None
    best_extra = None

    print(
        f"===== ARM START: {arm_name} =====",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_result = train_one_epoch(
            arm=arm,
            train_records=train_records,
            train_labels=y_train,
            class_weights=class_weights,
            image_processor=image_processor,
            device=device,
            padding_fraction=padding_fraction,
            physical_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            seed=seed,
            max_grad_norm=max_grad_norm,
        )

        (
            dev_metrics,
            per_class_rows,
            prediction_rows,
            extra,
        ) = evaluate(
            encoder=arm["encoder"],
            mode_module=mode_module,
            head=head,
            records=dev_records,
            labels=y_dev,
            class_weights=class_weights,
            target_attribute_ids=target_attribute_ids,
            target_attribute_names=target_attribute_names,
            image_processor=image_processor,
            device=device,
            padding_fraction=padding_fraction,
            batch_size=batch_size,
            ece_bins=ece_bins,
        )

        epoch_elapsed = (
            time.perf_counter() - epoch_start
        )

        epoch_row = {
            "epoch": epoch,
            "elapsed_seconds": round(epoch_elapsed, 3),
            **train_result,
            **dev_metrics,
        }

        epoch_rows.append(epoch_row)

        print(
            "epoch_summary "
            f"arm={arm_name} "
            f"epoch={epoch} "
            f"train_loss={train_result['train_weighted_cross_entropy']:.6f} "
            f"dev_macro_f1={dev_metrics['macro_f1']:.6f} "
            f"dev_balanced_accuracy={dev_metrics['balanced_accuracy']:.6f} "
            f"dev_ece={dev_metrics['top_label_ece']:.6f} "
            f"dev_nll={dev_metrics['negative_log_likelihood']:.6f}",
            flush=True,
        )

        is_better = False

        if best_metrics is None:
            is_better = True
        elif (
            dev_metrics["macro_f1"]
            > best_metrics["macro_f1"] + 1e-12
        ):
            is_better = True
        elif (
            abs(
                dev_metrics["macro_f1"]
                - best_metrics["macro_f1"]
            )
            <= 1e-12
            and dev_metrics["negative_log_likelihood"]
            < best_metrics["negative_log_likelihood"]
        ):
            is_better = True

        if is_better:
            best_epoch = epoch
            best_metrics = dict(dev_metrics)
            best_per_class = per_class_rows
            best_predictions = prediction_rows
            best_extra = extra

            best_artifacts = save_best_checkpoint(
                arm_name=arm_name,
                arm_dir=arm_dir,
                mode_module=mode_module,
                head=head,
                epoch=epoch,
                metrics=dev_metrics,
            )

            print(
                f"best_checkpoint_updated arm={arm_name} "
                f"epoch={epoch} "
                f"macro_f1={dev_metrics['macro_f1']:.6f}",
                flush=True,
            )

    if best_epoch is None:
        raise RuntimeError(
            f"No best epoch selected for arm={arm_name}"
        )

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
        rows=best_extra["confusion_rows"],
    )

    write_json(
        ece_path,
        {
            "top_label_ece": (
                best_metrics["top_label_ece"]
            ),
            "bin_count": int(
                config["evaluation"]["ece_bins"]
            ),
            "bins": best_extra["ece_rows"],
        },
    )

    expected_head_parameters = (
        int(config["head"]["input_dimension"])
        * int(config["head"]["output_classes"])
        + int(config["head"]["output_classes"])
    )

    checks = {
        "best_epoch_selected": best_epoch is not None,
        "finite_selected_macro_f1": bool(
            np.isfinite(best_metrics["macro_f1"])
        ),
        "finite_selected_nll": bool(
            np.isfinite(
                best_metrics["negative_log_likelihood"]
            )
        ),
        "all_epoch_optimizer_steps_positive": all(
            row["optimizer_steps"] > 0
            for row in epoch_rows
        ),
        "no_nonfinite_gradients": not any(
            row["nonfinite_gradient_detected"]
            for row in epoch_rows
        ),
        "head_parameter_count_expected": (
            count_parameters(head)
            == expected_head_parameters
        ),
        "best_prediction_count_matches_dev": (
            len(best_predictions) == len(dev_records)
        ),
    }

    if arm_name == "frozen_head":
        checks["frozen_adapter_parameter_count_zero"] = (
            arm["adapter_parameter_count"] == 0
        )
    else:
        checks["lora_adapter_parameter_count_matches"] = (
            arm["adapter_parameter_count"]
            == int(
                config["lora"][
                    "expected_adapter_parameter_count"
                ]
            )
        )

        checks["no_unexpected_base_trainables"] = (
            len(arm["unexpected_base_trainables"]) == 0
        )

        checks["lora_target_count_matches"] = (
            len(arm["selected_target_modules"])
            == int(
                config["lora"][
                    "expected_target_module_count"
                ]
            )
        )

    status = "PASS" if all(checks.values()) else "REVIEW"

    summary = {
        "status": status,
        "arm_name": arm_name,
        "selected_epoch": int(best_epoch),
        "selected_metrics": best_metrics,
        "selection_policy": config["selection"],
        "class_weight_map": class_weight_map,
        "parameter_counts": {
            "classification_head": count_parameters(head),
            "lora_adapter": arm["adapter_parameter_count"],
            "total_trainable": (
                count_parameters(head)
                + arm["adapter_parameter_count"]
            ),
        },
        "lora_scope": {
            "selected_target_modules": (
                arm["selected_target_modules"]
            ),
            "unexpected_base_trainables": (
                arm["unexpected_base_trainables"]
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
            **best_artifacts,
        },
    }

    write_json(summary_path, summary)

    print(
        f"===== ARM COMPLETE: {arm_name} =====",
        flush=True,
    )

    print(
        f"selected_epoch={best_epoch} "
        f"macro_f1={best_metrics['macro_f1']:.6f} "
        f"balanced_accuracy={best_metrics['balanced_accuracy']:.6f} "
        f"ece={best_metrics['top_label_ece']:.6f}",
        flush=True,
    )

    del arm
    del optimizer
    del scaler
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    return {
        "summary": summary,
        "summary_path": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--manifest-summary", required=True)
    parser.add_argument("--hog-config", required=True)
    parser.add_argument("--lora-audit-json", required=True)
    parser.add_argument("--lora-preflight-json", required=True)
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
    lora_audit_path = Path(args.lora_audit_json).resolve()
    lora_preflight_path = Path(
        args.lora_preflight_json
    ).resolve()
    config_path = Path(args.config).resolve()
    cache_root = Path(args.cache_dir).resolve()
    run_dir = Path(args.run_dir).resolve()

    required_paths = (
        train_manifest_path,
        dev_manifest_path,
        manifest_summary_path,
        hog_config_path,
        lora_audit_path,
        lora_preflight_path,
        config_path,
    )

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact missing: {path}"
            )

    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite run directory: {run_dir}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in the active environment."
        )

    device = torch.device("cuda")

    set_seed(20260701)

    run_dir.mkdir(parents=True, exist_ok=False)
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest_summary = load_json(manifest_summary_path)
    hog_config = load_json(hog_config_path)
    lora_audit = load_json(lora_audit_path)
    lora_preflight = load_json(lora_preflight_path)
    config = load_json(config_path)

    if manifest_summary.get("status") != "PASS":
        raise RuntimeError(
            "Source manifest summary is not PASS."
        )

    if not manifest_summary.get(
        "checks",
        {},
    ).get(
        "group_disjoint_pass",
        False,
    ):
        raise RuntimeError(
            "Source manifest failed group-disjoint check."
        )

    if lora_audit.get("status") != "PASS":
        raise RuntimeError(
            "LoRA target audit is not PASS."
        )

    if lora_preflight.get("status") != "PASS":
        raise RuntimeError(
            "LoRA matched-head preflight is not PASS."
        )

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
            "Detected train/dev group overlap: "
            f"{len(group_overlap)}"
        )

    target_attribute_ids = [
        int(value)
        for value in config["target_attribute_ids"]
    ]

    target_attribute_set = set(target_attribute_ids)

    if len(target_attribute_set) != len(
        target_attribute_ids
    ):
        raise RuntimeError(
            "Configured target attributes are not unique."
        )

    target_attribute_names = {
        int(attribute_id): str(attribute_name)
        for attribute_id, attribute_name in config[
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
            "Train labels do not match configured classes."
        )

    if dev_attribute_ids != target_attribute_set:
        raise RuntimeError(
            "Dev labels do not match configured classes."
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

    class_weights, class_weight_map = build_class_weights(
        labels=y_train,
        class_count=len(target_attribute_ids),
        device=device,
    )

    raw_audit_targets = lora_audit[
        "recommended_lora_scope"
    ]["full_target_module_paths"]

    lora_target_modules = sorted(
        path
        for path in raw_audit_targets
        if ENCODER_SELF_ATTN_PATTERN.fullmatch(path)
    )

    excluded_targets = sorted(
        set(raw_audit_targets)
        - set(lora_target_modules)
    )

    expected_target_count = int(
        config["lora"]["expected_target_module_count"]
    )

    if len(lora_target_modules) != expected_target_count:
        raise RuntimeError(
            "LoRA target count mismatch: "
            f"{len(lora_target_modules)} vs "
            f"{expected_target_count}"
        )

    if excluded_targets != [
        str(config["lora"]["excluded_module"])
    ]:
        raise RuntimeError(
            "Unexpected LoRA excluded target set: "
            f"{excluded_targets}"
        )

    padding_fraction = float(
        hog_config["crop"]["padding_fraction"]
    )

    if abs(
        padding_fraction
        - float(config["crop"]["padding_fraction"])
    ) > 1e-12:
        raise RuntimeError(
            "Crop padding mismatch between HOG and "
            "matched-head configuration."
        )

    image_processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        cache_dir=str(cache_root),
        use_fast=False,
    )

    set_seed(
        int(config["head"]["initialization_seed"])
    )

    template_head = torch.nn.Linear(
        int(config["head"]["input_dimension"]),
        int(config["head"]["output_classes"]),
    )

    initial_head_state = cpu_state_dict(template_head)

    del template_head

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print("===== SIGLIP2 MATCHED-HEAD DEVELOPMENT PRE-FLIGHT =====")
    print(f"TRAIN_SAMPLES: {len(train_records)}")
    print(f"DEV_SAMPLES: {len(dev_records)}")
    print(f"TRAIN_GROUPS: {len(train_groups)}")
    print(f"DEV_GROUPS: {len(dev_groups)}")
    print(f"GROUP_OVERLAP_COUNT: {len(group_overlap)}")
    print(f"MODEL_ID: {args.model_id}")
    print(f"DEVICE: {device}")
    print(
        "PHYSICAL_BATCH_SIZE: "
        f"{config['training']['physical_batch_size']}"
    )
    print(
        "GRADIENT_ACCUMULATION_STEPS: "
        f"{config['training']['gradient_accumulation_steps']}"
    )
    print(
        "EFFECTIVE_BATCH_SIZE: "
        f"{config['training']['effective_batch_size']}"
    )
    print(f"EPOCHS: {config['training']['epochs']}")
    print(
        "LORA_TARGET_MODULES: "
        f"{len(lora_target_modules)}"
    )

    frozen_result = run_arm(
        arm_name="frozen_head",
        run_dir=run_dir,
        config=config,
        model_id=args.model_id,
        cache_root=cache_root,
        target_modules=lora_target_modules,
        initial_head_state=initial_head_state,
        train_records=train_records,
        dev_records=dev_records,
        y_train=y_train,
        y_dev=y_dev,
        class_weights=class_weights,
        class_weight_map=class_weight_map,
        image_processor=image_processor,
        device=device,
        target_attribute_ids=target_attribute_ids,
        target_attribute_names=target_attribute_names,
    )

    lora_result = run_arm(
        arm_name="lora",
        run_dir=run_dir,
        config=config,
        model_id=args.model_id,
        cache_root=cache_root,
        target_modules=lora_target_modules,
        initial_head_state=initial_head_state,
        train_records=train_records,
        dev_records=dev_records,
        y_train=y_train,
        y_dev=y_dev,
        class_weights=class_weights,
        class_weight_map=class_weight_map,
        image_processor=image_processor,
        device=device,
        target_attribute_ids=target_attribute_ids,
        target_attribute_names=target_attribute_names,
    )

    frozen_summary = frozen_result["summary"]
    lora_summary = lora_result["summary"]

    frozen_metrics = frozen_summary["selected_metrics"]
    lora_metrics = lora_summary["selected_metrics"]

    comparison = {
        metric: {
            "frozen_head": frozen_metrics[metric],
            "lora": lora_metrics[metric],
            "lora_minus_frozen_head": (
                lora_metrics[metric]
                - frozen_metrics[metric]
            ),
        }
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "negative_log_likelihood",
            "multiclass_brier",
            "top_label_ece",
            "top_2_accuracy",
        )
    }

    qualification_checks = {
        "source_manifest_pass": (
            manifest_summary.get("status") == "PASS"
        ),
        "group_disjoint_pass": len(group_overlap) == 0,
        "lora_target_audit_pass": (
            lora_audit.get("status") == "PASS"
        ),
        "lora_preflight_pass": (
            lora_preflight.get("status") == "PASS"
        ),
        "frozen_arm_pass": (
            frozen_summary.get("status") == "PASS"
        ),
        "lora_arm_pass": (
            lora_summary.get("status") == "PASS"
        ),
        "same_head_parameter_count": (
            frozen_summary["parameter_counts"][
                "classification_head"
            ]
            == lora_summary["parameter_counts"][
                "classification_head"
            ]
        ),
        "lora_parameter_count_matches_config": (
            lora_summary["parameter_counts"]["lora_adapter"]
            == int(
                config["lora"][
                    "expected_adapter_parameter_count"
                ]
            )
        ),
        "finite_comparison_macro_f1": bool(
            np.isfinite(
                comparison["macro_f1"][
                    "lora_minus_frozen_head"
                ]
            )
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
            "siglip2_lora_matched_head_development_run_v1"
        ),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "selection_scope": "train_only_development",
        "model_id": args.model_id,
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
        "source_artifacts": {
            "train_manifest": str(train_manifest_path),
            "dev_manifest": str(dev_manifest_path),
            "manifest_summary": str(manifest_summary_path),
            "hog_config": str(hog_config_path),
            "lora_target_audit": str(lora_audit_path),
            "lora_preflight": str(lora_preflight_path),
            "config": str(config_path),
            "train_manifest_sha256": sha256_file(
                train_manifest_path
            ),
            "dev_manifest_sha256": sha256_file(
                dev_manifest_path
            ),
            "manifest_summary_sha256": sha256_file(
                manifest_summary_path
            ),
            "hog_config_sha256": sha256_file(
                hog_config_path
            ),
            "lora_target_audit_sha256": sha256_file(
                lora_audit_path
            ),
            "lora_preflight_sha256": sha256_file(
                lora_preflight_path
            ),
            "config_sha256": sha256_file(config_path),
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
        "frozen_head_arm": frozen_summary,
        "lora_arm": lora_summary,
        "comparison": comparison,
        "qualification_checks": qualification_checks,
        "elapsed_seconds": round(
            time.perf_counter() - started,
            3,
        ),
        "claim_boundary": config["claim_boundary"],
    }

    summary_path = run_dir / "run_summary.json"
    write_json(summary_path, summary)

    print("")
    print("===== SIGLIP2 MATCHED-HEAD DEVELOPMENT SUMMARY =====")
    print(f"STATUS: {status}")
    print(
        "FROZEN_SELECTED_EPOCH: "
        f"{frozen_summary['selected_epoch']}"
    )
    print(
        "LORA_SELECTED_EPOCH: "
        f"{lora_summary['selected_epoch']}"
    )
    print(
        "FROZEN_MACRO_F1: "
        f"{frozen_metrics['macro_f1']:.6f}"
    )
    print(
        "LORA_MACRO_F1: "
        f"{lora_metrics['macro_f1']:.6f}"
    )
    print(
        "LORA_MINUS_FROZEN_MACRO_F1: "
        f"{comparison['macro_f1']['lora_minus_frozen_head']:.6f}"
    )
    print(
        "FROZEN_BALANCED_ACCURACY: "
        f"{frozen_metrics['balanced_accuracy']:.6f}"
    )
    print(
        "LORA_BALANCED_ACCURACY: "
        f"{lora_metrics['balanced_accuracy']:.6f}"
    )
    print(
        "LORA_MINUS_FROZEN_BALANCED_ACCURACY: "
        f"{comparison['balanced_accuracy']['lora_minus_frozen_head']:.6f}"
    )
    print(
        "FROZEN_TOP_LABEL_ECE: "
        f"{frozen_metrics['top_label_ece']:.6f}"
    )
    print(
        "LORA_TOP_LABEL_ECE: "
        f"{lora_metrics['top_label_ece']:.6f}"
    )
    print(
        "LORA_MINUS_FROZEN_TOP_LABEL_ECE: "
        f"{comparison['top_label_ece']['lora_minus_frozen_head']:.6f}"
    )
    print(
        "GPU_PEAK_MEMORY_ALLOCATED_MIB: "
        f"{summary['environment']['gpu_peak_memory_allocated_mib']:.2f}"
    )
    print(f"RUN_SUMMARY_JSON: {summary_path}")

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()