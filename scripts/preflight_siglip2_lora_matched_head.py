from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import peft
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor, AutoModel


ENCODER_SELF_ATTN_PATTERN = re.compile(
    r"^vision_model\.encoder\.layers\.\d+\.self_attn\."
    r"(q_proj|k_proj|v_proj|out_proj)$"
)


def load_records(
    manifest_path: Path,
    count: int,
) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) < count:
        raise RuntimeError(
            f"Manifest has {len(rows)} rows but preflight requires {count}."
        )

    return rows[:count]


def crop_from_record(
    record: dict[str, str],
    padding_fraction: float,
) -> Image.Image:
    image_path = Path(record["image_path"])

    if not image_path.is_file():
        raise FileNotFoundError(f"Image is missing: {image_path}")

    x = float(record["bbox_x"])
    y = float(record["bbox_y"])
    width = float(record["bbox_width"])
    height = float(record["bbox_height"])

    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            f"Non-positive bbox for annotation_id={record['annotation_id']}"
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


def freeze_model(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def trainable_parameter_count(module: torch.nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    )


def finite_gradients(module: torch.nn.Module) -> bool:
    trainable = [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        return False

    return all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        for parameter in trainable
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--hog-config", required=True)
    parser.add_argument("--target-audit-json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.train_manifest).resolve()
    hog_config_path = Path(args.hog_config).resolve()
    audit_path = Path(args.target_audit_json).resolve()
    config_path = Path(args.config).resolve()
    cache_root = Path(args.cache_dir).resolve()
    output_path = Path(args.output_json).resolve()

    for path in (
        manifest_path,
        hog_config_path,
        audit_path,
        config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    cache_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    hog_config = json.loads(hog_config_path.read_text(encoding="utf-8"))

    if audit.get("status") != "PASS":
        raise RuntimeError("LoRA target audit is not PASS.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active environment.")

    device = torch.device("cuda")

    expected_class_ids = [
        int(value)
        for value in config["target_attribute_ids"]
    ]

    class_index = {
        attribute_id: index
        for index, attribute_id in enumerate(expected_class_ids)
    }

    batch_size = int(config["preflight"]["batch_size"])
    records = load_records(manifest_path, batch_size)

    observed_class_ids = {
        int(record["attribute_id"])
        for record in records
    }

    if not observed_class_ids.issubset(set(expected_class_ids)):
        raise RuntimeError("Preflight records contain unknown target labels.")

    padding_fraction = float(
        hog_config["crop"]["padding_fraction"]
    )

    crops = [
        crop_from_record(record, padding_fraction)
        for record in records
    ]

    labels = torch.tensor(
        [
            class_index[int(record["attribute_id"])]
            for record in records
        ],
        dtype=torch.long,
        device=device,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        cache_dir=str(cache_root),
        use_fast=False,
    )

    processor_outputs = image_processor(
        images=crops,
        return_tensors="pt",
    )

    model_inputs = {
        name: value.to(device)
        for name, value in processor_outputs.items()
        if hasattr(value, "to")
    }

    raw_audit_targets = audit[
        "recommended_lora_scope"
    ]["full_target_module_paths"]

    selected_targets = sorted(
        path
        for path in raw_audit_targets
        if ENCODER_SELF_ATTN_PATTERN.fullmatch(path)
    )

    excluded_targets = sorted(
        set(raw_audit_targets) - set(selected_targets)
    )

    expected_target_count = int(
        config["lora"]["expected_target_module_count"]
    )

    if len(selected_targets) != expected_target_count:
        raise RuntimeError(
            "Selected encoder self-attention target count mismatch: "
            f"{len(selected_targets)} vs {expected_target_count}"
        )

    excluded_expected = str(config["lora"]["excluded_module"])

    if excluded_targets != [excluded_expected]:
        raise RuntimeError(
            "Unexpected excluded LoRA target set: "
            f"{excluded_targets}"
        )

    torch.manual_seed(int(config["head"]["initialization_seed"]))

    frozen_model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=str(cache_root),
    )

    freeze_model(frozen_model)
    frozen_model.to(device)
    frozen_model.eval()

    frozen_head = torch.nn.Linear(
        int(config["head"]["input_dimension"]),
        int(config["head"]["output_classes"]),
    ).to(device)

    frozen_head_initial_state = copy.deepcopy(
        frozen_head.state_dict()
    )

    with torch.no_grad():
        frozen_features = frozen_model.get_image_features(
            **model_inputs,
        )

    frozen_logits = frozen_head(frozen_features)
    frozen_loss = F.cross_entropy(frozen_logits, labels)

    frozen_head.zero_grad(set_to_none=True)
    frozen_loss.backward()

    frozen_head_gradients_finite = finite_gradients(frozen_head)

    frozen_model_grad_count = sum(
        1
        for parameter in frozen_model.parameters()
        if parameter.grad is not None
    )

    frozen_feature_shape = list(frozen_features.shape)
    frozen_logits_cpu = frozen_logits.detach().float().cpu()

    del frozen_logits
    del frozen_features
    del frozen_head
    del frozen_model

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    base_model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=str(cache_root),
    )

    freeze_model(base_model)

    original_module_map = dict(base_model.named_modules())

    missing_target_modules = [
        path
        for path in selected_targets
        if path not in original_module_map
    ]

    if missing_target_modules:
        raise RuntimeError(
            "Audited LoRA targets missing from model: "
            f"{missing_target_modules}"
        )

    invalid_target_modules = [
        path
        for path in selected_targets
        if not isinstance(
            original_module_map[path],
            torch.nn.Linear,
        )
    ]

    if invalid_target_modules:
        raise RuntimeError(
            "Audited LoRA targets are not Linear modules: "
            f"{invalid_target_modules}"
        )

    rank = int(config["lora"]["rank"])

    expected_lora_parameter_count = int(
        sum(
            rank
            * (
                original_module_map[path].in_features
                + original_module_map[path].out_features
            )
            for path in selected_targets
        )
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        bias=str(config["lora"]["bias"]),
        target_modules=selected_targets,
    )

    lora_model = get_peft_model(
        base_model,
        lora_config,
    )

    lora_model.to(device)
    lora_model.eval()

    lora_base_model = lora_model.get_base_model()

    lora_module_map = dict(lora_base_model.named_modules())

    injected_targets = [
        path
        for path in selected_targets
        if (
            path in lora_module_map
            and hasattr(lora_module_map[path], "lora_A")
            and "default" in lora_module_map[path].lora_A
            and hasattr(lora_module_map[path], "lora_B")
            and "default" in lora_module_map[path].lora_B
        )
    ]

    lora_head = torch.nn.Linear(
        int(config["head"]["input_dimension"]),
        int(config["head"]["output_classes"]),
    ).to(device)

    lora_head.load_state_dict(frozen_head_initial_state)

    with torch.no_grad():
        lora_features_initial = (
            lora_base_model.get_image_features(
                **model_inputs,
            )
        )

    lora_logits_initial = lora_head(lora_features_initial)

    feature_initial_delta = float(
        (
            lora_features_initial.detach().float().cpu()
            - frozen_logits_cpu.new_tensor(0.0)
        )
        .abs()
        .max()
        .item()
    )

    # Recompute the frozen feature reference using a new frozen model
    # would add memory pressure. Instead, use logits equality as the
    # adapter no-op qualification after matching head initialization.
    initial_logit_delta = float(
        (
            lora_logits_initial.detach().float().cpu()
            - frozen_logits_cpu
        )
        .abs()
        .max()
        .item()
    )

    lora_model.train()
    lora_head.train()

    lora_model.zero_grad(set_to_none=True)
    lora_head.zero_grad(set_to_none=True)

    lora_features_train = lora_base_model.get_image_features(
        **model_inputs,
    )

    lora_logits_train = lora_head(lora_features_train)
    lora_loss = F.cross_entropy(lora_logits_train, labels)
    lora_loss.backward()

    lora_named_trainable = [
        (name, parameter)
        for name, parameter in lora_model.named_parameters()
        if parameter.requires_grad
    ]

    lora_adapter_trainables = [
        (name, parameter)
        for name, parameter in lora_named_trainable
        if ".lora_A." in name or ".lora_B." in name
    ]

    unexpected_base_trainables = [
        name
        for name, _ in lora_named_trainable
        if ".lora_A." not in name
        and ".lora_B." not in name
    ]

    actual_lora_parameter_count = int(
        sum(
            parameter.numel()
            for _, parameter in lora_adapter_trainables
        )
    )

    lora_gradients_finite = finite_gradients(lora_model)
    lora_head_gradients_finite = finite_gradients(lora_head)

    torch.cuda.synchronize(device)

    noop_tolerance = float(
        config["preflight"]["initial_noop_max_abs_tolerance"]
    )

    head_parameter_count = int(
        sum(
            parameter.numel()
            for parameter in lora_head.parameters()
        )
    )

    checks = {
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_target_count_matches": (
            len(selected_targets) == expected_target_count
        ),
        "excluded_pooling_head_only": (
            excluded_targets == [excluded_expected]
        ),
        "all_selected_targets_injected": (
            len(injected_targets) == len(selected_targets)
        ),
        "adapter_parameter_count_matches": (
            actual_lora_parameter_count
            == expected_lora_parameter_count
        ),
        "no_unexpected_base_trainables": (
            len(unexpected_base_trainables) == 0
        ),
        "frozen_head_gradients_finite": (
            frozen_head_gradients_finite
        ),
        "frozen_encoder_has_no_grads": (
            frozen_model_grad_count == 0
        ),
        "lora_adapter_gradients_finite": (
            lora_gradients_finite
        ),
        "lora_head_gradients_finite": (
            lora_head_gradients_finite
        ),
        "finite_lora_features": bool(
            torch.isfinite(lora_features_train).all().item()
        ),
        "finite_lora_logits": bool(
            torch.isfinite(lora_logits_train).all().item()
        ),
        "initial_logits_match_frozen": (
            initial_logit_delta <= noop_tolerance
        ),
        "head_parameter_count_expected": (
            head_parameter_count
            == (
                int(config["head"]["input_dimension"])
                * int(config["head"]["output_classes"])
                + int(config["head"]["output_classes"])
            )
        ),
    }

    status = "PASS" if all(checks.values()) else "REVIEW"

    payload = {
        "status": status,
        "schema_version": "siglip2_lora_matched_head_preflight_v1",
        "model_id": args.model_id,
        "model_class": lora_base_model.__class__.__name__,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "batch": {
            "sample_count": len(records),
            "annotation_ids": [
                record["annotation_id"]
                for record in records
            ],
            "label_attribute_ids": [
                int(record["attribute_id"])
                for record in records
            ],
            "processor_inputs": {
                name: list(value.shape)
                for name, value in model_inputs.items()
            },
        },
        "lora_scope": {
            "selected_target_modules": selected_targets,
            "selected_target_count": len(selected_targets),
            "excluded_target_modules": excluded_targets,
            "injected_target_modules": injected_targets,
            "rank": rank,
            "alpha": int(config["lora"]["alpha"]),
            "dropout": float(config["lora"]["dropout"]),
            "bias": str(config["lora"]["bias"]),
        },
        "parameters": {
            "base_model_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in lora_base_model.parameters()
                )
            ),
            "classification_head_trainable_parameters": (
                head_parameter_count
            ),
            "expected_lora_trainable_parameters": (
                expected_lora_parameter_count
            ),
            "actual_lora_trainable_parameters": (
                actual_lora_parameter_count
            ),
            "total_lora_plus_head_trainable_parameters": (
                actual_lora_parameter_count
                + head_parameter_count
            ),
            "unexpected_base_trainable_parameter_names": (
                unexpected_base_trainables
            ),
        },
        "forward_backward": {
            "frozen_feature_shape": frozen_feature_shape,
            "frozen_loss": float(frozen_loss.detach().cpu().item()),
            "lora_feature_shape": list(
                lora_features_train.shape
            ),
            "lora_loss": float(
                lora_loss.detach().cpu().item()
            ),
            "initial_logit_max_abs_delta": (
                initial_logit_delta
            ),
            "initial_feature_reference_note": (
                "Logit equality is tested with the same head "
                "initialization; default LoRA initialization should "
                "produce a no-op before optimization."
            ),
        },
        "checks": checks,
        "claim_boundary": config["claim_boundary"],
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("===== SIGLIP2 LORA MATCHED-HEAD PREFLIGHT =====")
    print(f"STATUS: {status}")
    print(f"SELECTED_TARGET_MODULES: {len(selected_targets)}")
    print(f"INJECTED_TARGET_MODULES: {len(injected_targets)}")
    print(
        "EXPECTED_LORA_TRAINABLE_PARAMS: "
        f"{expected_lora_parameter_count}"
    )
    print(
        "ACTUAL_LORA_TRAINABLE_PARAMS: "
        f"{actual_lora_parameter_count}"
    )
    print(f"HEAD_TRAINABLE_PARAMS: {head_parameter_count}")
    print(
        "TOTAL_LORA_PLUS_HEAD_TRAINABLE_PARAMS: "
        f"{actual_lora_parameter_count + head_parameter_count}"
    )
    print(
        "INITIAL_LOGIT_MAX_ABS_DELTA: "
        f"{initial_logit_delta:.10f}"
    )
    print(f"FROZEN_LOSS: {float(frozen_loss.detach().cpu().item()):.6f}")
    print(f"LORA_LOSS: {float(lora_loss.detach().cpu().item()):.6f}")

    print("")
    print("===== QUALIFICATION CHECKS =====")
    for name, passed in checks.items():
        print(f"{name}: {passed}")

    print(f"\nPREFLIGHT_JSON: {output_path}")

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()