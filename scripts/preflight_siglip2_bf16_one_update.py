from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")

    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def crop_from_record(
    record: dict[str, str],
    padding_fraction: float,
) -> Image.Image:
    image_path = Path(record["image_path"])

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")

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


def make_inputs(
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


def trainable_named_parameters(
    head: torch.nn.Module,
    mode_module: torch.nn.Module | None,
) -> list[tuple[str, torch.nn.Parameter]]:
    rows = [
        (f"head.{name}", parameter)
        for name, parameter in head.named_parameters()
        if parameter.requires_grad
    ]

    if mode_module is not None:
        rows.extend(
            (f"adapter.{name}", parameter)
            for name, parameter in mode_module.named_parameters()
            if parameter.requires_grad
        )

    return rows


def gradient_report(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict:
    missing = []
    nonfinite = []
    finite_count = 0
    max_abs_gradient = 0.0

    for name, parameter in named_parameters:
        gradient = parameter.grad

        if gradient is None:
            missing.append(name)
            continue

        if not torch.isfinite(gradient).all().item():
            nonfinite.append(name)
            continue

        finite_count += 1
        max_abs_gradient = max(
            max_abs_gradient,
            float(gradient.detach().abs().max().cpu().item()),
        )

    return {
        "trainable_parameter_tensors": len(named_parameters),
        "finite_gradient_tensors": finite_count,
        "missing_gradient_names": missing,
        "nonfinite_gradient_names": nonfinite,
        "max_abs_gradient": max_abs_gradient,
        "all_finite_and_present": (
            len(missing) == 0
            and len(nonfinite) == 0
            and finite_count == len(named_parameters)
        ),
    }


def snapshot_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in named_parameters
    }


def update_delta(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    before: dict[str, torch.Tensor],
) -> dict:
    changed = []
    max_abs_delta = 0.0

    for name, parameter in named_parameters:
        after = parameter.detach().float().cpu()
        delta = float((after - before[name]).abs().max().item())

        if delta > 0.0:
            changed.append(name)

        max_abs_delta = max(max_abs_delta, delta)

    return {
        "changed_parameter_tensors": len(changed),
        "max_abs_parameter_delta": max_abs_delta,
        "optimizer_update_observed": len(changed) > 0,
    }


def make_class_weights(
    labels: np.ndarray,
    class_count: int,
    device: torch.device,
) -> torch.Tensor:
    counts = np.bincount(labels, minlength=class_count)

    if np.any(counts <= 0):
        raise RuntimeError("At least one target class is absent.")

    total = float(len(labels))
    weights = total / (
        float(class_count)
        * counts.astype(np.float64)
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def run_one_update(
    arm_name: str,
    encoder: torch.nn.Module,
    mode_module: torch.nn.Module | None,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[list[dict[str, str]]],
    batch_labels: list[np.ndarray],
    image_processor,
    class_weights: torch.Tensor,
    device: torch.device,
    padding_fraction: float,
) -> dict:
    if mode_module is None:
        encoder.eval()
    else:
        mode_module.train()

    head.train()
    optimizer.zero_grad(set_to_none=True)

    named_trainable = trainable_named_parameters(
        head=head,
        mode_module=mode_module,
    )

    before = snapshot_parameters(named_trainable)
    losses = []

    for records, labels_np in zip(batches, batch_labels):
        model_inputs = make_inputs(
            records=records,
            image_processor=image_processor,
            device=device,
            padding_fraction=padding_fraction,
        )

        targets = torch.tensor(
            labels_np,
            dtype=torch.long,
            device=device,
        )

        if mode_module is None:
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

        # Keep loss calculation explicitly in float32.
        raw_loss = F.cross_entropy(
            logits.float(),
            targets,
            weight=class_weights,
        )

        if not torch.isfinite(raw_loss).item():
            raise FloatingPointError(
                f"Non-finite loss in arm={arm_name}"
            )

        (raw_loss / float(len(batches))).backward()
        losses.append(float(raw_loss.detach().cpu().item()))

    gradients = gradient_report(named_trainable)

    if not gradients["all_finite_and_present"]:
        return {
            "arm_name": arm_name,
            "losses": losses,
            "gradients": gradients,
            "optimizer_update": {
                "optimizer_update_observed": False,
                "max_abs_parameter_delta": 0.0,
                "changed_parameter_tensors": 0,
            },
        }

    torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in named_trainable],
        max_norm=1.0,
        error_if_nonfinite=True,
    )

    optimizer.step()

    return {
        "arm_name": arm_name,
        "losses": losses,
        "gradients": gradients,
        "optimizer_update": update_delta(
            named_parameters=named_trainable,
            before=before,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--development-config", required=True)
    parser.add_argument("--lora-audit-json", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.train_manifest).resolve()
    config_path = Path(args.development_config).resolve()
    audit_path = Path(args.lora_audit_json).resolve()
    cache_root = Path(args.cache_dir).resolve()
    output_path = Path(args.output_json).resolve()

    for path in (manifest_path, config_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    cache_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "This CUDA device does not report bf16 support."
        )

    config = load_json(config_path)
    audit = load_json(audit_path)
    records = load_manifest(manifest_path)

    if audit.get("status") != "PASS":
        raise RuntimeError("LoRA audit is not PASS.")

    device = torch.device("cuda")
    seed = int(config["training"]["seed"])
    set_seed(seed)

    target_attribute_ids = [
        int(value)
        for value in config["target_attribute_ids"]
    ]

    class_index = {
        attribute_id: index
        for index, attribute_id in enumerate(target_attribute_ids)
    }

    all_labels = np.asarray(
        [
            class_index[int(record["attribute_id"])]
            for record in records
        ],
        dtype=np.int64,
    )

    class_weights = make_class_weights(
        labels=all_labels,
        class_count=len(target_attribute_ids),
        device=device,
    )

    physical_batch_size = int(
        config["training"]["physical_batch_size"]
    )

    accumulation_steps = int(
        config["training"]["gradient_accumulation_steps"]
    )

    probe_sample_count = (
        physical_batch_size
        * accumulation_steps
    )

    # Matches epoch=1 shuffle policy from the failed trainer.
    order = np.random.default_rng(
        seed + 1
    ).permutation(len(records))[:probe_sample_count]

    probe_records = [
        records[int(index)]
        for index in order
    ]

    probe_labels = all_labels[order]

    batches = [
        probe_records[
            index:index + physical_batch_size
        ]
        for index in range(
            0,
            probe_sample_count,
            physical_batch_size,
        )
    ]

    batch_labels = [
        probe_labels[
            index:index + physical_batch_size
        ]
        for index in range(
            0,
            probe_sample_count,
            physical_batch_size,
        )
    ]

    padding_fraction = float(
        config["crop"]["padding_fraction"]
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

    initial_head_state = copy.deepcopy(
        template_head.state_dict()
    )

    del template_head

    raw_targets = audit[
        "recommended_lora_scope"
    ]["full_target_module_paths"]

    target_modules = sorted(
        module_name
        for module_name in raw_targets
        if ENCODER_SELF_ATTN_PATTERN.fullmatch(
            module_name
        )
    )

    expected_targets = int(
        config["lora"]["expected_target_module_count"]
    )

    if len(target_modules) != expected_targets:
        raise RuntimeError(
            "Unexpected LoRA target count: "
            f"{len(target_modules)} vs {expected_targets}"
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    frozen_model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=str(cache_root),
    )

    freeze_module(frozen_model)
    frozen_model.to(device)
    frozen_model.eval()

    frozen_head = torch.nn.Linear(
        int(config["head"]["input_dimension"]),
        int(config["head"]["output_classes"]),
    ).to(device)

    frozen_head.load_state_dict(
        copy.deepcopy(initial_head_state)
    )

    frozen_optimizer = torch.optim.AdamW(
        frozen_head.parameters(),
        lr=float(config["training"]["head_learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    frozen_result = run_one_update(
        arm_name="frozen_head",
        encoder=frozen_model,
        mode_module=None,
        head=frozen_head,
        optimizer=frozen_optimizer,
        batches=batches,
        batch_labels=batch_labels,
        image_processor=image_processor,
        class_weights=class_weights,
        device=device,
        padding_fraction=padding_fraction,
    )

    frozen_encoder_grad_tensors = sum(
        1
        for parameter in frozen_model.parameters()
        if parameter.grad is not None
    )

    del frozen_optimizer
    del frozen_head
    del frozen_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    base_model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=str(cache_root),
    )

    freeze_module(base_model)

    module_map = dict(base_model.named_modules())

    if any(
        target not in module_map
        for target in target_modules
    ):
        raise RuntimeError(
            "Audited LoRA target missing from base model."
        )

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
    lora_encoder = lora_model.get_base_model()

    lora_head = torch.nn.Linear(
        int(config["head"]["input_dimension"]),
        int(config["head"]["output_classes"]),
    ).to(device)

    lora_head.load_state_dict(
        copy.deepcopy(initial_head_state)
    )

    adapter_parameters = [
        parameter
        for parameter in lora_model.parameters()
        if parameter.requires_grad
    ]

    lora_optimizer = torch.optim.AdamW(
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
                "params": lora_head.parameters(),
                "lr": float(
                    config["training"]["head_learning_rate"]
                ),
                "weight_decay": float(
                    config["training"]["weight_decay"]
                ),
            },
        ]
    )

    lora_result = run_one_update(
        arm_name="lora",
        encoder=lora_encoder,
        mode_module=lora_model,
        head=lora_head,
        optimizer=lora_optimizer,
        batches=batches,
        batch_labels=batch_labels,
        image_processor=image_processor,
        class_weights=class_weights,
        device=device,
        padding_fraction=padding_fraction,
    )

    adapter_parameter_count = int(
        sum(parameter.numel() for parameter in adapter_parameters)
    )

    unexpected_base_trainables = [
        name
        for name, parameter in lora_model.named_parameters()
        if parameter.requires_grad
        and ".lora_A." not in name
        and ".lora_B." not in name
    ]

    torch.cuda.synchronize(device)

    expected_adapter_parameters = int(
        config["lora"][
            "expected_adapter_parameter_count"
        ]
    )

    checks = {
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": bool(
            torch.cuda.is_bf16_supported()
        ),
        "selected_lora_targets_match": (
            len(target_modules) == expected_targets
        ),
        "adapter_parameter_count_matches": (
            adapter_parameter_count
            == expected_adapter_parameters
        ),
        "frozen_gradients_finite_and_present": (
            frozen_result["gradients"][
                "all_finite_and_present"
            ]
        ),
        "frozen_optimizer_update_observed": (
            frozen_result["optimizer_update"][
                "optimizer_update_observed"
            ]
        ),
        "frozen_encoder_has_no_grads": (
            frozen_encoder_grad_tensors == 0
        ),
        "lora_gradients_finite_and_present": (
            lora_result["gradients"][
                "all_finite_and_present"
            ]
        ),
        "lora_optimizer_update_observed": (
            lora_result["optimizer_update"][
                "optimizer_update_observed"
            ]
        ),
        "no_unexpected_base_trainables": (
            len(unexpected_base_trainables) == 0
        ),
    }

    status = "PASS" if all(checks.values()) else "REVIEW"

    payload = {
        "status": status,
        "schema_version": (
            "siglip2_bf16_one_update_preflight_v1"
        ),
        "precision_policy": {
            "autocast_dtype": "bfloat16",
            "grad_scaler_used": False,
            "loss_dtype": "float32",
            "reason": (
                "Exact one-update numerical qualification "
                "before a full development run."
            ),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
            "bf16_supported": bool(
                torch.cuda.is_bf16_supported()
            ),
            "peak_gpu_memory_mib": round(
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 2),
                2,
            ),
        },
        "probe": {
            "physical_batch_size": physical_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": probe_sample_count,
            "annotation_ids": [
                record["annotation_id"]
                for record in probe_records
            ],
        },
        "frozen_head": {
            **frozen_result,
            "encoder_gradient_tensor_count": (
                frozen_encoder_grad_tensors
            ),
        },
        "lora": {
            **lora_result,
            "selected_target_count": len(target_modules),
            "adapter_parameter_count": adapter_parameter_count,
            "unexpected_base_trainables": (
                unexpected_base_trainables
            ),
        },
        "checks": checks,
        "claim_boundary": (
            "Numerical one-update preflight only. No full epoch, "
            "checkpoint selection, development metric, LoRA comparison, "
            "or Fashionpedia validation evaluation was performed."
        ),
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("===== SIGLIP2 BF16 ONE-UPDATE PREFLIGHT =====")
    print(f"STATUS: {status}")
    print(
        "BF16_SUPPORTED: "
        f"{payload['environment']['bf16_supported']}"
    )
    print(
        "FROZEN_GRADIENTS_FINITE_AND_PRESENT: "
        f"{checks['frozen_gradients_finite_and_present']}"
    )
    print(
        "FROZEN_OPTIMIZER_UPDATE_OBSERVED: "
        f"{checks['frozen_optimizer_update_observed']}"
    )
    print(
        "LORA_GRADIENTS_FINITE_AND_PRESENT: "
        f"{checks['lora_gradients_finite_and_present']}"
    )
    print(
        "LORA_OPTIMIZER_UPDATE_OBSERVED: "
        f"{checks['lora_optimizer_update_observed']}"
    )
    print(
        "FROZEN_MISSING_GRADS: "
        f"{len(frozen_result['gradients']['missing_gradient_names'])}"
    )
    print(
        "FROZEN_NONFINITE_GRADS: "
        f"{len(frozen_result['gradients']['nonfinite_gradient_names'])}"
    )
    print(
        "LORA_MISSING_GRADS: "
        f"{len(lora_result['gradients']['missing_gradient_names'])}"
    )
    print(
        "LORA_NONFINITE_GRADS: "
        f"{len(lora_result['gradients']['nonfinite_gradient_names'])}"
    )
    print(f"RESULT_JSON: {output_path}")

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()