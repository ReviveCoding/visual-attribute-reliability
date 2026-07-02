from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import transformers
from transformers import AutoImageProcessor, AutoModel


def load_first_record(manifest_path: Path) -> dict[str, str]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        record = next(csv.DictReader(handle), None)

    if record is None:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    return record


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
            "Non-positive bbox for "
            f"annotation_id={record['annotation_id']}"
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
                "Degenerate crop for "
                f"annotation_id={record['annotation_id']}"
            )

        return image.crop((left, top, right, bottom))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--hog-config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.train_manifest).resolve()
    hog_config_path = Path(args.hog_config).resolve()
    cache_root = Path(args.cache_dir).resolve()
    output_path = Path(args.output_json).resolve()

    for path in (manifest_path, hog_config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file is missing: {path}")

    cache_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with hog_config_path.open("r", encoding="utf-8") as handle:
        hog_config = json.load(handle)

    padding_fraction = float(
        hog_config["crop"]["padding_fraction"]
    )

    record = load_first_record(manifest_path)
    crop = crop_from_record(record, padding_fraction)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in the active venv."
        )

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Image-only path. No AutoProcessor, tokenizer, or SentencePiece.
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        cache_dir=str(cache_root),
        use_fast=False,
    )

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=str(cache_root),
    )

    model.eval()
    model.to(device)

    processor_outputs = image_processor(
        images=[crop],
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

    if features.ndim != 2 or features.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected embedding shape: {features.shape}"
        )

    if not np.isfinite(features).all():
        raise RuntimeError(
            "Embedding contains non-finite values."
        )

    payload = {
        "status": "PASS",
        "mode": "image_only_cuda_forward",
        "model_id": args.model_id,
        "model_class": model.__class__.__name__,
        "image_processor_class": image_processor.__class__.__name__,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_mib": round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
            2,
        ),
        "gpu_peak_memory_allocated_mib": round(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            2,
        ),
        "model_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "manifest_sample": {
            "annotation_id": record["annotation_id"],
            "image_id": record["image_id"],
            "attribute_id": record["attribute_id"],
            "attribute_name": record["attribute_name"],
        },
        "crop": {
            "padding_fraction": padding_fraction,
            "width": crop.width,
            "height": crop.height,
        },
        "model_inputs": {
            name: list(value.shape)
            for name, value in model_inputs.items()
            if hasattr(value, "shape")
        },
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "feature_l2_norm": float(np.linalg.norm(features[0])),
        "finite_feature_values": bool(
            np.isfinite(features).all()
        ),
        "claim_boundary": (
            "Single-crop frozen SigLIP2 image-forward smoke test only. "
            "No full-manifest feature extraction, classifier fitting, "
            "hyperparameter tuning, LoRA training, or validation evaluation."
        ),
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()