from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def load_first_manifest_record(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        record = next(reader, None)

    if record is None:
        raise RuntimeError(f"Manifest is empty: {path}")

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
            "Non-positive bounding box for "
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
    parser = argparse.ArgumentParser(
        description=(
            "Image-only SigLIP2 runtime preflight. "
            "Does not initialize a text tokenizer."
        )
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--hog-config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    train_manifest_path = Path(args.train_manifest).resolve()
    hog_config_path = Path(args.hog_config).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_path = Path(args.output_json).resolve()

    for path in (train_manifest_path, hog_config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with hog_config_path.open("r", encoding="utf-8") as handle:
        hog_config = json.load(handle)

    padding_fraction = float(
        hog_config["crop"]["padding_fraction"]
    )

    record = load_first_manifest_record(train_manifest_path)
    crop = crop_from_record(record, padding_fraction)

    cuda_available = bool(torch.cuda.is_available())

    if args.require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is unavailable in the active PyTorch environment."
        )

    device = torch.device("cuda" if cuda_available else "cpu")

    # Image-only processor. No tokenizer, GemmaTokenizer, or SentencePiece.
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        cache_dir=str(cache_dir),
    )

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=str(cache_dir),
    )

    model.eval()
    model.to(device)

    inputs = image_processor(
        images=[crop],
        return_tensors="pt",
    )

    model_inputs = {
        key: value.to(device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        image_features = model.get_image_features(
            **model_inputs,
        )

    if cuda_available:
        torch.cuda.synchronize()

    features = (
        image_features.detach()
        .float()
        .cpu()
        .numpy()
    )

    if features.ndim != 2 or features.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected image feature shape: {features.shape}"
        )

    if not np.isfinite(features).all():
        raise RuntimeError(
            "Image features contain non-finite values."
        )

    payload = {
        "status": "PASS",
        "mode": "image_only",
        "model_id": args.model_id,
        "model_class": model.__class__.__name__,
        "image_processor_class": image_processor.__class__.__name__,
        "model_commit_hash": getattr(
            model.config,
            "_commit_hash",
            None,
        ),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if cuda_available
            else None
        ),
        "gpu_capability": (
            list(torch.cuda.get_device_capability(0))
            if cuda_available
            else None
        ),
        "model_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
            )
        ),
        "manifest_sample": {
            "annotation_id": record["annotation_id"],
            "image_id": record["image_id"],
            "attribute_id": record["attribute_id"],
            "attribute_name": record["attribute_name"],
        },
        "crop": {
            "padding_fraction": padding_fraction,
            "crop_width": crop.width,
            "crop_height": crop.height,
        },
        "model_inputs": {
            key: list(value.shape)
            for key, value in model_inputs.items()
            if hasattr(value, "shape")
        },
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "feature_l2_norm": float(
            np.linalg.norm(features[0])
        ),
        "finite_feature_values": bool(
            np.isfinite(features).all()
        ),
        "claim_boundary": (
            "Image-only runtime and single-crop frozen-encoder "
            "preflight. No tokenizer, full-manifest embedding "
            "extraction, classifier fitting, development metric, "
            "checkpoint selection, or external validation evaluation "
            "was performed."
        ),
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()