from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import transformers
from transformers import AutoImageProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Explicitly pin slow processor behavior.
    # No tokenizer, AutoProcessor, model weights, CUDA, or image forward pass.
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        use_fast=False,
    )

    payload = {
        "status": "PASS",
        "model_id": args.model_id,
        "transformers_version": transformers.__version__,
        "image_processor_class": image_processor.__class__.__name__,
        "python_executable": sys.executable,
        "claim_boundary": (
            "Image-processor initialization smoke test only. "
            "No tokenizer, model loading, GPU forward pass, embedding "
            "extraction, classifier fitting, or Fashionpedia evaluation."
        ),
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()