from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers
from transformers import AutoModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    cache_root = Path(args.cache_dir).resolve()
    output_path = Path(args.output_json).resolve()

    cache_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=str(cache_root),
    )

    all_linear_modules = []
    vision_linear_modules = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        row = {
            "name": name,
            "leaf_name": name.split(".")[-1],
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "has_bias": module.bias is not None,
        }

        all_linear_modules.append(row)

        if name.startswith("vision_model."):
            vision_linear_modules.append(row)

    candidate_leaf_names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "o_proj",
    }

    vision_attention_candidates = [
        row
        for row in vision_linear_modules
        if row["leaf_name"] in candidate_leaf_names
    ]

    candidate_leaf_counts = Counter(
        row["leaf_name"]
        for row in vision_attention_candidates
    )

    full_target_paths = [
        row["name"]
        for row in vision_attention_candidates
    ]

    expected_projection_names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    }

    observed_projection_names = set(candidate_leaf_counts)

    target_audit_pass = (
        len(full_target_paths) > 0
        and expected_projection_names.issubset(
            observed_projection_names
        )
    )

    payload = {
        "status": "PASS" if target_audit_pass else "REVIEW",
        "model_id": args.model_id,
        "model_class": model.__class__.__name__,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "model_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "linear_module_inventory": {
            "all_linear_module_count": len(all_linear_modules),
            "vision_linear_module_count": len(vision_linear_modules),
            "vision_attention_candidate_count": (
                len(vision_attention_candidates)
            ),
            "vision_attention_candidate_leaf_counts": dict(
                sorted(candidate_leaf_counts.items())
            ),
        },
        "recommended_lora_scope": {
            "scope": "vision_attention_only",
            "full_target_module_paths": full_target_paths,
            "target_module_count": len(full_target_paths),
            "selection_rule": (
                "All torch.nn.Linear modules under vision_model whose "
                "leaf name is q_proj, k_proj, v_proj, or out_proj."
            ),
        },
        "sample_vision_attention_modules": (
            vision_attention_candidates[:20]
        ),
        "qualification_checks": {
            "has_vision_linear_modules": (
                len(vision_linear_modules) > 0
            ),
            "has_attention_candidates": (
                len(vision_attention_candidates) > 0
            ),
            "has_q_proj": "q_proj" in observed_projection_names,
            "has_k_proj": "k_proj" in observed_projection_names,
            "has_v_proj": "v_proj" in observed_projection_names,
            "has_out_proj": "out_proj" in observed_projection_names,
        },
        "claim_boundary": (
            "Architecture and target-module audit only. "
            "No PEFT package installation, adapter injection, "
            "GPU training, embedding extraction, classifier fitting, "
            "or Fashionpedia evaluation was performed."
        ),
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()