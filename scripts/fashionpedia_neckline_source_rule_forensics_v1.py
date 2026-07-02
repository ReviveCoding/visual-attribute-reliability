from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

TARGET_IDS_DEFAULT = [182, 183, 185, 187, 189, 190, 200]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def canonical_pair(annotation_id: Any, attribute_id: Any) -> tuple[int, int]:
    return int(annotation_id), int(attribute_id)


def valid_bbox(annotation: dict[str, Any]) -> bool:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        return float(bbox[2]) > 0.0 and float(bbox[3]) > 0.0
    except (TypeError, ValueError):
        return False


def target_hits(annotation: dict[str, Any], target_ids: set[int]) -> list[int]:
    raw = annotation.get("attribute_ids", [])
    if not isinstance(raw, list):
        return []
    values: set[int] = set()
    for value in raw:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed in target_ids:
            values.add(parsed)
    return sorted(values)


def segmentation_kind(annotation: dict[str, Any]) -> str:
    value = annotation.get("segmentation")
    if value is None:
        return "missing"
    if isinstance(value, dict):
        return "rle"
    if isinstance(value, list):
        if not value:
            return "empty_list"
        return "polygon_list"
    return type(value).__name__


def has_nonempty_segmentation(annotation: dict[str, Any]) -> bool:
    value = annotation.get("segmentation")
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return len(value) > 0
    return False


def annotation_row(
    annotation: dict[str, Any],
    target_ids: set[int],
    category_names: dict[int, str],
) -> dict[str, Any]:
    category_id = int(annotation.get("category_id", -1))
    raw_attributes = annotation.get("attribute_ids", [])
    if not isinstance(raw_attributes, list):
        raw_attributes = []
    bbox = annotation.get("bbox")
    return {
        "annotation_id": int(annotation["id"]),
        "image_id": int(annotation["image_id"]),
        "category_id": category_id,
        "category_name": category_names.get(category_id, "<unknown>"),
        "iscrowd": int(annotation.get("iscrowd", 0)),
        "area": annotation.get("area"),
        "bbox": bbox,
        "bbox_valid": valid_bbox(annotation),
        "segmentation_kind": segmentation_kind(annotation),
        "segmentation_nonempty": has_nonempty_segmentation(annotation),
        "all_attribute_ids": [int(value) for value in raw_attributes if str(value).lstrip("-").isdigit()],
        "target_attribute_hits": target_hits(annotation, target_ids),
    }


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(record.get(key)) for record in records)
    return dict(sorted(counter.items()))


def pair_set_from_annotations(
    annotations: list[dict[str, Any]],
    target_ids: set[int],
    predicate: Callable[[dict[str, Any]], bool],
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for annotation in annotations:
        if not valid_bbox(annotation):
            continue
        hits = target_hits(annotation, target_ids)
        if len(hits) != 1:
            continue
        if predicate(annotation):
            pairs.add(canonical_pair(annotation["id"], hits[0]))
    return pairs


def compare_filter(
    name: str,
    annotations: list[dict[str, Any]],
    target_ids: set[int],
    source_pairs: set[tuple[int, int]],
    predicate: Callable[[dict[str, Any]], bool],
    data_derived: bool,
) -> dict[str, Any]:
    selected = pair_set_from_annotations(annotations, target_ids, predicate)
    missing = source_pairs - selected
    unexpected = selected - source_pairs
    return {
        "name": name,
        "data_derived": data_derived,
        "reconstructed_pair_count": len(selected),
        "missing_source_count": len(missing),
        "unexpected_count": len(unexpected),
        "exact_match": selected == source_pairs,
        "missing_examples": [list(pair) for pair in sorted(missing)[:10]],
        "unexpected_examples": [list(pair) for pair in sorted(unexpected)[:10]],
    }


def selected_config_hints(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = (
        "category",
        "crowd",
        "area",
        "segment",
        "eligible",
        "exclude",
        "allow",
        "filter",
        "attribute",
        "neck",
    )
    results: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        lowered = path.lower()
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                visit(child, child_path)
        elif isinstance(value, list):
            if any(token in lowered for token in tokens):
                results.append({"path": path, "value": value})
            else:
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")
        elif any(token in lowered for token in tokens):
            results.append({"path": path, "value": value})

    visit(payload, "")
    return results[:200]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Forensic source-task audit for Fashionpedia neckline final confirmation. "
            "Does not read validation annotations or images and does not load a model."
        )
    )
    parser.add_argument("--source-train-manifest", required=True)
    parser.add_argument("--source-dev-manifest", required=True)
    parser.add_argument("--source-train-annotation", required=True)
    parser.add_argument("--task-config", required=False)
    parser.add_argument("--task-audit-json", required=False)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    train_manifest_path = Path(args.source_train_manifest).resolve()
    dev_manifest_path = Path(args.source_dev_manifest).resolve()
    train_annotation_path = Path(args.source_train_annotation).resolve()
    output_path = Path(args.output_json).resolve()
    task_config_path = Path(args.task_config).resolve() if args.task_config else None
    task_audit_path = Path(args.task_audit_json).resolve() if args.task_audit_json else None

    for path in (train_manifest_path, dev_manifest_path, train_annotation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source artifact missing: {path}")
    for optional_path in (task_config_path, task_audit_path):
        if optional_path is not None and not optional_path.is_file():
            raise FileNotFoundError(f"Optional source artifact was requested but missing: {optional_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite forensic output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows = load_csv(train_manifest_path)
    dev_rows = load_csv(dev_manifest_path)
    required_columns = {"annotation_id", "image_id", "attribute_id", "split_group_id"}
    for name, rows, path in (
        ("train", train_rows, train_manifest_path),
        ("dev", dev_rows, dev_manifest_path),
    ):
        missing = sorted(required_columns - set(rows[0]))
        if missing:
            raise RuntimeError(f"{name} source manifest missing required columns: {missing}")

    source_rows = train_rows + dev_rows
    source_pairs = {
        canonical_pair(row["annotation_id"], row["attribute_id"])
        for row in source_rows
    }
    if len(source_pairs) != len(source_rows):
        raise RuntimeError("Source manifests contain duplicate annotation/attribute pairs.")

    target_ids = sorted({int(row["attribute_id"]) for row in source_rows})
    if target_ids != TARGET_IDS_DEFAULT:
        raise RuntimeError(
            f"Unexpected source target IDs: {target_ids}; expected={TARGET_IDS_DEFAULT}"
        )
    target_set = set(target_ids)

    payload = load_json(train_annotation_path)
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if not isinstance(annotations, list) or not isinstance(categories, list):
        raise RuntimeError("Fashionpedia train annotation JSON has invalid annotations/categories fields.")
    category_names = {int(row["id"]): str(row.get("name", "")) for row in categories}
    annotation_by_id = {int(row["id"]): row for row in annotations}
    if len(annotation_by_id) != len(annotations):
        raise RuntimeError("Duplicate annotation IDs in Fashionpedia train JSON.")

    baseline_pairs = pair_set_from_annotations(
        annotations,
        target_set,
        lambda annotation: True,
    )
    missing_source_pairs = source_pairs - baseline_pairs
    unexpected_pairs = baseline_pairs - source_pairs

    source_annotation_ids = {pair[0] for pair in source_pairs}
    baseline_annotation_ids = {pair[0] for pair in baseline_pairs}
    if source_annotation_ids - set(annotation_by_id):
        raise RuntimeError("Source manifests reference annotation IDs absent from source train JSON.")

    source_records = [
        annotation_row(annotation_by_id[annotation_id], target_set, category_names)
        for annotation_id in sorted(source_annotation_ids)
    ]
    unexpected_records = [
        annotation_row(annotation_by_id[annotation_id], target_set, category_names)
        for annotation_id, _ in sorted(unexpected_pairs)
    ]

    source_category_ids = {record["category_id"] for record in source_records}

    fixed_field_filters: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("not_crowd", lambda annotation: int(annotation.get("iscrowd", 0)) == 0),
        ("positive_area", lambda annotation: float(annotation.get("area", 0.0)) > 0.0),
        ("nonempty_segmentation", has_nonempty_segmentation),
    ]

    filter_reports: list[dict[str, Any]] = []
    filter_reports.append(
        compare_filter(
            "baseline_exactly_one_target_and_positive_bbox",
            annotations,
            target_set,
            source_pairs,
            lambda annotation: True,
            data_derived=False,
        )
    )

    for size in range(1, len(fixed_field_filters) + 1):
        for combo in itertools.combinations(fixed_field_filters, size):
            name = " AND ".join(item[0] for item in combo)
            filter_reports.append(
                compare_filter(
                    name,
                    annotations,
                    target_set,
                    source_pairs,
                    lambda annotation, combo=combo: all(rule(annotation) for _, rule in combo),
                    data_derived=False,
                )
            )

    filter_reports.append(
        compare_filter(
            "category_id_in_categories_observed_in_source_manifest",
            annotations,
            target_set,
            source_pairs,
            lambda annotation: int(annotation.get("category_id", -1)) in source_category_ids,
            data_derived=True,
        )
    )

    exact_non_data_matches = [
        report["name"]
        for report in filter_reports
        if report["exact_match"] and not report["data_derived"]
    ]
    exact_data_derived_matches = [
        report["name"]
        for report in filter_reports
        if report["exact_match"] and report["data_derived"]
    ]

    config_hints: dict[str, Any] = {}
    for label, optional_path in (("task_config", task_config_path), ("task_audit_json", task_audit_path)):
        if optional_path is None:
            continue
        loaded = load_json(optional_path)
        config_hints[label] = {
            "path": str(optional_path),
            "sha256": sha256_file(optional_path),
            "relevant_keys": selected_config_hints(loaded),
        }

    payload_out = {
        "status": "PASS",
        "schema_version": "fashionpedia_neckline_source_rule_forensics_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source_train_only_forensics_no_validation_access_no_model_loading",
        "validation_annotation_loaded": False,
        "validation_images_scanned": False,
        "model_loaded": False,
        "source": {
            "train_manifest": str(train_manifest_path),
            "dev_manifest": str(dev_manifest_path),
            "train_annotation": str(train_annotation_path),
            "sha256": {
                "train_manifest": sha256_file(train_manifest_path),
                "dev_manifest": sha256_file(dev_manifest_path),
                "train_annotation": sha256_file(train_annotation_path),
            },
        },
        "task": {
            "target_attribute_ids": target_ids,
            "source_manifest_pairs": len(source_pairs),
            "baseline_reconstructed_pairs": len(baseline_pairs),
            "source_pair_reproduction_pass": source_pairs == baseline_pairs,
            "missing_source_pair_count": len(missing_source_pairs),
            "unexpected_pair_count": len(unexpected_pairs),
            "missing_source_examples": [list(pair) for pair in sorted(missing_source_pairs)[:20]],
            "unexpected_pair_examples": [list(pair) for pair in sorted(unexpected_pairs)[:20]],
        },
        "unexpected_pair_forensics": {
            "record_count": len(unexpected_records),
            "records": unexpected_records,
            "category_id_counts": count_by(unexpected_records, "category_id"),
            "category_name_counts": count_by(unexpected_records, "category_name"),
            "iscrowd_counts": count_by(unexpected_records, "iscrowd"),
            "segmentation_kind_counts": count_by(unexpected_records, "segmentation_kind"),
            "segmentation_nonempty_counts": count_by(unexpected_records, "segmentation_nonempty"),
        },
        "source_pair_forensics": {
            "record_count": len(source_records),
            "category_id_counts": count_by(source_records, "category_id"),
            "category_name_counts": count_by(source_records, "category_name"),
            "iscrowd_counts": count_by(source_records, "iscrowd"),
            "segmentation_kind_counts": count_by(source_records, "segmentation_kind"),
            "segmentation_nonempty_counts": count_by(source_records, "segmentation_nonempty"),
            "observed_category_ids": sorted(source_category_ids),
        },
        "candidate_filter_tests": filter_reports,
        "exact_non_data_derived_filter_matches": exact_non_data_matches,
        "exact_data_derived_filter_matches": exact_data_derived_matches,
        "source_config_hints": config_hints,
        "next_action": (
            "Do not evaluate official validation yet. Use the forensic report to identify the exact original source-task "
            "predicate or recover the task-definition configuration, then patch final confirmation to reproduce it before "
            "opening validation annotations."
        ),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload_out, handle, indent=2)

    print("===== FASHIONPEDIA SOURCE-TASK RULE FORENSICS =====", flush=True)
    print("STATUS: PASS", flush=True)
    print(f"SOURCE_MANIFEST_PAIRS: {len(source_pairs)}", flush=True)
    print(f"BASELINE_RECONSTRUCTED_PAIRS: {len(baseline_pairs)}", flush=True)
    print(f"MISSING_SOURCE_PAIRS: {len(missing_source_pairs)}", flush=True)
    print(f"UNEXPECTED_RECONSTRUCTED_PAIRS: {len(unexpected_pairs)}", flush=True)
    print(f"EXACT_NON_DATA_DERIVED_FILTER_MATCHES: {exact_non_data_matches}", flush=True)
    print(f"EXACT_DATA_DERIVED_FILTER_MATCHES: {exact_data_derived_matches}", flush=True)
    print("VALIDATION_ANNOTATION_LOADED: False", flush=True)
    print("VALIDATION_IMAGES_SCANNED: False", flush=True)
    print("MODEL_LOADED: False", flush=True)
    print(f"FORENSICS_JSON: {output_path}", flush=True)


if __name__ == "__main__":
    main()
