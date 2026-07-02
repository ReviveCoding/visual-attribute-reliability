from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def valid_bbox(bbox: object) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False

    try:
        width = float(bbox[2])
        height = float(bbox[3])
    except (TypeError, ValueError):
        return False

    return width > 0.0 and height > 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only Fashionpedia neckline-task audit. "
            "Does not load validation annotations, images, or model code."
        )
    )
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    annotation_path = Path(args.annotation).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (annotation_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    category_names = {
        int(row["id"]): str(row["name"])
        for row in payload.get("categories", [])
    }

    attribute_names = {
        int(row["id"]): str(row["name"])
        for row in payload.get("attributes", [])
    }

    target_category_id = int(config["target_category"]["id"])
    target_category_expected_name = str(
        config["target_category"]["expected_name"]
    )

    observed_category_name = category_names.get(target_category_id)

    if observed_category_name != target_category_expected_name:
        raise ValueError(
            "Target category mismatch. "
            f"Expected '{target_category_expected_name}', "
            f"observed '{observed_category_name}'."
        )

    target_attribute_rows = config["target_attributes"]
    target_attribute_ids = {
        int(row["id"])
        for row in target_attribute_rows
    }

    target_attribute_config_names = {
        int(row["id"]): str(row["name"])
        for row in target_attribute_rows
    }

    missing_attribute_ids = sorted(
        target_attribute_ids - set(attribute_names)
    )

    if missing_attribute_ids:
        raise ValueError(
            "Target attributes are absent from source annotation: "
            f"{missing_attribute_ids}"
        )

    total_target_category_instances = 0
    zero_target_attribute_instances = 0
    multi_target_attribute_instances = 0
    invalid_bbox_instances = 0
    eligible_instances = 0

    eligible_counts = Counter()
    eligible_image_ids = defaultdict(set)
    image_class_counts = defaultdict(Counter)

    for annotation in payload.get("annotations", []):
        if int(annotation["category_id"]) != target_category_id:
            continue

        total_target_category_instances += 1

        active_target_attributes = sorted(
            {
                int(attribute_id)
                for attribute_id in annotation.get("attribute_ids", [])
                if int(attribute_id) in target_attribute_ids
            }
        )

        if len(active_target_attributes) == 0:
            zero_target_attribute_instances += 1
            continue

        if len(active_target_attributes) > 1:
            multi_target_attribute_instances += 1
            continue

        if not valid_bbox(annotation.get("bbox")):
            invalid_bbox_instances += 1
            continue

        attribute_id = active_target_attributes[0]
        image_id = int(annotation["image_id"])

        eligible_instances += 1
        eligible_counts[attribute_id] += 1
        eligible_image_ids[attribute_id].add(image_id)
        image_class_counts[image_id][attribute_id] += 1

    class_rows = []

    for attribute_id in sorted(target_attribute_ids):
        class_rows.append(
            {
                "attribute_id": attribute_id,
                "config_name": target_attribute_config_names[attribute_id],
                "source_name": attribute_names[attribute_id],
                "eligible_instance_count": int(
                    eligible_counts[attribute_id]
                ),
                "eligible_image_count": int(
                    len(eligible_image_ids[attribute_id])
                ),
            }
        )

    class_rows.sort(
        key=lambda row: (
            -row["eligible_instance_count"],
            row["attribute_id"],
        )
    )

    thresholds = config["qualification_thresholds"]

    per_class_support_pass = all(
        row["eligible_instance_count"]
        >= int(thresholds["minimum_eligible_instances_per_class"])
        for row in class_rows
    )

    total_support_pass = (
        eligible_instances
        >= int(thresholds["minimum_total_eligible_instances"])
    )

    multi_target_rate = (
        float(multi_target_attribute_instances)
        / float(total_target_category_instances)
        if total_target_category_instances > 0
        else 1.0
    )

    multi_target_pass = (
        multi_target_rate
        <= float(thresholds["maximum_multi_target_rate"])
    )

    multi_class_image_count = sum(
        1
        for counts in image_class_counts.values()
        if len(counts) > 1
    )

    status = (
        "PASS"
        if per_class_support_pass
        and total_support_pass
        and multi_target_pass
        else "REVIEW"
    )

    result = {
        "status": status,
        "scope": "train_only_neckline_task_audit",
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "target_category": {
            "id": target_category_id,
            "name": observed_category_name,
        },
        "target_attributes": class_rows,
        "counts": {
            "target_category_instances": total_target_category_instances,
            "zero_target_attribute_instances": (
                zero_target_attribute_instances
            ),
            "multi_target_attribute_instances": (
                multi_target_attribute_instances
            ),
            "invalid_bbox_instances": invalid_bbox_instances,
            "eligible_instances": eligible_instances,
            "eligible_images": int(len(image_class_counts)),
            "images_with_multiple_target_classes": (
                multi_class_image_count
            ),
        },
        "rates": {
            "multi_target_rate": round(multi_target_rate, 8),
            "eligible_fraction_of_target_category": round(
                float(eligible_instances)
                / float(total_target_category_instances),
                8,
            )
            if total_target_category_instances > 0
            else 0.0,
        },
        "qualification_thresholds": thresholds,
        "qualification_checks": {
            "per_class_support_pass": per_class_support_pass,
            "total_support_pass": total_support_pass,
            "multi_target_rate_pass": multi_target_pass,
        },
        "claim_boundary": config["claim_boundary"],
    }

    output_path = output_dir / "fashionpedia_neckline_train_only_audit_v1.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("===== FASHIONPEDIA NECKLINE V1 TRAIN-ONLY AUDIT =====")
    print(f"STATUS: {result['status']}")
    print(f"TARGET_CATEGORY: {observed_category_name} ({target_category_id})")
    print(
        "TARGET_CATEGORY_INSTANCES: "
        f"{total_target_category_instances}"
    )
    print(f"ELIGIBLE_INSTANCES: {eligible_instances}")
    print(f"ELIGIBLE_IMAGES: {len(image_class_counts)}")
    print(
        "ZERO_TARGET_ATTRIBUTE_INSTANCES: "
        f"{zero_target_attribute_instances}"
    )
    print(
        "MULTI_TARGET_ATTRIBUTE_INSTANCES: "
        f"{multi_target_attribute_instances}"
    )
    print(f"MULTI_TARGET_RATE: {multi_target_rate:.6f}")
    print(f"INVALID_BBOX_INSTANCES: {invalid_bbox_instances}")

    print("\n===== ELIGIBLE CLASS SUPPORT =====")
    print("ID   INSTANCES  IMAGES  ATTRIBUTE")

    for row in class_rows:
        print(
            f'{row["attribute_id"]:>3}  '
            f'{row["eligible_instance_count"]:>9}  '
            f'{row["eligible_image_count"]:>6}  '
            f'{row["config_name"]}'
        )

    print("\n===== QUALIFICATION CHECKS =====")
    for name, passed in result["qualification_checks"].items():
        print(f"{name}: {passed}")

    print(f"\nAUDIT_JSON: {output_path}")

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()