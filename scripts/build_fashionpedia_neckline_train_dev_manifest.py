from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return False

    return width > 0.0 and height > 0.0 and x >= 0.0 and y >= 0.0


def resolve_image_path(image_root: Path, file_name: str) -> Path:
    normalized = str(file_name).replace("\\", "/")
    relative = Path(normalized)

    candidates = [
        image_root / relative,
        image_root / relative.name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve annotation image under image root. "
        f"file_name={file_name!r}, image_root={image_root}"
    )


def objective(
    counts: Counter[int],
    target_counts: dict[int, float],
    target_total: float,
) -> float:
    class_loss = sum(
        (
            (float(counts[attribute_id]) - target_count)
            / max(target_count, 1.0)
        ) ** 2
        for attribute_id, target_count in target_counts.items()
    )

    total_loss = (
        (float(sum(counts.values())) - target_total)
        / max(target_total, 1.0)
    ) ** 2

    overshoot_loss = sum(
        max(
            float(counts[attribute_id]) - target_count,
            0.0,
        )
        / max(target_count, 1.0)
        for attribute_id, target_count in target_counts.items()
    )

    return class_loss + (0.35 * total_loss) + (0.15 * overshoot_loss)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only, image-group-disjoint Fashionpedia "
            "neckline train/dev manifest."
        )
    )
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    annotation_path = Path(args.annotation).resolve()
    image_root = Path(args.image_root).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (annotation_path, image_root, config_path):
        if not path.exists():
            raise FileNotFoundError(f"Required path is missing: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    target_category_id = int(config["target_category"]["id"])
    target_category_name = str(config["target_category"]["expected_name"])

    target_attribute_rows = config["target_attributes"]
    target_attribute_ids = {
        int(row["id"])
        for row in target_attribute_rows
    }

    target_attribute_names = {
        int(row["id"]): str(row["name"])
        for row in target_attribute_rows
    }

    development_policy = config["development_split_policy"]
    train_fraction = float(development_policy["train_fraction"])
    dev_fraction = float(development_policy["development_fraction"])
    seed = int(development_policy["seed"])

    if abs((train_fraction + dev_fraction) - 1.0) > 1e-9:
        raise ValueError(
            "Train and development fractions must sum to 1.0."
        )

    category_names = {
        int(row["id"]): str(row["name"])
        for row in payload.get("categories", [])
    }

    observed_category_name = category_names.get(target_category_id)

    if observed_category_name != target_category_name:
        raise ValueError(
            "Target category mismatch. "
            f"Expected {target_category_name!r}, "
            f"observed {observed_category_name!r}."
        )

    image_rows = {
        int(row["id"]): str(row["file_name"])
        for row in payload.get("images", [])
    }

    records_by_group: dict[int, list[dict]] = defaultdict(list)
    class_total_counts: Counter[int] = Counter()

    skipped_zero_target = 0
    skipped_multi_target = 0
    skipped_invalid_bbox = 0
    skipped_missing_image = 0

    for annotation in payload.get("annotations", []):
        if int(annotation["category_id"]) != target_category_id:
            continue

        active_attributes = sorted(
            {
                int(attribute_id)
                for attribute_id in annotation.get("attribute_ids", [])
                if int(attribute_id) in target_attribute_ids
            }
        )

        if len(active_attributes) == 0:
            skipped_zero_target += 1
            continue

        if len(active_attributes) > 1:
            skipped_multi_target += 1
            continue

        bbox = annotation.get("bbox")

        if not valid_bbox(bbox):
            skipped_invalid_bbox += 1
            continue

        image_id = int(annotation["image_id"])
        file_name = image_rows.get(image_id)

        if file_name is None:
            skipped_missing_image += 1
            continue

        try:
            image_path = resolve_image_path(image_root, file_name)
        except FileNotFoundError:
            skipped_missing_image += 1
            continue

        attribute_id = active_attributes[0]
        x, y, width, height = (float(value) for value in bbox)

        record = {
            "schema_version": "fashionpedia_neckline_attribute_manifest_v1",
            "annotation_id": int(annotation["id"]),
            "image_id": image_id,
            "split_group_id": image_id,
            "image_path": str(image_path),
            "source_file_name": file_name,
            "target_category_id": target_category_id,
            "target_category_name": observed_category_name,
            "attribute_id": attribute_id,
            "attribute_name": target_attribute_names[attribute_id],
            "bbox_x": x,
            "bbox_y": y,
            "bbox_width": width,
            "bbox_height": height,
        }

        records_by_group[image_id].append(record)
        class_total_counts[attribute_id] += 1

    if not records_by_group:
        raise RuntimeError("No eligible neckline samples were found.")

    rng = random.Random(seed)

    group_ids = list(records_by_group)
    tie_break = {
        group_id: rng.random()
        for group_id in group_ids
    }

    group_class_counts: dict[int, Counter[int]] = {
        group_id: Counter(
            record["attribute_id"]
            for record in records
        )
        for group_id, records in records_by_group.items()
    }

    ordered_group_ids = sorted(
        group_ids,
        key=lambda group_id: (
            -len(records_by_group[group_id]),
            -max(group_class_counts[group_id].values()),
            tie_break[group_id],
        ),
    )

    target_dev_counts = {
        attribute_id: float(class_total_counts[attribute_id]) * dev_fraction
        for attribute_id in sorted(target_attribute_ids)
    }

    target_dev_total = (
        float(sum(class_total_counts.values()))
        * dev_fraction
    )

    dev_group_ids: set[int] = set()
    dev_counts: Counter[int] = Counter()

    for group_id in ordered_group_ids:
        candidate_counts = dev_counts.copy()
        candidate_counts.update(group_class_counts[group_id])

        keep_score = objective(
            dev_counts,
            target_dev_counts,
            target_dev_total,
        )

        add_score = objective(
            candidate_counts,
            target_dev_counts,
            target_dev_total,
        )

        if add_score < keep_score:
            dev_group_ids.add(group_id)
            dev_counts = candidate_counts

    train_group_ids = set(group_ids) - dev_group_ids

    if not train_group_ids or not dev_group_ids:
        raise RuntimeError(
            "Degenerate split produced an empty train or dev partition."
        )

    train_records = [
        record
        for group_id in sorted(train_group_ids)
        for record in records_by_group[group_id]
    ]

    dev_records = [
        record
        for group_id in sorted(dev_group_ids)
        for record in records_by_group[group_id]
    ]

    train_counts = Counter(
        int(record["attribute_id"])
        for record in train_records
    )

    dev_counts = Counter(
        int(record["attribute_id"])
        for record in dev_records
    )

    overlap = train_group_ids & dev_group_ids

    class_rows = []

    for attribute_id in sorted(target_attribute_ids):
        total_count = int(class_total_counts[attribute_id])
        train_count = int(train_counts[attribute_id])
        dev_count = int(dev_counts[attribute_id])

        class_rows.append(
            {
                "attribute_id": attribute_id,
                "attribute_name": target_attribute_names[attribute_id],
                "total_count": total_count,
                "train_count": train_count,
                "dev_count": dev_count,
                "dev_fraction": round(
                    float(dev_count) / float(total_count),
                    6,
                )
                if total_count > 0
                else 0.0,
            }
        )

    overall_dev_fraction = (
        float(len(dev_records))
        / float(len(train_records) + len(dev_records))
    )

    per_class_minimum_train = min(
        row["train_count"]
        for row in class_rows
    )

    per_class_minimum_dev = min(
        row["dev_count"]
        for row in class_rows
    )

    group_disjoint_pass = len(overlap) == 0
    all_classes_in_train_pass = all(
        row["train_count"] > 0
        for row in class_rows
    )

    all_classes_in_dev_pass = all(
        row["dev_count"] > 0
        for row in class_rows
    )

    dev_fraction_pass = (
        abs(overall_dev_fraction - dev_fraction) <= 0.035
    )

    class_fraction_pass = all(
        0.12 <= row["dev_fraction"] <= 0.28
        for row in class_rows
    )

    image_resolution_pass = skipped_missing_image == 0

    status = (
        "PASS"
        if group_disjoint_pass
        and all_classes_in_train_pass
        and all_classes_in_dev_pass
        and dev_fraction_pass
        and class_fraction_pass
        and image_resolution_pass
        else "REVIEW"
    )

    manifest_root = output_dir
    train_csv_path = manifest_root / "train_manifest.csv"
    dev_csv_path = manifest_root / "dev_manifest.csv"
    summary_path = manifest_root / "manifest_summary.json"

    fieldnames = [
        "schema_version",
        "annotation_id",
        "image_id",
        "split_group_id",
        "image_path",
        "source_file_name",
        "target_category_id",
        "target_category_name",
        "attribute_id",
        "attribute_name",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
    ]

    for path, rows in (
        (train_csv_path, train_records),
        (dev_csv_path, dev_records),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "status": status,
        "schema_version": "fashionpedia_neckline_attribute_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "train_only",
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "image_root": str(image_root),
        "task": {
            "category_id": target_category_id,
            "category_name": observed_category_name,
            "target_attribute_ids": sorted(target_attribute_ids),
        },
        "split_policy": {
            "group_key": "image_id",
            "seed": seed,
            "requested_train_fraction": train_fraction,
            "requested_dev_fraction": dev_fraction,
            "implementation": (
                "deterministic greedy group-aware class-balance assignment"
            ),
        },
        "counts": {
            "eligible_samples": len(train_records) + len(dev_records),
            "eligible_images": len(records_by_group),
            "train_samples": len(train_records),
            "dev_samples": len(dev_records),
            "train_images": len(train_group_ids),
            "dev_images": len(dev_group_ids),
            "skipped_zero_target": skipped_zero_target,
            "skipped_multi_target": skipped_multi_target,
            "skipped_invalid_bbox": skipped_invalid_bbox,
            "skipped_missing_image": skipped_missing_image,
        },
        "class_distribution": class_rows,
        "checks": {
            "group_disjoint_pass": group_disjoint_pass,
            "all_classes_in_train_pass": all_classes_in_train_pass,
            "all_classes_in_dev_pass": all_classes_in_dev_pass,
            "dev_fraction_pass": dev_fraction_pass,
            "class_fraction_pass": class_fraction_pass,
            "image_resolution_pass": image_resolution_pass,
            "overall_dev_fraction": round(
                overall_dev_fraction,
                6,
            ),
            "minimum_train_class_count": per_class_minimum_train,
            "minimum_dev_class_count": per_class_minimum_dev,
            "group_overlap_count": len(overlap),
        },
        "artifacts": {
            "train_manifest_csv": str(train_csv_path),
            "dev_manifest_csv": str(dev_csv_path),
        },
        "claim_boundary": (
            "Train-only manifest construction. No validation annotations, "
            "validation images, model training, checkpoint selection, or "
            "final metrics were used."
        ),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("===== FASHIONPEDIA NECKLINE V1 TRAIN/DEV MANIFEST =====")
    print(f"STATUS: {status}")
    print(f"TRAIN_SAMPLES: {len(train_records)}")
    print(f"DEV_SAMPLES: {len(dev_records)}")
    print(f"TRAIN_IMAGES: {len(train_group_ids)}")
    print(f"DEV_IMAGES: {len(dev_group_ids)}")
    print(f"OVERALL_DEV_FRACTION: {overall_dev_fraction:.6f}")
    print(f"GROUP_OVERLAP_COUNT: {len(overlap)}")
    print(f"SKIPPED_MISSING_IMAGE: {skipped_missing_image}")

    print("\n===== CLASS DISTRIBUTION =====")
    print("ID   TOTAL  TRAIN  DEV  DEV_FRACTION  ATTRIBUTE")

    for row in class_rows:
        print(
            f'{row["attribute_id"]:>3}  '
            f'{row["total_count"]:>5}  '
            f'{row["train_count"]:>5}  '
            f'{row["dev_count"]:>4}  '
            f'{row["dev_fraction"]:>12.3f}  '
            f'{row["attribute_name"]}'
        )

    print("\n===== MANIFEST CHECKS =====")
    for name, passed in summary["checks"].items():
        print(f"{name}: {passed}")

    print(f"\nTRAIN_MANIFEST: {train_csv_path}")
    print(f"DEV_MANIFEST: {dev_csv_path}")
    print(f"SUMMARY_JSON: {summary_path}")

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()