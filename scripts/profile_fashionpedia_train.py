from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only Fashionpedia attribute profiling. "
            "Does not load validation annotations or images."
        )
    )
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    annotation_path = Path(args.annotation).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing train annotation: {annotation_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    category_names = {
        int(row["id"]): str(row["name"])
        for row in payload.get("categories", [])
    }

    attribute_names = {
        int(row["id"]): str(row["name"])
        for row in payload.get("attributes", [])
    }

    category_instance_counts = Counter()
    category_image_ids = defaultdict(set)

    attribute_instance_counts = Counter()
    attribute_image_ids = defaultdict(set)
    attribute_category_counts = defaultdict(Counter)

    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])

        category_instance_counts[category_id] += 1
        category_image_ids[category_id].add(image_id)

        attribute_ids = {
            int(attribute_id)
            for attribute_id in annotation.get("attribute_ids", [])
            if int(attribute_id) >= 0
            and int(attribute_id) in attribute_names
        }

        for attribute_id in attribute_ids:
            attribute_instance_counts[attribute_id] += 1
            attribute_image_ids[attribute_id].add(image_id)
            attribute_category_counts[attribute_id][category_id] += 1

    category_rows = []

    for category_id, category_name in category_names.items():
        category_rows.append(
            {
                "category_id": int(category_id),
                "category_name": category_name,
                "train_instance_count": int(category_instance_counts[category_id]),
                "train_image_count": int(len(category_image_ids[category_id])),
            }
        )

    category_rows.sort(
        key=lambda row: (
            -row["train_instance_count"],
            row["category_name"],
        )
    )

    attribute_rows = []

    for attribute_id, attribute_name in attribute_names.items():
        instance_count = int(attribute_instance_counts[attribute_id])
        image_count = int(len(attribute_image_ids[attribute_id]))
        category_counts = attribute_category_counts[attribute_id]

        top_categories = [
            {
                "category_id": int(category_id),
                "category_name": category_names.get(
                    int(category_id),
                    f"UNKNOWN_{category_id}",
                ),
                "instance_count": int(count),
            }
            for category_id, count in category_counts.most_common(5)
        ]

        dominant_category_count = (
            int(top_categories[0]["instance_count"])
            if top_categories
            else 0
        )

        dominant_category_share = (
            float(dominant_category_count) / float(instance_count)
            if instance_count > 0
            else 0.0
        )

        attribute_rows.append(
            {
                "attribute_id": int(attribute_id),
                "attribute_name": attribute_name,
                "train_instance_count": instance_count,
                "train_image_count": image_count,
                "category_support_count": int(len(category_counts)),
                "dominant_category_share": round(
                    dominant_category_share,
                    6,
                ),
                "top_categories_text": "; ".join(
                    f'{row["category_name"]}:{row["instance_count"]}'
                    for row in top_categories
                ),
                "top_categories": top_categories,
            }
        )

    attribute_rows.sort(
        key=lambda row: (
            -row["train_instance_count"],
            -row["train_image_count"],
            row["attribute_name"],
        )
    )

    support_candidates = [
        row
        for row in attribute_rows
        if row["train_instance_count"] >= 750
        and row["train_image_count"] >= 500
    ]

    result = {
        "status": "PASS",
        "scope": "train_only_attribute_profiling",
        "source_annotation": str(annotation_path),
        "train_image_count": int(len(payload.get("images", []))),
        "train_instance_count": int(len(payload.get("annotations", []))),
        "category_count": int(len(category_names)),
        "attribute_count": int(len(attribute_names)),
        "support_candidate_rule": {
            "minimum_train_instance_count": 750,
            "minimum_train_image_count": 500,
            "note": (
                "Support screening only. Final task attributes will be "
                "selected using train data only, before validation is read."
            ),
        },
        "support_candidate_count": int(len(support_candidates)),
        "top_categories": category_rows,
        "all_attributes": attribute_rows,
        "support_candidates": support_candidates,
        "claim_boundary": (
            "Train-only data profile. No validation labels, images, "
            "model predictions, thresholds, hyperparameters, or metrics used."
        ),
    }

    profile_json = output_dir / "fashionpedia_train_only_attribute_profile.json"
    profile_csv = output_dir / "fashionpedia_train_only_attribute_profile.csv"
    candidates_csv = output_dir / "fashionpedia_train_only_support_candidates.csv"

    with profile_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    fieldnames = [
        "attribute_id",
        "attribute_name",
        "train_instance_count",
        "train_image_count",
        "category_support_count",
        "dominant_category_share",
        "top_categories_text",
    ]

    for output_path, rows in (
        (profile_csv, attribute_rows),
        (candidates_csv, support_candidates),
    ):
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        field: row[field]
                        for field in fieldnames
                    }
                )

    print(
        json.dumps(
            {
                "status": result["status"],
                "profile_json": str(profile_json),
                "profile_csv": str(profile_csv),
                "candidates_csv": str(candidates_csv),
                "train_image_count": result["train_image_count"],
                "train_instance_count": result["train_instance_count"],
                "category_count": result["category_count"],
                "attribute_count": result["attribute_count"],
                "support_candidate_count": result["support_candidate_count"],
            },
            indent=2,
        )
    )

    print("\n===== TOP 30 TRAIN CATEGORIES =====")
    for row in category_rows[:30]:
        print(
            f'{row["category_id"]:>3}  '
            f'{row["train_instance_count"]:>8}  '
            f'{row["train_image_count"]:>7}  '
            f'{row["category_name"]}'
        )

    print("\n===== TRAIN-ONLY ATTRIBUTE SUPPORT CANDIDATES =====")
    print("ID   INSTANCES  IMAGES  CATS  DOM_SHARE  ATTRIBUTE  TOP_CATEGORIES")

    for row in support_candidates[:80]:
        print(
            f'{row["attribute_id"]:>3}  '
            f'{row["train_instance_count"]:>9}  '
            f'{row["train_image_count"]:>6}  '
            f'{row["category_support_count"]:>4}  '
            f'{row["dominant_category_share"]:>9.3f}  '
            f'{row["attribute_name"]:<32}  '
            f'{row["top_categories_text"]}'
        )


if __name__ == "__main__":
    main()