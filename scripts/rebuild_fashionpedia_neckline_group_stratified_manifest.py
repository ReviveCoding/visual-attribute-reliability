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


def stable_seed(base_seed: int, value: str) -> int:
    encoded = f"{base_seed}|{value}".encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return int(digest[:16], 16)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def score_multi_candidate(
    counts: Counter[int],
    target_counts: dict[int, float],
    total_target: float,
) -> float:
    class_error = sum(
        (
            (float(counts[attribute_id]) - target_counts[attribute_id])
            / max(target_counts[attribute_id], 1.0)
        ) ** 2
        for attribute_id in target_counts
    )

    total_error = (
        (float(sum(counts.values())) - total_target)
        / max(total_target, 1.0)
    ) ** 2

    overshoot_error = sum(
        max(
            float(counts[attribute_id]) - target_counts[attribute_id],
            0.0,
        )
        / max(target_counts[attribute_id], 1.0)
        for attribute_id in target_counts
    )

    return class_error + (0.35 * total_error) + (0.10 * overshoot_error)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a train-only Fashionpedia neckline manifest with "
            "group-size stratification while preserving image disjointness."
        )
    )
    parser.add_argument("--source-train-manifest", required=True)
    parser.add_argument("--source-dev-manifest", required=True)
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    train_path = Path(args.source_train_manifest).resolve()
    dev_path = Path(args.source_dev_manifest).resolve()
    summary_path = Path(args.source_summary).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (train_path, dev_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source artifact is missing: {path}")

    if output_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite manifest output directory: "
            f"{output_dir}"
        )

    with summary_path.open("r", encoding="utf-8") as handle:
        source_summary = json.load(handle)

    if source_summary.get("status") != "PASS":
        raise RuntimeError(
            "Source manifest summary is not PASS. "
            f"Observed status={source_summary.get('status')!r}"
        )

    dev_fraction = float(args.dev_fraction)

    if not 0.05 <= dev_fraction <= 0.40:
        raise ValueError("dev_fraction must be between 0.05 and 0.40.")

    source_train_records = load_csv(train_path)
    source_dev_records = load_csv(dev_path)
    all_records = source_train_records + source_dev_records

    if not all_records:
        raise RuntimeError("Source manifests contain no records.")

    required_columns = {
        "annotation_id",
        "image_id",
        "split_group_id",
        "attribute_id",
        "attribute_name",
    }

    observed_columns = set(all_records[0])

    missing_columns = sorted(required_columns - observed_columns)

    if missing_columns:
        raise ValueError(
            "Source manifest columns are missing: "
            f"{missing_columns}"
        )

    annotation_ids = [
        int(record["annotation_id"])
        for record in all_records
    ]

    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError(
            "Source manifests contain duplicate annotation IDs."
        )

    records_by_group: dict[int, list[dict[str, str]]] = defaultdict(list)

    for record in all_records:
        group_id = int(record["split_group_id"])
        records_by_group[group_id].append(record)

    all_attribute_ids = sorted(
        {
            int(record["attribute_id"])
            for record in all_records
        }
    )

    attribute_names = {
        int(record["attribute_id"]): record["attribute_name"]
        for record in all_records
    }

    singleton_groups_by_class: dict[int, list[int]] = defaultdict(list)
    multi_group_ids: list[int] = []
    group_class_counts: dict[int, Counter[int]] = {}
    group_buckets: dict[int, str] = {}

    for group_id, group_records in records_by_group.items():
        class_counts = Counter(
            int(record["attribute_id"])
            for record in group_records
        )

        group_class_counts[group_id] = class_counts

        if len(group_records) == 1:
            attribute_id = int(group_records[0]["attribute_id"])
            singleton_groups_by_class[attribute_id].append(group_id)
            group_buckets[group_id] = "singleton"
        else:
            multi_group_ids.append(group_id)
            group_buckets[group_id] = "multi_instance"

    if not multi_group_ids:
        raise RuntimeError(
            "No multi-instance image groups exist, so the detected "
            "distributional issue cannot be checked."
        )

    dev_group_ids: set[int] = set()

    # For singleton groups, each group contains one sample and one label.
    # Stratify exactly by class, using a deterministic per-class shuffle.
    for attribute_id in all_attribute_ids:
        group_ids = list(singleton_groups_by_class[attribute_id])

        rng = random.Random(
            stable_seed(args.seed, f"singleton|{attribute_id}")
        )

        rng.shuffle(group_ids)

        target_count = int(round(len(group_ids) * dev_fraction))

        if len(group_ids) >= 2:
            target_count = max(1, min(len(group_ids) - 1, target_count))

        dev_group_ids.update(group_ids[:target_count])

    # For multi-instance groups, use deterministic greedy selection.
    # This preserves multi-instance images in BOTH train and dev while
    # targeting the class distribution inside this subgroup.
    multi_target_group_count = int(
        round(len(multi_group_ids) * dev_fraction)
    )

    multi_target_group_count = max(
        1,
        min(len(multi_group_ids) - 1, multi_target_group_count),
    )

    multi_total_counts = Counter()

    for group_id in multi_group_ids:
        multi_total_counts.update(group_class_counts[group_id])

    multi_target_counts = {
        attribute_id: float(multi_total_counts[attribute_id]) * dev_fraction
        for attribute_id in all_attribute_ids
    }

    multi_target_samples = (
        float(sum(multi_total_counts.values())) * dev_fraction
    )

    rng = random.Random(stable_seed(args.seed, "multi_instance"))

    random_tie = {
        group_id: rng.random()
        for group_id in multi_group_ids
    }

    remaining_multi_groups = set(multi_group_ids)
    selected_multi_groups: set[int] = set()
    selected_multi_counts: Counter[int] = Counter()

    while len(selected_multi_groups) < multi_target_group_count:
        best_group_id = None
        best_score = None

        for group_id in remaining_multi_groups:
            candidate_counts = selected_multi_counts.copy()
            candidate_counts.update(group_class_counts[group_id])

            candidate_score = score_multi_candidate(
                candidate_counts,
                multi_target_counts,
                multi_target_samples,
            )

            ranking = (
                candidate_score,
                -len(records_by_group[group_id]),
                random_tie[group_id],
                group_id,
            )

            if best_score is None or ranking < best_score:
                best_score = ranking
                best_group_id = group_id

        if best_group_id is None:
            raise RuntimeError(
                "Failed to select a multi-instance group for dev."
            )

        selected_multi_groups.add(best_group_id)
        selected_multi_counts.update(group_class_counts[best_group_id])
        remaining_multi_groups.remove(best_group_id)

    dev_group_ids.update(selected_multi_groups)
    train_group_ids = set(records_by_group) - dev_group_ids

    if not train_group_ids or not dev_group_ids:
        raise RuntimeError(
            "Rebuild produced an empty train or dev partition."
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

    overlap = train_group_ids & dev_group_ids

    total_counts = Counter(
        int(record["attribute_id"])
        for record in all_records
    )

    train_counts = Counter(
        int(record["attribute_id"])
        for record in train_records
    )

    dev_counts = Counter(
        int(record["attribute_id"])
        for record in dev_records
    )

    def bucket_summary(group_ids: set[int]) -> dict[str, int]:
        singleton_group_count = sum(
            1
            for group_id in group_ids
            if group_buckets[group_id] == "singleton"
        )

        multi_group_count = sum(
            1
            for group_id in group_ids
            if group_buckets[group_id] == "multi_instance"
        )

        singleton_sample_count = sum(
            len(records_by_group[group_id])
            for group_id in group_ids
            if group_buckets[group_id] == "singleton"
        )

        multi_sample_count = sum(
            len(records_by_group[group_id])
            for group_id in group_ids
            if group_buckets[group_id] == "multi_instance"
        )

        return {
            "singleton_group_count": singleton_group_count,
            "multi_instance_group_count": multi_group_count,
            "singleton_sample_count": singleton_sample_count,
            "multi_instance_sample_count": multi_sample_count,
        }

    total_bucket = bucket_summary(set(records_by_group))
    train_bucket = bucket_summary(train_group_ids)
    dev_bucket = bucket_summary(dev_group_ids)

    class_rows = []

    for attribute_id in all_attribute_ids:
        total_count = int(total_counts[attribute_id])
        train_count = int(train_counts[attribute_id])
        dev_count = int(dev_counts[attribute_id])

        class_rows.append(
            {
                "attribute_id": attribute_id,
                "attribute_name": attribute_names[attribute_id],
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

    class_rows.sort(key=lambda row: row["attribute_id"])

    overall_dev_fraction = (
        float(len(dev_records))
        / float(len(all_records))
    )

    singleton_dev_group_fraction = (
        float(dev_bucket["singleton_group_count"])
        / float(total_bucket["singleton_group_count"])
    )

    multi_dev_group_fraction = (
        float(dev_bucket["multi_instance_group_count"])
        / float(total_bucket["multi_instance_group_count"])
    )

    singleton_dev_sample_fraction = (
        float(dev_bucket["singleton_sample_count"])
        / float(total_bucket["singleton_sample_count"])
    )

    multi_dev_sample_fraction = (
        float(dev_bucket["multi_instance_sample_count"])
        / float(total_bucket["multi_instance_sample_count"])
    )

    checks = {
        "source_annotation_coverage_pass": (
            len(train_records) + len(dev_records) == len(all_records)
        ),
        "group_disjoint_pass": len(overlap) == 0,
        "all_classes_in_train_pass": all(
            row["train_count"] > 0
            for row in class_rows
        ),
        "all_classes_in_dev_pass": all(
            row["dev_count"] > 0
            for row in class_rows
        ),
        "overall_dev_fraction_pass": (
            0.18 <= overall_dev_fraction <= 0.22
        ),
        "class_fraction_pass": all(
            0.17 <= row["dev_fraction"] <= 0.23
            for row in class_rows
        ),
        "multi_instance_groups_in_train_pass": (
            train_bucket["multi_instance_group_count"] > 0
        ),
        "multi_instance_groups_in_dev_pass": (
            dev_bucket["multi_instance_group_count"] > 0
        ),
        "singleton_group_fraction_pass": (
            0.17 <= singleton_dev_group_fraction <= 0.23
        ),
        "multi_instance_group_fraction_pass": (
            0.17 <= multi_dev_group_fraction <= 0.23
        ),
        "multi_instance_sample_fraction_pass": (
            0.15 <= multi_dev_sample_fraction <= 0.25
        ),
    }

    status = "PASS" if all(checks.values()) else "REVIEW"

    output_dir.mkdir(parents=True, exist_ok=False)

    fieldnames = list(all_records[0].keys())

    train_csv_path = output_dir / "train_manifest.csv"
    dev_csv_path = output_dir / "dev_manifest.csv"
    summary_output_path = output_dir / "manifest_summary.json"

    for path, records in (
        (train_csv_path, train_records),
        (dev_csv_path, dev_records),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(records)

    summary = {
        "status": status,
        "schema_version": (
            "fashionpedia_neckline_attribute_manifest_v1b_group_stratified"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "train_only",
        "source_manifest_artifacts": {
            "source_train_manifest": str(train_path),
            "source_dev_manifest": str(dev_path),
            "source_summary": str(summary_path),
            "source_train_manifest_sha256": sha256_file(train_path),
            "source_dev_manifest_sha256": sha256_file(dev_path),
            "source_summary_sha256": sha256_file(summary_path),
        },
        "reason_for_rebuild": (
            "The earlier v1 manifest was group-disjoint and class-balanced, "
            "but all multi-instance eligible image groups were assigned to "
            "dev. v1b preserves singleton and multi-instance image groups "
            "in both train and dev."
        ),
        "split_policy": {
            "group_key": "split_group_id=image_id",
            "dev_fraction": dev_fraction,
            "seed": int(args.seed),
            "singleton_assignment": (
                "deterministic class-stratified group selection"
            ),
            "multi_instance_assignment": (
                "deterministic greedy group selection with class-distribution "
                "and group-size preservation"
            ),
        },
        "counts": {
            "eligible_samples": len(all_records),
            "eligible_images": len(records_by_group),
            "train_samples": len(train_records),
            "dev_samples": len(dev_records),
            "train_images": len(train_group_ids),
            "dev_images": len(dev_group_ids),
        },
        "group_size_distribution": {
            "total": total_bucket,
            "train": train_bucket,
            "dev": dev_bucket,
            "singleton_dev_group_fraction": round(
                singleton_dev_group_fraction,
                6,
            ),
            "multi_instance_dev_group_fraction": round(
                multi_dev_group_fraction,
                6,
            ),
            "singleton_dev_sample_fraction": round(
                singleton_dev_sample_fraction,
                6,
            ),
            "multi_instance_dev_sample_fraction": round(
                multi_dev_sample_fraction,
                6,
            ),
        },
        "class_distribution": class_rows,
        "checks": {
            **checks,
            "group_overlap_count": len(overlap),
            "overall_dev_fraction": round(
                overall_dev_fraction,
                6,
            ),
        },
        "artifacts": {
            "train_manifest_csv": str(train_csv_path),
            "dev_manifest_csv": str(dev_csv_path),
        },
        "claim_boundary": (
            "Train-only manifest rebuild. No official validation annotations, "
            "validation images, model training, checkpoint selection, or "
            "final performance metrics were used."
        ),
    }

    with summary_output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("===== FASHIONPEDIA NECKLINE V1B GROUP-STRATIFIED MANIFEST =====")
    print(f"STATUS: {status}")
    print(f"ELIGIBLE_SAMPLES: {len(all_records)}")
    print(f"ELIGIBLE_IMAGES: {len(records_by_group)}")
    print(f"TRAIN_SAMPLES: {len(train_records)}")
    print(f"DEV_SAMPLES: {len(dev_records)}")
    print(f"TRAIN_IMAGES: {len(train_group_ids)}")
    print(f"DEV_IMAGES: {len(dev_group_ids)}")
    print(f"OVERALL_DEV_FRACTION: {overall_dev_fraction:.6f}")
    print(f"GROUP_OVERLAP_COUNT: {len(overlap)}")

    print("\n===== GROUP-SIZE DISTRIBUTION =====")
    print(json.dumps(summary["group_size_distribution"], indent=2))

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
    for name, passed in checks.items():
        print(f"{name}: {passed}")

    print(f"\nTRAIN_MANIFEST: {train_csv_path}")
    print(f"DEV_MANIFEST: {dev_csv_path}")
    print(f"SUMMARY_JSON: {summary_output_path}")

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()