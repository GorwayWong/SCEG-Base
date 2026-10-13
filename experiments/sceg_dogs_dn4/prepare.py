"""Verify the released DN4 Dogs CSVs, then reuse the original Dogs archive adapter."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tarfile

from experiments.sceg_dogs_baseline.prepare import (
    COUNTS, archive_record, check_prepared, csv_rows, image_members, prepare,
)
from tools.prepare_data import sha256_file
from utils.config import load_config
from utils.experiment import write_json

BASE = Path(__file__).resolve().parent
REVISION = "440857b23a5bc9798004f81f732c5cc9272a99b8"
PREFIX = f"https://raw.githubusercontent.com/WenbinLee/DN4/{REVISION}/DN4_2019_Version/dataset/StanfordDogs/"
IMAGE_COUNTS = {"train": 11962, "val": 3478, "test": 5140}


def verify_protocol(config):
    source = {"url": "https://github.com/WenbinLee/DN4", "revision": REVISION,
              "mapping_method": "Exact Images/<label>/<filename> membership; no resplit or JPEG encoding",
              "artifacts": {part: {"url": PREFIX+part+".csv",
                  "path": str(BASE/"data_protocol"/(part+".csv")),
                  "sha256": sha256_file(BASE/"data_protocol"/(part+".csv"))}
                  for part in COUNTS}}
    rows = csv_rows(source)
    classes = {part: sorted({label for label, _ in pairs}) for part, pairs in rows.items()}
    union = set().union(*(set(ids) for ids in classes.values()))
    csv_inventory = Counter(pair for pairs in rows.values() for pair in pairs)
    archive_source = archive_record(config)
    with tarfile.open(archive_source["path"]) as archive:
        archive_inventory = {(label, name) for _, label, name in image_members(archive)}
    missing = sorted(set(csv_inventory)-archive_inventory)
    uncovered = sorted(archive_inventory-set(csv_inventory))
    duplicates = sorted(pair for pair, count in csv_inventory.items() if count != 1)
    summary = {"protocol_id": "dn4_dogs_2019", "source_verified": False,
               "sceg_exact_split_match": "unconfirmed", "source": source,
               "archive_source": archive_source,
               "archive_decode_audit": config["data_protocol"]["raw_inventory_manifest"],
               "classes": classes, "class_counts": {part: len(ids) for part, ids in classes.items()},
               "image_counts": {part: len(pairs) for part, pairs in rows.items()},
               "union_classes": len(union), "union_images": len(csv_inventory),
               "mapping_check": {"csv_minus_archive": missing, "archive_minus_csv": uncovered,
                                 "duplicate_pairs": duplicates}}
    verified = (summary["class_counts"] == COUNTS and summary["image_counts"] == IMAGE_COUNTS
                and len(union) == 120 and len(csv_inventory) == 20580
                and sum(len(ids) for ids in classes.values()) == len(union)
                and not missing and not uncovered and not duplicates)
    summary.update(status="verified" if verified else "failed", source_verified=verified)
    write_json(config["data_protocol"]["split_manifest"], summary)
    if not verified:
        raise ValueError("DN4 CSV/archive mapping failed; exact differences are in split.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(BASE/"configs/dogs.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    split = verify_protocol(config)
    # Reuse only a preparation with matching sources, class mapping and file inventory.
    record = check_prepared(config) if Path(config["data"]["root"]).exists() else prepare(config)
    for part, expected in IMAGE_COUNTS.items():
        if record["classes"][part]["images"] != expected:
            raise ValueError(f"Prepared image count differs from CSV: {part}")
    print(json.dumps({"status": record["status"], "protocol_id": split["protocol_id"],
                      "source_verified": split["source_verified"],
                      "sceg_exact_split_match": split["sceg_exact_split_match"],
                      "class_counts": split["class_counts"], "image_counts": split["image_counts"]}))


if __name__ == "__main__":
    main()
