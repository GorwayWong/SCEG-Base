"""Compare the user-provided author class lists with this run's exact DN4 CSVs."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from tools.prepare_data import sha256_file
from utils.config import load_config
from utils.experiment import write_json
from .prepare import BASE

PARTS = ("train", "val", "test")


def author_classes(path, expected, numeric=False):
    classes = {part: [] for part in PARTS}
    declared, part = {}, None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        header = re.fullmatch(r"\[(train|val|test)\]\s*\((\d+) classes\)", line)
        if header:
            part = header.group(1)
            declared[part] = int(header.group(2))
        elif part and line:
            classes[part].append(int(line) if numeric else line)
    counts = {part: len(ids) for part, ids in classes.items()}
    union = [value for values in classes.values() for value in values]
    assert counts == declared == expected
    assert len(union) == len(set(union)) == sum(expected.values())
    if numeric:
        assert set(union) == set(range(1, 197))
    return classes


def main():
    dogs_path = BASE/"data_protocol/author_splits/StanfordDogs_fewshot_splits.txt"
    cars_path = BASE/"data_protocol/author_splits/StanfordCars_fewshot_splits.txt"
    dogs = author_classes(dogs_path, {"train": 70, "val": 20, "test": 30})
    cars = author_classes(cars_path, {"train": 130, "val": 17, "test": 49}, numeric=True)
    csv_source = {}
    for part in PARTS:
        path = BASE/"data_protocol"/(part+".csv")
        with path.open(newline="", encoding="utf-8") as stream:
            csv_source[part] = {row["label"] for row in csv.DictReader(stream)}
    author_map = {name: part for part, names in dogs.items() for name in names}
    dn4_map = {name: part for part, names in csv_source.items() for name in names}
    inventory = json.loads((BASE.parent/"sceg_dogs_baseline/data_protocol/raw_inventory.json").read_text())
    assert set(author_map) == set(dn4_map) == set(inventory["class_images"])
    config = load_config(BASE/"runs/dogs800_seed42_dn4/config.yaml")
    root = Path(config["data"]["root"])
    prepared_counts = {}
    for part in PARTS:
        folder = root/(part if part == "train" else part+"_pre")
        actual = {path.name for path in folder.iterdir() if path.is_dir()}
        assert actual == csv_source[part]
        prepared_counts[part] = len(actual)
    matrix = {part: {other: len(csv_source[part]&set(dogs[other])) for other in PARTS}
              for part in PARTS}
    changes = [{"class": name, "dn4_split": dn4_map[name], "author_split": author_map[name]}
               for name in sorted(author_map) if dn4_map[name] != author_map[name]]
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "source_provenance": "User supplied attachments obtained from the paper authors",
              "scope": "Exact class identities/assignments; author files contain no image-level lists",
              "author_sources": {name: {"path": str(path.relative_to(BASE)), "sha256": sha256_file(path)}
                                 for name, path in (("dogs", dogs_path), ("cars", cars_path))},
              "dogs": {"protocol_id": "dn4_dogs_2019", "sceg_exact_split_match": "mismatch" if changes else "matched",
                       "author_class_counts": {part: len(ids) for part, ids in dogs.items()},
                       "dn4_class_counts": {part: len(ids) for part, ids in csv_source.items()},
                       "prepared_class_counts": prepared_counts, "prepared_classes_match_dn4_csv": True,
                       "class_universe_matches_archive": True, "total_classes": len(author_map),
                       "matched_class_count": len(author_map)-len(changes), "changed_class_count": len(changes),
                       "matrix_dn4_rows_author_columns": matrix, "changed_assignments": changes,
                       "per_split_difference": {part: {"dn4_only": sorted(csv_source[part]-set(dogs[part])),
                           "author_only": sorted(set(dogs[part])-csv_source[part])} for part in PARTS},
                       "author_test_classes_in_dn4_train": sorted(csv_source["train"]&set(dogs["test"]))},
              "cars": {"author_class_counts": {part: len(ids) for part, ids in cars.items()},
                       "unique_classes": 196, "mutually_disjoint": True, "covers_original_ids_1_to_196": True,
                       "current_experiment": None, "comparison_status": "not_applicable_no_Cars_run_in_this_task"},
              "frozen_protocol_files_modified": False, "training_or_evaluation_changed": False,
              "conclusion": "Current Dogs run is a DN4 public-split baseline, not a reproduction of the author-provided class split"}
    write_json(BASE/"data_protocol/author_split_comparison.json", result)
    print(json.dumps({"dogs_split_match": result["dogs"]["sceg_exact_split_match"],
                      "changed_classes": len(changes), "matrix": matrix,
                      "actual_prepared_classes_match_dn4": True, "cars_current_experiment": None}))


if __name__ == "__main__":
    main()
