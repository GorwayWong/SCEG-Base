"""Prepare the published FRN CUB split; never invent a split or change source data.

FRN init_CUB_fewshot.py loops original IDs 1..200: even=train, ID%4==1=val,
ID%4==3=test. This differs from the supplied zero-based CloserLook-style split.
All image bytes are copied unchanged into a separate physical directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from tools.prepare_data import tree_manifest, sha256_file

SOURCE_URL = "https://raw.githubusercontent.com/Tsingularity/FRN/64e43f3ca3a4953445788fc853e288e032d0083c/data/init_CUB_fewshot.py"
SOURCE_SHA256 = "271ce56cbc9442c20fb2918cbd863e62b37ada44dd2f7a8658f111c76b584a7c"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/CUB_fewshot_raw"))
    parser.add_argument("--destination", type=Path, default=Path("data/frn/CUB_fewshot_raw"))
    args = parser.parse_args()
    published = urllib.request.urlopen(SOURCE_URL, timeout=20).read()
    assert hashlib.sha256(published).hexdigest() == SOURCE_SHA256
    original_folders = {}
    for split in ("train", "val", "test"):
        for folder in sorted((args.source/split).iterdir()):
            if folder.is_dir():
                class_id = int(folder.name.split(".", 1)[0])
                assert class_id not in original_folders
                original_folders[class_id] = folder
    assert set(original_folders) == set(range(1, 201))
    byte_identity = True
    for class_id, folder in sorted(original_folders.items()):
        split = "train" if class_id % 2 == 0 else ("val" if class_id % 4 == 1 else "test")
        target = args.destination/split/folder.name
        if not target.exists():
            shutil.copytree(folder, target, symlinks=False)
        for image_path in folder.iterdir():
            assert not image_path.is_symlink()
            copied_path = target/image_path.name
            assert not copied_path.is_symlink()
            assert image_path.stat().st_ino != copied_path.stat().st_ino
            byte_identity = byte_identity and sha256_file(image_path) == sha256_file(copied_path)
    assert byte_identity
    manifest = tree_manifest(args.destination, verify_images=True)
    expected_classes = {
        "train": [folder.name for class_id, folder in sorted(original_folders.items()) if class_id % 2 == 0],
        "val": [folder.name for class_id, folder in sorted(original_folders.items()) if class_id % 4 == 1],
        "test": [folder.name for class_id, folder in sorted(original_folders.items()) if class_id % 4 == 3],
    }
    assert all(manifest["counts"][split]["class_ids"] == classes for split, classes in expected_classes.items())
    record = {"source": str(args.source), "destination": str(args.destination),
        "verified_equal": False, "verified_images_match_source": True,
        "split_unchanged_from_source": False, "symlinks": False, "hardlinks_to_source": False,
        "source_provenance": {"url": SOURCE_URL, "sha256": SOURCE_SHA256,
            "commit": "64e43f3ca3a4953445788fc853e288e032d0083c",
            "original_class_ids": "1..200", "train": "id%2==0", "val": "id%4==1", "test": "id%4==3",
            "reason": "Follow author's README reference to published FRN; supplied split has different identities"},
        **manifest}
    path = Path("data/provenance/copied_datasets.json")
    records = json.loads(path.read_text())
    records["cub_frn"] = record
    path.write_text(json.dumps(records, indent=2))
    print(json.dumps({"destination": str(args.destination), "tree_sha256": manifest["tree_sha256"],
                      "counts": {split: {key: value for key, value in count.items() if key != "class_ids"}
                                 for split, count in manifest["counts"].items()}}, indent=2))


if __name__ == "__main__":
    main()
