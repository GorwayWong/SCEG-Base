"""Prepare author-split Dogs data only; no model, trainer, or DN4 protocol dependency."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import make_dataset
from tools.prepare_data import sha256_file
from utils.experiment import write_json

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PROTOCOL = "sceg_author_dogs_v1"
COUNTS = {"train": 70, "val": 20, "test": 30}
SOURCES = ROOT/"data/provenance/sceg_author"
ARCHIVE = Path("/root/projects/aris/sceg_fsfg/data/raw/dogs/images.tar")
ARCHIVE_AUDIT = Path("/root/projects/aris/sceg_fsfg/data/provenance/raw_archive_audit.json")


def parse_classes(path, counts=COUNTS, numeric=False):
    classes, declared, part = {key: [] for key in counts}, {}, None
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        match = re.fullmatch(r"\[(train|val|test)\]\s*\((\d+) classes\)", line)
        if match:
            part = match.group(1)
            if part in declared:
                raise ValueError(f"Duplicate split section: {part}")
            declared[part] = int(match.group(2))
        elif part and line:
            if not numeric and (not re.fullmatch(r"n\d{8}-.+", line) or
                    PurePosixPath(line).name != line or "\\" in line):
                raise ValueError(f"Invalid breed directory: {line}")
            classes[part].append(int(line) if numeric else line)
    if declared != counts or {key: len(ids) for key, ids in classes.items()} != counts:
        raise ValueError("Declared/actual author class counts differ")
    flattened = [value for values in classes.values() for value in values]
    if len(set(flattened)) != sum(counts.values()):
        raise ValueError("Duplicate or overlapping author class identities")
    if numeric and set(flattened) != set(range(1, 197)):
        raise ValueError("Cars IDs must cover 1..196")
    return classes


def image_members(archive):
    # Same regular Images/<breed>/<filename> mapping as the established archive adapter.
    seen = set()
    for member in archive:
        if member.isdir():
            continue
        parts = PurePosixPath(member.name).parts
        if not member.isfile() or len(parts) != 3 or parts[0] != "Images" or ".." in parts or member.name in seen:
            raise ValueError(f"Unsafe/duplicate archive member: {member.name}")
        seen.add(member.name)
        yield member, parts[1], parts[2]


def data_config(root):
    # Existing data component settings; this is not a training recipe/configuration.
    return {"data": {"root": str(root), "image_size": 84, "eval_resize": 92,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "crop_scale": [0.08, 1.0], "crop_ratio": [0.75, 4/3], "jitter": 0.4,
        "flip_probability": 0.5, "pre_resized": True, "workers": 0}}


def reuse_or_write(target, content, candidate=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f"Duplicate destination: {target}")
    if candidate is not None and candidate.is_file() and not candidate.is_symlink() and candidate.read_bytes() == content:
        if candidate.stat().st_dev == target.parent.stat().st_dev:
            os.link(candidate, target)
            return "hardlink"
    target.write_bytes(content)
    return "written"


def verify_data(root, prep_dir, new_pre_classes=()):
    """Full inventory/class checks plus small CPU-only real data reads."""
    root, prep_dir = Path(root), Path(prep_dir)
    split = json.loads((prep_dir/"split.json").read_text())
    if (prep_dir/"prepared.json").exists():
        previous = json.loads((prep_dir/"prepared.json").read_text())
        if (sha256_file(prep_dir/"split.json") != previous["split_manifest_sha256"] or
                sha256_file(prep_dir/"samples.jsonl") != previous["sample_manifest_sha256"]):
            raise ValueError("Prepared split/sample manifest changed")
    config = data_config(root)
    paths, counts, first = set(), Counter(), {}
    with (prep_dir/"samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            part, breed = row["split"], row["class"]
            if breed not in split["classes"][part]:
                raise ValueError("Sample assignment differs from author classes")
            counts[part] += 1
            first.setdefault((part, breed), row)
            for key in ("raw_path", "pre_path"):
                if key not in row:
                    continue
                relative = row[key]
                if relative in paths or (root/relative).is_symlink() or not (root/relative).is_file():
                    raise ValueError(f"Missing, duplicate, or linked sample path: {relative}")
                paths.add(relative)
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if paths != actual or dict(counts) != split["image_counts"]:
        raise ValueError("Full sample/file inventory differs")
    selected = {(part, breed) for part, breed in first if breed in new_pre_classes}
    for part in ("val", "test"):
        selected.update((part, breed) for breed in split["classes"][part][:3])
    for part in COUNTS:
        dataset = make_dataset(config, part)
        if dataset.class_to_idx != split["class_to_idx"][part] or len(dataset) != counts[part]:
            raise ValueError("Actual ImageFolder class mapping/count differs")
        images, labels = next(iter(DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)))
        if tuple(images.shape) != (3, 3, 84, 84) or not torch.isfinite(images).all() or labels.min() < 0:
            raise ValueError("CPU data-component batch check failed")
    checked = []
    raw_config = deepcopy(config)
    raw_config["data"]["pre_resized"] = False
    raw_datasets = {part: make_dataset(raw_config, part) for part in ("val", "test")}
    pre_datasets = {part: make_dataset(config, part) for part in ("val", "test")}
    for part, breed in sorted(selected):
        row = first[part, breed]
        with Image.open(root/row["raw_path"]) as image:
            online = raw_datasets[part].transform(image.convert("RGB"))
        with Image.open(root/row["pre_path"]) as image:
            if image.mode != "RGB" or image.size != (84, 84):
                raise ValueError("Preprocessed PNG is not RGB 84x84")
            offline = pre_datasets[part].transform(image)
        if not torch.equal(online, offline):
            raise ValueError(f"Online/pre input mismatch: {breed}")
        checked.append(row["raw_path"])
    return {"full_inventory_checked": True, "ImageFolder_checked": True, "CPU_batches": 3,
            "batch_size": 3, "input_shape": [3, 84, 84], "online_pre_tensor_equal_samples": checked,
            "new_pre_classes_checked": sorted(new_pre_classes), "model_loaded": False}


def prepare_data(root, prep_dir, sources=SOURCES, archive_path=ARCHIVE, audit_path=ARCHIVE_AUDIT,
                 reuse_root=None, expected_images=20580):
    root, prep_dir, sources = Path(root), Path(prep_dir), Path(sources)
    prepared_path = prep_dir/"prepared.json"
    source = json.loads((sources/"source.json").read_text())
    author_path = sources/"StanfordDogs_fewshot_splits.txt"
    if sha256_file(author_path) != source["files"]["dogs"]["sha256"]:
        raise ValueError("Author Dogs original hash differs from authoritative source record")
    classes = parse_classes(author_path)
    if root.exists():
        prepared = json.loads(prepared_path.read_text())
        if prepared["root"] != str(root.resolve()) or prepared["author_split_sha256"] != sha256_file(author_path):
            raise ValueError("Existing author preparation differs; preserve it for inspection")
        verify_data(root, prep_dir, prepared["reuse"]["new_pre_classes"])
        return prepared
    archive_source = json.loads(Path(audit_path).read_text())["dogs"]
    if Path(archive_path).stat().st_size != archive_source["bytes"]:
        raise ValueError("Raw archive size differs from existing source audit")
    image_counts = Counter()
    with tarfile.open(archive_path) as archive:
        for _, breed, _ in image_members(archive):
            image_counts[breed] += 1
    available = set(image_counts)
    author_set = {breed for group in classes.values() for breed in group}
    if available != author_set or sum(image_counts.values()) != expected_images:
        raise ValueError(f"Archive/author coverage differs: missing={sorted(author_set-available)}, extra={sorted(available-author_set)}")
    assignments = {breed: part for part, breeds in classes.items() for breed in breeds}
    prep_dir.mkdir(parents=True, exist_ok=True)
    split = {"protocol_id": PROTOCOL, "source_kind": "author_via_email_user_supplied",
        "source_record": str(sources/"source.json"), "author_split_sha256": sha256_file(author_path),
        "classes": classes, "class_to_idx": {part: {breed: index for index, breed in enumerate(sorted(breeds))}
                                             for part, breeds in classes.items()},
        "class_image_counts": dict(sorted(image_counts.items())),
        "image_counts": {part: sum(image_counts[breed] for breed in breeds) for part, breeds in classes.items()},
        "mapping": "All Images/<breed>/<filename> members assigned by the author breed list",
        "coverage": {"missing_classes": [], "extra_classes": [], "duplicate_image_paths": []}}
    write_json(prep_dir/"split.json", split)
    root.mkdir(parents=True, exist_ok=False)
    resize_crop = transforms.Compose([transforms.Resize(92), transforms.CenterCrop(84)])
    reuse = Counter()
    new_pre_classes = set()
    manifest = prep_dir/"samples.jsonl"
    with manifest.open("w") as output, tarfile.open(archive_path) as archive:
        for member, breed, filename in image_members(archive):
            part = assignments[breed]
            content = archive.extractfile(member).read()
            raw_path = Path(part)/breed/filename
            candidates = [Path(reuse_root)/old/breed/filename for old in COUNTS] if reuse_root else []
            candidate = next((path for path in candidates if path.is_file()), None)
            reuse["raw_"+reuse_or_write(root/raw_path, content, candidate)] += 1
            row = {"split": part, "class": breed, "class_index": split["class_to_idx"][part][breed],
                   "raw_path": str(raw_path), "raw_sha256": hashlib.sha256(content).hexdigest()}
            if part != "train":
                with Image.open(io.BytesIO(content)) as image:
                    cropped = resize_crop(image.convert("RGB"))
                pre_path = Path(part+"_pre")/breed/Path(filename).with_suffix(".png")
                pre_candidates = [Path(reuse_root)/(old+"_pre")/breed/Path(filename).with_suffix(".png")
                                  for old in ("val", "test")] if reuse_root else []
                pre_candidate = next((path for path in pre_candidates if path.is_file() and not path.is_symlink()), None)
                if pre_candidate is not None:
                    with Image.open(pre_candidate) as saved:
                        if saved.mode != "RGB" or saved.size != (84, 84) or saved.tobytes() != cropped.tobytes():
                            raise ValueError(f"Existing PNG pixels differ: {pre_candidate}")
                    reuse["pre_"+reuse_or_write(root/pre_path, pre_candidate.read_bytes(), pre_candidate)] += 1
                else:
                    new_pre_classes.add(breed)
                    (root/pre_path).parent.mkdir(parents=True, exist_ok=True)
                    if (root/pre_path).exists():
                        raise ValueError(f"Preprocessed name collision: {pre_path}")
                    cropped.save(root/pre_path)
                    reuse["pre_generated"] += 1
                row.update(pre_path=str(pre_path), pre_sha256=sha256_file(root/pre_path))
            output.write(json.dumps(row, separators=(",", ":"))+"\n")
    validation = verify_data(root, prep_dir, new_pre_classes)
    prepared = {"stage": "author_data_preparation", "status": "completed", "protocol_id": PROTOCOL,
        "root": str(root.resolve()), "data_ready": True, "training_started": False, "accuracy": None,
        "author_split_sha256": sha256_file(author_path), "split_manifest": "split.json",
        "split_manifest_sha256": sha256_file(prep_dir/"split.json"), "sample_manifest": "samples.jsonl",
        "sample_manifest_sha256": sha256_file(manifest), "raw_images": sum(image_counts.values()),
        "image_counts": split["image_counts"], "class_counts": COUNTS,
        "raw_archive": {"path": str(archive_path), "bytes": archive_source["bytes"],
            "sha256": archive_source["sha256"], "hash_source": str(audit_path), "sha256_recomputed": False},
        "data_component_config": data_config(root), "transform": {"RGB": True, "resize_short_edge": 92,
            "center_crop": 84, "pre_format": "PNG", "pre_loader_resizes_again": False,
            "source": "Legenddddd/SCEG@f7f130aeb5b1b49246150574c53a874d4ce8de80 utils/util.py:get_pre_folder(type=0)"},
        "reuse": {**reuse, "new_pre_classes": sorted(new_pre_classes), "symlinks_created": 0},
        "validation": validation, "cars_image_mapping_checked": False}
    write_json(prepared_path, prepared)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT/"data/dogs_author/StanfordDogs_fewshot")
    parser.add_argument("--reuse-root", type=Path, help="Optional byte/pixel-checked hardlink donor, never a dependency")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.check_only:
        prepared = json.loads((BASE/"prep/prepared.json").read_text())
        verify_data(args.root, BASE/"prep", prepared["reuse"]["new_pre_classes"])
        print(json.dumps({"data_ready": True, "training_started": False, "checked_existing": True}))
    else:
        result = prepare_data(args.root, BASE/"prep", reuse_root=args.reuse_root)
        print(json.dumps({key: result[key] for key in ("status", "data_ready", "training_started", "image_counts", "reuse")}))


if __name__ == "__main__":
    main()
