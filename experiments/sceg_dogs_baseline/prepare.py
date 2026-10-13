"""Dogs-only archive adapter. Exact class identities must be supplied with evidence.

No split is inferred from order, a seed, breed counts or Stanford image-level lists.
The downloaded archive and all historical provenance are read-only.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from PIL import Image
from torchvision import transforms
from datasets import make_dataset
from tools.prepare_data import sha256_file
from utils.config import load_config
from utils.experiment import write_json

BASE = Path(__file__).resolve().parent
COUNTS = {"train": 70, "val": 20, "test": 30}


def archive_record(config):
    path = Path(config["data_protocol"]["archive"])
    manifest = Path(config["data_protocol"]["existing_archive_manifest"])
    record = json.loads(manifest.read_text())["dogs"]
    if path.stat().st_size != record["bytes"]:
        raise ValueError("Archive size differs from existing source audit")
    return {"path": str(path.resolve()), "bytes": record["bytes"],
            "sha256": record["sha256"], "hash_source": str(manifest),
            "sha256_recomputed": False}


def image_members(archive):
    seen = set()
    for member in archive:
        if member.isdir():
            continue
        parts = PurePosixPath(member.name).parts
        if (not member.isfile() or len(parts) != 3 or parts[0] != "Images"
                or ".." in parts or member.name in seen):
            raise ValueError(f"Unsafe or duplicate Dogs archive entry: {member.name}")
        seen.add(member.name)
        yield member, parts[1], parts[2]


def audit_raw(config, output):
    source = archive_record(config)
    counts = Counter()
    with tarfile.open(source["path"]) as archive:
        for member, breed, _ in image_members(archive):
            with archive.extractfile(member) as stream, Image.open(stream) as image:
                image.convert("RGB").load()  # decode pixels, not just JPEG headers
            counts[breed] += 1
    if len(counts) != 120 or sum(counts.values()) != 20580:
        raise ValueError("Raw Dogs archive does not contain the expected 120/20580")
    result = {"source": source, "images": sum(counts.values()), "classes": len(counts),
              "all_images_decoded": True, "class_images": dict(sorted(counts.items())),
              "exact_fsfg_split_verified": False, "fsfg_class_to_index": None}
    write_json(output, result)
    return result


def trusted_split(path, available):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("status") != "verified":
        raise RuntimeError("BLOCKED: trusted exact Dogs 70/20/30 class split missing")
    source = manifest.get("source") or {}
    for key in ("url", "revision", "mapping_method"):
        if not source.get(key):
            raise ValueError(f"Split source lacks {key}")
    if "artifacts" in source:
        rows = csv_rows(source)
        for part in COUNTS:
            if sorted({label for label, _ in rows[part]}) != sorted(manifest["classes"][part]):
                raise ValueError("CSV class identities differ from split manifest")
    else:
        for key in ("artifact_path", "artifact_sha256"):
            if not source.get(key):
                raise ValueError(f"Split source lacks {key}")
        if sha256_file(source["artifact_path"]) != source["artifact_sha256"]:
            raise ValueError("Exact split source artifact hash mismatch")
    classes = manifest["classes"]
    flattened = []
    for split, expected in COUNTS.items():
        ids = classes[split]
        if not isinstance(ids, list) or len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f"Invalid exact class list: {split}")
        flattened.extend(ids)
    if len(set(flattened)) != 120 or set(flattened) != set(available):
        raise ValueError("Class lists must be disjoint and cover all 120 raw breeds")
    return manifest


def csv_rows(source):
    """Read the three released filename,label artifacts without altering their bytes."""
    if set(source["artifacts"]) != set(COUNTS):
        raise ValueError("CSV source requires train, val and test artifacts")
    rows = {}
    for part, artifact in source["artifacts"].items():
        if not artifact.get("url") or sha256_file(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"CSV source artifact hash/URL mismatch: {part}")
        with Path(artifact["path"]).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["filename", "label"]:
                raise ValueError(f"Unexpected CSV header: {part}")
            pairs = []
            for row in reader:
                name, label = row["filename"], row["label"]
                if (None in row or not name or not label or
                        any(PurePosixPath(value).name != value or value in {".", ".."}
                            or "\\" in value for value in (name, label))):
                    raise ValueError(f"Invalid CSV path: {part}, {row}")
                pairs.append((label, name))
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"Duplicate (label, filename) in CSV: {part}")
        rows[part] = pairs
    return rows


def file_state(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def prepare(config):
    source = archive_record(config)
    with tarfile.open(source["path"]) as archive:
        available = {breed for _, breed, _ in image_members(archive)}
    split_path = Path(config["data_protocol"]["split_manifest"])
    split = trusted_split(split_path, available)  # before any dataset writes
    destination = Path(config["data"]["root"])
    destination.mkdir(parents=True, exist_ok=False)
    lookup = {breed: part for part, breeds in split["classes"].items() for breed in breeds}
    transform = transforms.Compose([transforms.Resize(config["data"]["eval_resize"]),
                                    transforms.CenterCrop(config["data"]["image_size"])])
    files = {}
    with tarfile.open(source["path"]) as archive:
        for member, breed, name in image_members(archive):
            part = lookup[breed]
            target = destination/part/breed/name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream:
                content = stream.read()
            target.write_bytes(content)
            files[str(target.relative_to(destination))] = {
                "sha256": hashlib.sha256(content).hexdigest(), "stat": file_state(target)}
            with Image.open(target) as image:
                image = image.convert("RGB")
                image.load()
                if part != "train":
                    cropped = transform(image)
                    pre = destination/(part+"_pre")/breed/Path(name).with_suffix(".png")
                    pre.parent.mkdir(parents=True, exist_ok=True)
                    if pre.exists():
                        raise ValueError("Pre-resized filename collision")
                    cropped.save(pre)
                    with Image.open(pre) as saved:
                        if saved.tobytes() != cropped.tobytes():
                            raise ValueError("Pre-resized PNG pixel mismatch")
                    files[str(pre.relative_to(destination))] = {
                        "sha256": sha256_file(pre), "stat": file_state(pre)}
    classes = {}
    for part in COUNTS:
        dataset = make_dataset(config, part)
        if set(dataset.classes) != set(split["classes"][part]):
            raise ValueError("Prepared ImageFolder class identity mismatch")
        counts = Counter(dataset.targets)
        if min(counts.values()) < (20 if part == "train" else 21):
            raise ValueError("Insufficient disjoint support/query images")
        classes[part] = {"class_to_idx": dataset.class_to_idx, "images": len(dataset)}
    result = {"status": "verified", "root": str(destination.resolve()), "source": source,
              "split_manifest_sha256": sha256_file(split_path), "classes": classes,
              "files": files, "all_images_decoded": True, "pre_pixels_verified": True,
              "transform": {"resize": 92, "center_crop": 84, "format": "RGB PNG"}}
    write_json(config["data_protocol"]["prepared_manifest"], result)
    return result


def check_prepared(config):
    split_path = Path(config["data_protocol"]["split_manifest"])
    inventory_path = config["data_protocol"].get("raw_inventory_manifest",
                                                   BASE/"data_protocol/raw_inventory.json")
    inventory = json.loads(Path(inventory_path).read_text())
    split = trusted_split(split_path, inventory["class_images"])
    record = json.loads(Path(config["data_protocol"]["prepared_manifest"]).read_text())
    root = Path(config["data"]["root"]).resolve()
    if (record["status"] != "verified" or record["root"] != str(root)
            or record["split_manifest_sha256"] != sha256_file(split_path)):
        raise ValueError("Prepared data provenance differs from configuration/split")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(record["files"]):
        raise ValueError("Prepared file inventory changed")
    for relative, fingerprint in record["files"].items():
        path = root/relative
        if path.is_symlink() or file_state(path) != fingerprint["stat"]:
            raise ValueError(f"Prepared file state changed: {relative}")
    for part in COUNTS:
        dataset = make_dataset(config, part)
        if (set(dataset.classes) != set(split["classes"][part])
                or dataset.class_to_idx != record["classes"][part]["class_to_idx"]):
            raise ValueError("ImageFolder class-to-index changed")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(BASE/"configs/dogs.yaml"))
    parser.add_argument("--audit-raw", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = (audit_raw(config, BASE/"data_protocol/raw_inventory.json")
              if args.audit_raw else prepare(config))
    print(json.dumps({key: result.get(key) for key in ("status", "classes", "images")},
                     default=str))


if __name__ == "__main__":
    main()
