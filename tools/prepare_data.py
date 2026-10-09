"""Physical-copy audit and raw downloads. Never guesses missing class splits.

Source project is read-only. Downloads are staged under data/raw, with URL,
hash, byte count, HTTP failures and extraction policy recorded in provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from PIL import Image


DOWNLOADS = {
    "dogs_images": ("https://vision.stanford.edu/aditya86/ImageNetDogs/images.tar", "dogs/images.tar"),
    "dogs_lists": ("https://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar", "dogs/lists.tar"),
    "flowers_images": ("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", "flowers/102flowers.tgz"),
    "flowers_labels": ("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat", "flowers/imagelabels.mat"),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root, verify_images=False, allow_source_links=False):
    root = Path(root)
    files = {}
    for split in ("train", "val", "test"):
        for path in sorted((root/split).rglob("*")):
            if path.is_symlink() and not allow_source_links:
                raise ValueError(f"External-link policy violated: {path}")
            if path.is_file():
                if verify_images:
                    with Image.open(path) as image:
                        image.verify()
                files[str(path.relative_to(root))] = sha256_file(path)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    counts = {}
    class_sets = []
    for split in ("train", "val", "test"):
        classes = sorted(path.name for path in (root/split).iterdir() if path.is_dir())
        class_sets.append(set(classes))
        counts[split] = {"classes": len(classes), "class_ids": classes,
                         "images": sum(name.startswith(split+"/") for name in files)}
    assert not (class_sets[0] & class_sets[1] or class_sets[0] & class_sets[2]
                or class_sets[1] & class_sets[2]), "Overlapping class splits"
    return {"tree_sha256": digest, "files": files, "counts": counts}


def verify_copies(source, destination):
    records = {}
    for name, target in (("cub", "CUB_fewshot_raw"), ("cars", "StanfordCars_fewshot")):
        source_manifest = tree_manifest(source/name, allow_source_links=True)
        copy_manifest = tree_manifest(destination/target, verify_images=True)
        assert source_manifest == copy_manifest, f"Copy mismatch: {name}"
        source_files = [source/name/relative for relative in source_manifest["files"]]
        assert all(path.stat().st_ino != (destination/target/path.relative_to(source/name)).stat().st_ino
                   for path in source_files), "Hard links violate independent-data policy"
        provenance = json.loads((source/"provenance"/f"{name}_prepared.json").read_text())
        records[name] = {"source": str(source/name), "destination": str(destination/target),
                        "verified_equal": True, "symlinks": False, "hardlinks_to_source": False,
                        "source_provenance": provenance, **copy_manifest}
        print(json.dumps({"dataset": name, "images": len(copy_manifest["files"]),
                          "tree_sha256": copy_manifest["tree_sha256"], "verified_equal": True}), flush=True)
    (destination/"provenance").mkdir(parents=True, exist_ok=True)
    (destination/"provenance/copied_datasets.json").write_text(json.dumps(records, indent=2))


def download_missing(destination, timeout, retries):
    manifest_path = destination/"provenance/downloads.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, (url, relative) in DOWNLOADS.items():
        target = destination/"raw"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        previous = records.get(name, {})
        if target.exists() and previous.get("sha256") == sha256_file(target):
            print(f"cached {name}", flush=True)
            continue
        partial = target.with_suffix(target.suffix+".part")
        failures = []
        for attempt in range(retries):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "SCEG-Reproduction/1.0"})
                with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as stream:
                    expected_size = response.headers.get("Content-Length")
                    while chunk := response.read(1024*1024):
                        stream.write(chunk)
                if expected_size is not None and partial.stat().st_size != int(expected_size):
                    raise ValueError("Incomplete HTTP body")
                partial.replace(target)
                records[name] = {"url": url, "path": str(target), "bytes": target.stat().st_size,
                                 "sha256": sha256_file(target), "status": "downloaded",
                                 "class_split_verified": False, "failures": failures}
                print(f"downloaded {name}: {target.stat().st_size} bytes", flush=True)
                break
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")
                print(f"attempt {attempt+1} failed {name}: {error}", flush=True)
                time.sleep(min(attempt+1, 3))
        else:
            records[name] = {"url": url, "path": str(target), "status": "failed",
                             "failures": failures, "class_split_verified": False}
        manifest_path.write_text(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/root/projects/aris/fewshot/data/fsfg"))
    parser.add_argument("--destination", type=Path, default=Path("data"))
    parser.add_argument("--verify-copies", action="store_true")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.verify_copies:
        verify_copies(args.source, args.destination)
    if args.download_missing:
        download_missing(args.destination, args.timeout, args.retries)


if __name__ == "__main__":
    main()
