"""Bounded one-time Dogs closeout: preserve sources, move evidence, then clean donors."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from tools.prepare_data import sha256_file
from utils.experiment import write_json
from ..prepare import BASE, ROOT, SOURCES, COUNTS, parse_classes, verify_data

OLD = ROOT/"experiments/sceg_dogs_dn4"
RUN_ID = "dogs800_seed42_dn4"
DEST = ROOT/"experiments/_archive/sceg_dogs_dn4"/RUN_ID
OLD_DATA = ROOT/"data/dogs_dn4/StanfordDogs_fewshot"
WATCH = Path("/root/.codex/tmp/sceg_dogs_dn4_after_finish")
MANIFEST = BASE/"prep/cleanup_manifest.json"
HASHES = {"dogs": "350aa6cbc28605cfe5371354ec8f440878825fb420960e51fcc9766809c55df1",
          "cars": "2677107210b2bf1c9dfe1741ddc7d6d8dad992232ff04efa24ba4676aa48052c"}


def now():
    return datetime.now(timezone.utc).isoformat()


def scoped_processes():
    signatures = {"experiments.sceg_dogs_dn4.run", "sceg_dogs_dn4_seed42",
                  "sceg_dogs_dn4_after_finish", str(WATCH/"watch.py")}
    matches, open_files = [], []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc/"cmdline").read_bytes().decode(errors="replace").split("\0")
            if any(arg in signatures or arg.startswith(str(OLD)+"/") for arg in args):
                matches.append({"pid": int(proc.name), "argv": args, "cwd": os.readlink(proc/"cwd")})
            for fd in (proc/"fd").iterdir():
                path = os.readlink(fd)
                if path.startswith(str(OLD/"runs"/RUN_ID)+"/") or path.startswith(str(OLD_DATA)+"/"):
                    open_files.append({"pid": int(proc.name), "path": path})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return matches, open_files


def protection_snapshot():
    roots = [Path("/root/projects/aris/sceg_fsfg/data"),
             Path("/root/projects/aris/sceg_fsfg/experiments/sceg_original"),
             ROOT/"experiments/sceg_original", Path("/root/projects/aris/sceg_proker/experiments/sceg_proker")]
    result = {}
    for root in roots:
        digest, count = hashlib.sha256(), 0
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                stat = path.stat()
                digest.update(json.dumps([str(path.relative_to(root)), stat.st_size,
                                         stat.st_mtime_ns, stat.st_ino]).encode())
                count += 1
        result[str(root)] = {"files": count, "metadata_sha256": digest.hexdigest()}
    executable = Path("/root/projects/aris/sceg_fsfg/.venv/bin/python")
    stat = executable.stat()
    result["shared_environment"] = {"executable": str(executable.resolve()), "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "packages": {key: importlib.metadata.version(key)
            for key in ("torch", "torchvision", "Pillow", "numpy")}}
    return result


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"stage": "author_data_preparation", "status": "in_progress", "started_utc": now(),
            "code_commit": subprocess.run(["rtk", "proxy", "git", "rev-parse", "HEAD"], cwd=ROOT,
                capture_output=True, text=True, check=True).stdout.strip(),
            "actions": [], "retained_nonessential_items": [], "allocated_bytes_released_by_unlink": 0,
            "before_protection": protection_snapshot()}


def action(manifest, kind, path, reason, new=None, **values):
    entry = {"action": kind, "path": str(path), "reason": reason, **values}
    if new:
        entry["new_path"] = str(new)
    manifest["actions"].append(entry)
    write_json(MANIFEST, manifest)


def move(manifest, old, new, reason):
    if not old.exists() and new.exists():
        return
    if not old.exists():
        raise FileNotFoundError(old)
    if new.exists():
        raise ValueError(f"Archive collision; preserve both: {new}")
    new.parent.mkdir(parents=True, exist_ok=True)
    before = old.stat()
    if before.st_dev != new.parent.stat().st_dev:
        raise RuntimeError("Cross-filesystem move requires a separately verified copy")
    old.rename(new)
    after = new.stat()
    if (before.st_ino, before.st_size) != (after.st_ino, after.st_size):
        raise ValueError("Moved object identity changed")
    action(manifest, "archive", old, reason, new, copied_bytes=0, disk_bytes_released=0)


def preserve_authors(manifest):
    SOURCES.mkdir(parents=True, exist_ok=True)
    records = {}
    for dataset, filename, counts, numeric in (
            ("dogs", "StanfordDogs_fewshot_splits.txt", COUNTS, False),
            ("cars", "StanfordCars_fewshot_splits.txt", {"train": 130, "val": 17, "test": 49}, True)):
        old = OLD/"data_protocol/author_splits"/filename
        target = SOURCES/filename
        available = target if target.exists() else old
        if sha256_file(available) != HASHES[dataset]:
            raise ValueError(f"Possible updated author original; do not overwrite: {available}")
        classes = parse_classes(available, counts, numeric=numeric)
        if not target.exists():
            move(manifest, old, target, "single active authoritative author original")
        if sha256_file(target) != HASHES[dataset]:
            raise ValueError("Neutral author original verification failed")
        records[dataset] = {"original_filename": filename, "path": filename,
            "original_location": str(old.relative_to(ROOT)), "sha256": HASHES[dataset],
            "bytes": target.stat().st_size, "class_id_semantics": "original 1-based IDs" if numeric else "complete breed directory names with synset IDs",
            "class_counts": {key: len(ids) for key, ids in classes.items()}, "unique_classes": sum(counts.values()),
            "class_list_checked": True, "image_mapping_checked": False if numeric else None}
    source_path = SOURCES/"source.json"
    if source_path.exists():
        existing = json.loads(source_path.read_text())
        if any(existing["files"][key]["sha256"] != value for key, value in HASHES.items()):
            raise ValueError("Existing neutral source record differs")
    else:
        write_json(source_path, {"source_kind": "author_via_email_user_supplied", "email_date": "unknown",
            "publication_permission": "unknown", "author_supplied_image_lists": False,
            "author_supplied_complete_training_recipe": False, "files": records})
    manifest["author_sources_preserved"] = True
    manifest["cars_class_list_checked"] = True
    manifest["cars_image_mapping_checked"] = False
    action(manifest, "keep", source_path, "neutral source record; no email/private message content")


def archive_old(manifest):
    matches, opened = scoped_processes()
    if matches or opened:
        raise RuntimeError(f"Old task ownership/use requires targeted stop: processes={matches}, open_files={opened}")
    manifest["old_task_stopped"] = True
    manifest["termination"] = {"checked_utc": now(), "historical_pids": [15906, 15905, 30135, 30134],
        "matching_processes": matches, "open_run_or_data_files": opened, "terminated_objects": [],
        "result": "already absent; no signals were necessary",
        "cuda_compute_apps": subprocess.run(["rtk", "proxy", "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                             "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()}
    write_json(MANIFEST, manifest)
    preserve_authors(manifest)
    old_run = OLD/"runs"/RUN_ID
    run = DEST/"run"
    current = old_run if old_run.exists() else run
    weights = {}
    for name in ("checkpoint.pth", "latest.pth"):
        path = current/name
        if not path.is_file():
            raise FileNotFoundError(f"Unique old weight missing: {path}")
        digest = manifest.get("weights", {}).get(name, {}).get("sha256") or sha256_file(path)
        weights[name] = {"path": str(run/name), "bytes": path.stat().st_size, "sha256": digest,
                         "hash_method": "single archive verification; rename preserves inode"}
    tests = []
    for path in sorted(current.glob("test_*way_*shot.json")):
        payload = json.loads(path.read_text())
        if payload["checkpoint_sha256"] != weights["checkpoint.pth"]["sha256"]:
            raise ValueError("Old test/weight hash differs")
        tests.append({key: payload[key] for key in ("way", "shot", "episodes", "accuracy", "confidence_interval")})
    manifest["weights"] = weights
    manifest["old_actual_tests"] = tests
    manifest["old_missing_settings"] = [{"way": way, "shot": shot, "accuracy": None}
        for way in (5, 30) for shot in (1, 5) if not (current/f"test_{way}way_{shot}shot.json").exists()]
    write_json(MANIFEST, manifest)
    move(manifest, old_run, run, "both unique weights, original config/metrics/results, curves, actual execution snapshots")
    for name in ("data_protocol", "configs", "report.md", "results.json", "technical_checks.json", "preservation_check.json"):
        move(manifest, OLD/name, DEST/name, "historical protocol/source or unique audit evidence; bytes unchanged")
    manifest["old_evidence_archived"] = True
    write_json(MANIFEST, manifest)


def remove(manifest, path, reason):
    if not path.exists():
        return
    nodes = [path, *path.rglob("*")] if path.is_dir() else [path]
    inodes = {}
    for node in nodes:
        stat = node.lstat()
        key = (stat.st_dev, stat.st_ino)
        item = inodes.setdefault(key, {"links_removed": 0, "links": stat.st_nlink,
            "blocks": stat.st_blocks*512, "directory": node.is_dir()})
        item["links_removed"] += 1
    freed = sum(item["blocks"] for item in inodes.values()
                if item["directory"] or item["links_removed"] >= item["links"])
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    manifest["allocated_bytes_released_by_unlink"] += freed
    action(manifest, "delete", path, reason, disk_bytes_released=freed,
           accounting="allocated st_blocks; files with remaining hardlinks release zero content blocks")


def clean(manifest):
    prepared = json.loads((BASE/"prep/prepared.json").read_text())
    if not prepared["data_ready"] or prepared["training_started"]:
        raise ValueError("New author data must be ready without training before cleanup")
    if scoped_processes() != ([], []):
        raise RuntimeError("Old task or another process still uses the cleanup targets")
    for item in manifest["weights"].values():
        if not Path(item["path"]).exists() or Path(item["path"]).stat().st_size != item["bytes"]:
            raise ValueError("Archive weight must remain accessible before cleanup")
    required = [DEST/"run/config.yaml", DEST/"run/epoch_history.jsonl", DEST/"report.md",
                DEST/"data_protocol/split.json", DEST/"data_protocol/author_split_comparison.json"]
    if any(not path.exists() for path in required):
        raise ValueError("Required archive evidence absent")
    remove(manifest, OLD_DATA, "old DN4 derived images; author data has independent regular files/hardlinks")
    if OLD_DATA.parent.exists() and not any(OLD_DATA.parent.iterdir()):
        remove(manifest, OLD_DATA.parent, "empty old derived-data parent")
    for path in sorted((DEST/"run").glob("activations_epoch_*.json")):
        remove(manifest, path, "per-epoch activation diagnostics; original curves and gradients retained")
    remove(manifest, DEST/"data_protocol/author_splits", "originals moved to neutral authority; duplicate README is recoverable from Git")
    # Everything left in OLD is retired source/duplicate family summary or screen output.
    allowed = {"check.py", "compare_author_split.py", "prepare.py", "run.py", "report.py", "result.json", "session.log", "__pycache__", "runs"}
    if OLD.exists():
        for path in list(OLD.iterdir()):
            if path.name == "runs" and any(path.iterdir()):
                manifest["retained_nonessential_items"].append({"path": str(path), "reason": "additional old runs not covered by this archive; retained"})
            elif path.name in allowed:
                remove(manifest, path, "retired DN4 entry/source from commit 56627bf or duplicate output; execution patch preserved")
            else:
                manifest["retained_nonessential_items"].append({"path": str(path), "reason": "unexpected item; deletion scope not expanded"})
        if not any(OLD.iterdir()):
            remove(manifest, OLD, "empty retired experiment root")
    remove(manifest, ROOT/"tests/test_dogs_dn4_protocol.py", "tests solely for retired DN4 wrapper; original test code preserved by Git/run snapshot")
    if WATCH.exists():
        known = {"watch.py", "instructions.md", "state.json", "watch.lock"}
        for path in list(WATCH.iterdir()):
            if path.name in known:
                remove(manifest, path, "obsolete disabled DN4 wakeup control; no active watcher")
            else:
                manifest["retained_nonessential_items"].append({"path": str(path), "reason": "unknown runtime-control file; retained"})
        if not any(WATCH.iterdir()):
            remove(manifest, WATCH, "empty old DN4 watcher directory")
    # Crucially checks after donor deletion, including classes newly moved from training to eval.
    manifest["post_cleanup_data_validation"] = verify_data(Path(prepared["root"]), BASE/"prep", prepared["reuse"]["new_pre_classes"])
    manifest["after_protection"] = protection_snapshot()
    manifest["protected_items_unchanged"] = manifest["before_protection"] == manifest["after_protection"]
    if not manifest["protected_items_unchanged"]:
        raise ValueError("Protected data/results/environment metadata changed")
    manifest.update(status="completed", finished_utc=now(), dogs_data_ready=True, training_started=False,
                    accuracy=None, no_old_writers_confirmed=scoped_processes() == ([], []))
    manifest["retained_nonessential_items"].append({"path": str(ROOT/"experiments/sceg_dogs_baseline"),
        "reason": "historical synthetic/blocked evidence and helpers referenced by retained tests; not an author preparation dependency"})
    write_json(MANIFEST, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("archive", "clean"))
    args = parser.parse_args()
    manifest = load_manifest()
    archive_old(manifest) if args.phase == "archive" else clean(manifest)
    print(json.dumps({key: manifest.get(key) for key in ("status", "old_task_stopped", "old_evidence_archived",
        "author_sources_preserved", "dogs_data_ready", "allocated_bytes_released_by_unlink")}))


if __name__ == "__main__":
    main()
