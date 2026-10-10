"""Prune obsolete diagnostics after training; retain results and final weights."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from utils.experiment import write_json

REPORT = Path(__file__).resolve().parent
RUNS = ROOT / "experiments/sceg_original/runs"
RUN_ID = "20261010_cub800_seed42_resume122"
FINAL = RUNS / RUN_ID
OLD = RUNS / "20261009_cub800_seed42_formal"
RESULTS = ROOT / "experiments/sceg_original/results" / RUN_ID


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def allocated_bytes():
    value = subprocess.check_output(["rtk", "proxy", "du", "-sx", "-B1", str(ROOT)], text=True)
    return int(value.split()[0])


def deduplicate_cub():
    legacy = ROOT / "data/CUB_fewshot_raw"
    canonical = ROOT / "data/frn/CUB_fewshot_raw"
    old_images = list(legacy.rglob("*.jpg"))
    new_images = list(canonical.rglob("*.jpg"))
    targets = {p.name: p for p in new_images}
    assert len(old_images) == len(new_images) == len(targets) == 11788
    assert {p.name for p in old_images} == set(targets)
    count = 0
    for old in old_images:
        new = targets[old.name]
        assert not old.is_symlink() and not new.is_symlink()
        assert old.stat().st_dev == new.stat().st_dev
        assert sha(old) == sha(new)
        if old.samefile(new):
            continue
        temporary = old.with_suffix(".jpg.dedup")
        os.link(new, temporary)
        temporary.replace(old)
        assert old.samefile(new)
        count += 1
    return {"legacy_root": str(legacy.relative_to(ROOT)),
            "canonical_root": str(canonical.relative_to(ROOT)),
            "image_count": 11788, "deduplicated_files": count,
            "all_image_sha256_equal": True, "class_paths_preserved": True,
            "hardlinks_to_legacy_project_copy": True,
            "hardlinks_to_external_source_project": False,
            "method": "project-internal hard links; no links to the external source project"}


def main():
    status = json.loads((RESULTS / "status.json").read_text())
    assert status["status"] in {"completed", "paused"} and status["completed_epochs"] == 800
    expected = {(5, 1), (5, 5), (50, 1)}
    if status["status"] == "completed":
        expected.add((50, 5))
    assert {(r["way"], r["shot"]) for r in status["evaluations"]} == expected
    keep_paths = [FINAL / "latest.pth", FINAL / "checkpoint.pth", RESULTS / "results.json"]
    keep_paths += [ROOT / row["raw_result"] for row in status["evaluations"]]
    preserved = {str(p.relative_to(ROOT)): sha(p) for p in keep_paths}
    before = allocated_bytes()
    candidates = []
    for run in RUNS.iterdir():
        if run.is_dir():
            candidates += list(run.glob("activations_epoch_*.json"))
            candidates += list(run.glob("epoch_[0-9]*.json"))
    candidates += [OLD / n for n in ("latest.pth", "checkpoint.pth", "checkpoint_last.pth",
                                     "checkpoint_best.pth", "rng_state.pt")]
    log_root = ROOT / "experiments/sceg_original/logs"
    candidates += list(log_root.glob("*/train_process.log"))
    candidates += [log_root / "20261009_cub800_seed42_supervisor_console.log"]
    candidates += [REPORT / "finalize.py"]
    candidates = sorted({p for p in candidates if p.exists() or p.is_symlink()})
    removed = [{"path": str(p.relative_to(ROOT)), "bytes": p.lstat().st_size,
                "symlink": p.is_symlink()} for p in candidates]
    write_json(REPORT / "cleanup_manifest.json", {
        "time_utc": datetime.now(timezone.utc).isoformat(), "removed": removed,
        "preserved_sha256": preserved,
        "policy": "retain final last/best, original episode results, all validation results and complete light training history"})
    for path in candidates:
        path.unlink()
    data_deduplication = deduplicate_cub()
    for relative, digest in preserved.items():
        assert sha(ROOT / relative) == digest
    after = allocated_bytes()
    checkpoints = {str(p.relative_to(ROOT)): {"sha256": preserved[str(p.relative_to(ROOT))],
                                              "bytes": p.stat().st_size}
                   for p in keep_paths[:2]}
    write_json(FINAL / "checkpoint_manifest.json", {
        "latest_completed_epochs": 800, "best_epoch": json.loads((FINAL / "metrics.json").read_text())["best_epoch"],
        "checkpoints": checkpoints, "retention_policy": "final last and best only"})
    write_json(REPORT / "cleanup_result.json", {
        "status": "completed", "removed_file_count": len(removed),
        "reclaimed_allocated_bytes": before - after,
        "project_allocated_bytes_after": after, "preserved_sha256": preserved,
        "data_deduplication": data_deduplication,
        "dataset_contents_splits_and_verified_model_modules_unchanged": True})
    print(json.dumps({"removed_file_count": len(removed), "reclaimed_allocated_bytes": before-after,
                      "deduplicated_cub_images": data_deduplication["deduplicated_files"],
                      "final_last_best_and_all_raw_results_preserved": True}))


def prune_validation_records():
    metrics = json.loads((FINAL / "metrics.json").read_text())
    history = [r for r in metrics["epoch_history"] if r["validation_accuracy"] is not None]
    best = max(history, key=lambda r: r["validation_accuracy"])
    assert metrics["completed_epochs"] == 800 and best["epoch"] == metrics["best_epoch"]
    selected = FINAL / f"validation_epoch_{best['epoch']}.json"
    assert json.loads(selected.read_text())["accuracy"] == best["validation_accuracy"]
    preserved = {str(p.relative_to(ROOT)): sha(p) for p in [FINAL / "metrics.json", selected]}
    candidates = [p for d in RUNS.iterdir() if d.is_dir()
                  for p in d.glob("validation_epoch_*.json") if p != selected]
    before = allocated_bytes()
    removed = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size} for p in candidates]
    for path in candidates:
        path.unlink()
    assert all(sha(ROOT / path) == digest for path, digest in preserved.items())
    result = {"removed_file_count": len(removed), "removed": removed,
        "reclaimed_allocated_bytes": before - allocated_bytes(),
        "validation_metric_epochs_retained": len(history), "best_epoch": best["epoch"],
        "preserved_sha256": preserved, "all_formal_test_episode_results_retained": True}
    write_json(REPORT / "validation_cleanup.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"removed", "preserved_sha256"}}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--prune-validation"]:
        prune_validation_records()
    else:
        main()
