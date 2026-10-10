"""One-time removal of this baseline's redundant weight archives only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from utils.experiment import write_json

REPORT = Path(__file__).resolve().parent
RUNS = ROOT / "experiments/sceg_original/runs"
FORMAL = RUNS / "20261009_cub800_seed42_formal"
DRY_RUNS = [RUNS / name for name in (
    "20261009_cub800_seed42_dry_epoch1",
    "20261009_cub800_seed42_resume_epoch2",
    "20261009_cub800_seed42_eval_io_preflight")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_bytes():
    output = subprocess.check_output(["rtk", "proxy", "du", "-sx", "-B1", str(ROOT)], text=True)
    return int(output.split()[0])


def main():
    for command_line in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            arguments = command_line.read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if any(Path(arg).name in {"baseline.py", "workflow.py", "evaluate.py"} or arg in
               {"tools.train", "tools.test"} for arg in arguments):
            raise RuntimeError("Training/evaluation is active; leave its files alone")
    last, best = FORMAL / "latest.pth", FORMAL / "checkpoint.pth"
    last_state = torch.load(last, map_location="cpu", weights_only=False)
    best_state = torch.load(best, map_location="cpu", weights_only=False)
    recorded = json.loads((FORMAL / "metrics.json").read_text())["epoch_history"][-1]
    assert last_state["completed_epochs"] == recorded["epoch"] == 122
    assert sha(last) == recorded["checkpoint_sha256"]
    assert best_state["completed_epochs"] == best_state["best_epoch"] == 100
    kept = {str(path.relative_to(ROOT)): {"sha256": sha(path), "bytes": path.stat().st_size}
            for path in (last, best)}
    candidates = list((FORMAL / "checkpoints").glob("*.pth"))
    candidates += list(FORMAL.glob("rng_state_epoch_*.pt"))
    for run in DRY_RUNS:
        candidates += list(run.glob("*.pth")) + list(run.glob("*.pt"))
        candidates += list((run / "checkpoints").glob("*.pth"))
    candidates = sorted(set(candidates))
    entries = [{"path": str(path.relative_to(ROOT)),
                "bytes": path.lstat().st_size if not path.is_symlink() else 0,
                "symlink": path.is_symlink()} for path in candidates]
    before = project_bytes()
    write_json(REPORT / "cleanup_manifest.json", {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "user requested light retention and removal of old heavy archives",
        "kept_checkpoints": kept, "removed_paths": entries,
        "light_evidence_policy": "preserve all configs, commands, environments, source/data manifests, epoch/validation/evaluation metrics and logs",
        "old_sceg_debug_and_preflight_policy": "untouched"})
    # Repair aliases before removing their obsolete archived targets.
    for alias, target in (("checkpoint_last.pth", "latest.pth"),
                          ("checkpoint_best.pth", "checkpoint.pth")):
        (FORMAL / alias).unlink(missing_ok=True)
        (FORMAL / alias).symlink_to(target)
    rng = FORMAL / "rng_state.pt.part"
    torch.save(last_state["rng"], rng)
    rng.replace(FORMAL / "rng_state.pt")
    del last_state, best_state
    for path in candidates:
        path.unlink(missing_ok=True)
    for run in [FORMAL, *DRY_RUNS]:
        directory = run / "checkpoints"
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
    for relative, record in kept.items():
        assert sha(ROOT / relative) == record["sha256"]
    after = project_bytes()
    result = {"status": "completed", "training_resumed": False,
        "project_allocated_bytes_before": before, "project_allocated_bytes_after": after,
        "reclaimed_allocated_bytes": before - after, "removed_path_count": len(entries),
        "remaining_baseline_full_checkpoint_count": 2, "kept_checkpoints": kept,
        "last_alias": "checkpoint_last.pth -> latest.pth",
        "best_alias": "checkpoint_best.pth -> checkpoint.pth",
        "rng_state": "one current rng_state.pt, recovered from verified latest checkpoint"}
    write_json(REPORT / "cleanup_result.json", result)
    write_json(FORMAL / "checkpoint_manifest.json", {
        "retention_policy": "last_and_best_only", "latest_completed_epochs": 122,
        "best_epoch": 100, "checkpoints": kept, "cleanup_manifest": str((REPORT / "cleanup_manifest.json").relative_to(ROOT))})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
