"""Recorded invocation of existing tools.test; checkpoint never trains here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import torch
import tools.test as entry
from baseline import RecordedRun, PHASE_A, verify_protected, sha
from utils.config import load_config
from utils.experiment import write_json

EPISODES = 2000
PREFLIGHT = False


class EvaluationRun(RecordedRun):
    def __init__(self, config, run_id=None):
        if PREFLIGHT:
            config["purpose"] = "evaluation_io_preflight"
        super().__init__(config, run_id)

    def publish_result(self):
        if PREFLIGHT and self.metrics["status"] == "completed":
            self.metrics["status"] = "evaluation_io_preflight_completed"
            self.metrics["metric_scope"] = "few held-out episodes used only to verify evaluation I/O, not baseline results"
        super().publish_result()


def evaluation_config(path):
    config = load_config(path)
    config["eval"]["test_episodes"] = EPISODES
    return config


def reference_checkpoint(source, destination):
    """Record the evaluated checkpoint by reference, without another weight copy."""
    destination = Path(destination)
    destination.symlink_to(os.path.relpath(Path(source).resolve(), destination.parent))
    return destination


def verify_data_bytes():
    manifest = json.loads((PHASE_A / "data_manifest.json").read_text())
    image_hashes = json.loads((PHASE_A / "image_hashes.json").read_text())
    root = Path(manifest["root"])
    for relative, digest in image_hashes.items():
        if sha(root / relative) != digest:
            raise RuntimeError(f"Dataset changed: {relative}")
    pre = json.loads((ROOT / "data/provenance/pre_resized.json").read_text())["CUB_fewshot_raw"]
    for split in ("val_pre", "test_pre"):
        for relative, digest in pre[split]["files"].items():
            if sha(root / split / relative) != digest:
                raise RuntimeError(f"Prepared image changed: {split}/{relative}")


def main():
    global EPISODES, PREFLIGHT
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--way", type=int, choices=(5, 50), required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    EPISODES, PREFLIGHT = args.episodes, args.preflight
    if not PREFLIGHT and EPISODES != 2000:
        raise ValueError("Formal baseline tests require 2000 episodes per setting")
    verify_protected()
    verify_data_bytes()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not PREFLIGHT:
        parent = args.checkpoint.parent
        training_metrics = json.loads((parent / "metrics.json").read_text())
        assert training_metrics["full_training_completed"] and training_metrics["completed_epochs"] == 800
        assert payload["best_epoch"] is not None and payload["completed_epochs"] == payload["best_epoch"]
        assert args.checkpoint.name == "checkpoint_best.pth"
    del payload
    entry.ExperimentRun = EvaluationRun
    entry.load_config = evaluation_config
    sys.argv = ["tools.test", "--config", "experiments/sceg_original/configs/cub.yaml",
                "--run-id", args.run_id, "--checkpoint", str(args.checkpoint), "--way", str(args.way)]
    copy2 = shutil.copy2
    def copy_metadata_or_reference_checkpoint(source, destination, *args, **kwargs):
        if Path(destination).name == "checkpoint.pth" and Path(source).resolve() == args_checkpoint:
            return reference_checkpoint(source, destination)
        return copy2(source, destination, *args, **kwargs)
    args_checkpoint = args.checkpoint.resolve()
    with patch("shutil.copy2", copy_metadata_or_reference_checkpoint):
        entry.main()
    run = ROOT / "experiments/sceg_original/runs" / args.run_id
    records = json.loads((run / "evaluation.json").read_text())
    assert len(records) == 2 and all(r["episodes"] == EPISODES for r in records)
    assert {r["shot"] for r in records} == {1, 5}
    write_json(run / "evaluation_checks.json", {"status": "passed", "preflight": PREFLIGHT,
        "checkpoint_strict_load_passed": True, "episodes_per_setting": EPISODES,
        "way": args.way, "checkpoint_sha256": sha(args.checkpoint), "dataset_rehashed": True})


if __name__ == "__main__":
    main()
