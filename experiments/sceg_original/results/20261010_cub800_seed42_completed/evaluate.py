"""Recorded invocation of existing tools.test; checkpoint never trains here."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import torch
import numpy as np
import tools.test as entry
import evaluators.fewshot_eval as evaluator
from baseline import RecordedRun, PHASE_A, verify_protected, sha
from utils.config import load_config
from utils.experiment import write_json

EPISODES = 2000
PREFLIGHT = False
SHOTS = (1, 5)


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
    config["eval"]["shots"] = list(SHOTS)
    return config


class ProgressLoader:
    """Log after inference returns to the loader; leave batch contents unchanged."""
    def __init__(self, loader, run, episodes):
        self.loader, self.run, self.episodes = loader, run, episodes

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        start = last = time.monotonic()
        for completed, batch in enumerate(self.loader, 1):
            yield batch
            now = time.monotonic()
            if completed == 1 or completed == self.episodes or now - last >= 60:
                elapsed = now - start
                record = {"status": "completed" if completed == self.episodes else "running",
                    "completed_episodes": completed, "requested_episodes": self.episodes,
                    "elapsed_seconds": elapsed, "seconds_per_episode": elapsed / completed,
                    "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated()}
                write_json(self.run / "evaluation_progress.json", record)
                logging.getLogger(f"sceg.sceg_original.{self.run.name}").info("episodes=%d/%d elapsed=%.1fs seconds_per_episode=%.3f gpu_max_allocated=%d",
                    completed, self.episodes, elapsed, record["seconds_per_episode"], record["gpu_max_allocated_bytes"])
                last = now


def replay_completed_sampler(config, split, way, episodes, source, run, make_loader):
    raw = json.loads(source.read_text())
    assert (raw["way"], raw["shot"], raw["episodes"], raw["seed"], raw["query"]) == (way, 1, episodes, 42, 16)
    loader = make_loader(config, split, way=way, shot=1, episodes=episodes)
    paths = [str(Path(path).relative_to(loader.dataset.root)) for path, _ in loader.dataset.samples]
    assert paths == raw["sample_index_paths"]
    count = 0
    # Iterate the original batch sampler only: no image loading, model forward or accuracy computation.
    for count, indices in enumerate(loader.batch_sampler, 1):
        assert list(map(int, indices)) == raw["episode_indices"][count - 1]
    assert count == len(raw["episode_indices"]) == episodes
    state = np.random.get_state()
    state_sha = hashlib.sha256(state[1].tobytes() + repr((state[0], *state[2:])).encode()).hexdigest()
    write_json(run / "rng_replay.json", {"source": str(source.relative_to(ROOT)), "source_sha256": sha(source),
        "replayed_sampler_episodes": count, "all_indices_match_completed_1shot": True,
        "numpy_state_after_replay_sha256": state_sha, "model_forwards_replayed": 0,
        "note": "Preserve original [1,5] NumPy sampling continuation; eval transforms are deterministic and DropBlock is disabled."})
    logging.getLogger(f"sceg.sceg_original.{run.name}").info("Replayed %d original 1-shot sampler episodes; every index matched; no model forward.", count)


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
    global EPISODES, PREFLIGHT, SHOTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--way", type=int, choices=(5, 50), required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--shots", type=int, nargs="+", choices=(1, 5), default=(1, 5))
    parser.add_argument("--replay-1shot", type=Path)
    args = parser.parse_args()
    EPISODES, PREFLIGHT = args.episodes, args.preflight
    SHOTS = tuple(args.shots)
    if args.replay_1shot and (SHOTS != (5,) or args.way != 50):
        raise ValueError("Sampler continuation replay is for the missing 50-way 5-shot setting only")
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
    original_evaluate = entry.evaluate_episodes
    original_make_loader = evaluator.make_loader
    def recorded_evaluate(model, config, split, way, shot, episodes):
        run = ROOT / "experiments/sceg_original/runs" / args.run_id
        write_json(run / "evaluation_adapter.json", {"path": str(Path(__file__).relative_to(ROOT)),
            "sha256": sha(__file__), "requested_shots": list(SHOTS),
            "entrypoint": "tools.test.main -> evaluators.fewshot_eval.evaluate_episodes (unchanged)",
            "rng_protocol": "original 1-shot sampler replay then 5-shot" if args.replay_1shot else "existing shot loop"})
        if args.replay_1shot:
            replay_completed_sampler(config, split, way, episodes, args.replay_1shot.resolve(), run, original_make_loader)
        def recorded_loader(*arguments, **keywords):
            return ProgressLoader(original_make_loader(*arguments, **keywords), run, episodes)
        with patch.object(evaluator, "make_loader", recorded_loader):
            return original_evaluate(model, config, split, way, shot, episodes)
    entry.evaluate_episodes = recorded_evaluate
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
    assert len(records) == len(SHOTS) and all(r["episodes"] == EPISODES for r in records)
    assert {r["shot"] for r in records} == set(SHOTS)
    write_json(run / "evaluation_checks.json", {"status": "passed", "preflight": PREFLIGHT,
        "checkpoint_strict_load_passed": True, "episodes_per_setting": EPISODES,
        "way": args.way, "shots": list(SHOTS), "checkpoint_sha256": sha(args.checkpoint), "dataset_rehashed": True})


if __name__ == "__main__":
    main()
