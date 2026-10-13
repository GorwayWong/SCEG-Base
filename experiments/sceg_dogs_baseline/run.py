"""Thin Dogs training/suite entrypoint; uses the existing trainer and evaluator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from evaluators import evaluate_episodes
from models import SCEG
from trainers.trainer import train, restore_rng
from tools.prepare_data import sha256_file
from utils.config import load_config, ModelConfig
from utils.experiment import ExperimentRun, write_json
from utils.reproducibility import seed_everything, rng_state
from .prepare import BASE, check_prepared


class DogsRun(ExperimentRun):
    """Only recording additions; checkpoint policy and model training are unchanged."""
    def __init__(self, config, run_id, *, reuse_existing=False):
        if config["experiment"] not in {"sceg_dogs_baseline", "sceg_dogs_dn4"}:
            raise ValueError("Dogs entrypoint requires an isolated Dogs experiment")
        self.dn4 = config["experiment"] == "sceg_dogs_dn4"
        self.config = config
        self.started = time.perf_counter()
        self.last_epoch_time = self.started
        self.prior_elapsed = 0.0
        if reuse_existing:
            metrics_path = Path(__file__).resolve().parents[2]/"experiments"/config["experiment"]/"runs"/run_id/"metrics.json"
            self.prior_elapsed = json.loads(metrics_path.read_text()).get("elapsed_seconds", 0.0)
        super().__init__(config, run_id, reuse_existing=reuse_existing)
        if reuse_existing:
            return
        # source_manifest() and git diff HEAD omit untracked experiments/ sources.
        patches = []
        base = self.root/"experiments"/config["experiment"]
        sources = [*base.glob("*.py"), base/"configs/dogs.yaml"]
        if self.dn4:
            sources.append(self.root/"tests/test_dogs_dn4_protocol.py")
        for path in sorted(sources):
            if self.dn4 and subprocess.run(["rtk", "proxy", "git", "ls-files", "--error-unmatch",
                    str(path.relative_to(self.root))], cwd=self.root, capture_output=True).returncode == 0:
                continue  # HEAD + git_diff.patch already covers tracked sources.
            result = subprocess.run(["rtk", "proxy", "git", "diff", "--no-index", "--",
                                     "/dev/null", str(path.relative_to(self.root))],
                                    cwd=self.root, capture_output=True, text=True)
            if result.returncode not in (0, 1):
                raise RuntimeError(result.stderr)
            patches.append(result.stdout)
        (self.path/"experiment_source.patch").write_text("".join(patches))
        write_json(self.path/"experiment_source_manifest.json", {
            str(path.relative_to(self.root)): sha256_file(path) for path in sources})

    def update_metrics(self, **values):
        if self.dn4:
            values.update(elapsed_seconds=self.prior_elapsed+time.perf_counter()-self.started,
                          gpu_peak_allocated_bytes=max(self.metrics.get("gpu_peak_allocated_bytes", 0) or 0,
                          torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0))
        super().update_metrics(**values)
        if "completed_epochs" in values:
            now = time.perf_counter()
            with (self.path/"epoch_history.jsonl").open("a") as stream:
                stream.write(json.dumps({**values, "epoch_seconds": now-self.last_epoch_time,
                                         "elapsed_seconds": self.prior_elapsed+now-self.started})+"\n")
            self.last_epoch_time = now
        if self.dn4 and (self.path/"git_commit.txt").exists():
            self.publish_result()


def evaluation_rng():
    state = rng_state()
    state["numpy"] = [state["numpy"][0], state["numpy"][1].tolist(), *state["numpy"][2:]]
    state["torch"] = state["torch"].tolist()
    state["cuda"] = [tensor.tolist() for tensor in state["cuda"]]
    return state


def restore_evaluation_rng(state):
    state = dict(state)
    state["python"] = (state["python"][0], tuple(state["python"][1]), state["python"][2])
    state["numpy"] = (state["numpy"][0], np.array(state["numpy"][1], dtype=np.uint32), *state["numpy"][2:])
    state["torch"] = torch.tensor(state["torch"], dtype=torch.uint8)
    state["cuda"] = [torch.tensor(value, dtype=torch.uint8) for value in state["cuda"]]
    restore_rng(state)


def checkpoint_records(run):
    records = {}
    for name in ("latest.pth", "checkpoint.pth"):
        path = run.path/name
        if path.exists():
            records[name] = {"path": str(path), "bytes": path.stat().st_size,
                             "sha256": sha256_file(path)}
    write_json(run.path/"checkpoint_manifest.json", records)
    return records


def complete_suite(model, config, run):
    """Reuse trainer's two 5-way files; evaluate only missing 30-way settings."""
    best = run.path/"checkpoint.pth"
    source_hash = sha256_file(best)
    payload = torch.load(best, map_location="cpu", weights_only=False)
    if payload["best_epoch"] is None:
        raise RuntimeError("No validation-selected best checkpoint")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    dn4 = config.get("experiment") == "sceg_dogs_dn4"
    results = []
    for way in (5, 30):
        for shot in (1, 5):
            path = run.path/f"test_{way}way_{shot}shot.json"
            if path.exists():
                result = json.loads(path.read_text())
                if (result["way"], result["shot"], result["query"], result["episodes"]) != (
                        way, shot, 16, config["eval"]["test_episodes"]):
                    raise ValueError("Existing test result has incompatible protocol")
                if result.get("checkpoint_sha256", source_hash) != source_hash:
                    raise ValueError("Existing test result uses a different checkpoint")
                if dn4 and "rng_after" in result:
                    restore_evaluation_rng(result["rng_after"])
            else:
                started = time.perf_counter()
                result = evaluate_episodes(model, config, "test", way, shot,
                                           config["eval"]["test_episodes"])
                result["evaluation_seconds"] = time.perf_counter()-started
                if dn4:
                    result["rng_after"] = evaluation_rng()
                else:
                    write_json(path, result)
            if dn4:
                result.update(checkpoint_sha256=source_hash, best_epoch=payload["best_epoch"])
                write_json(path, result)
            result = {key: value for key, value in result.items()
                      if key not in {"episode_indices", "sample_index_paths", "episode_accuracies", "rng_after"}}
            result["checkpoint_sha256"] = source_hash
            results.append(result)
    run.update_metrics(status="completed", full_training_completed=True,
                       full_evaluation_completed=True, evaluation=results,
                       best_epoch=payload["best_epoch"])
    write_json(run.path/"evaluation.json", results)


def main(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config or BASE/"configs/dogs.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Continue missing evaluations in a fully trained existing run")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["model"]["dataset"] != "stanford_dog" or config["model"]["base_classes"] != 70:
        raise ValueError("Dogs baseline requires stanford_dog and a 70-class global head")
    if (config["train"]["epochs"], config["train"]["validation_every"], config["eval"]["way"],
            config["eval"]["validation_episodes"], config["eval"]["test_episodes"],
            config["eval"]["query"], config["eval"]["many_way"]) != (800, 20, 5, 1000, 2000, 16, 30):
        raise ValueError("Formal Dogs protocol differs from frozen baseline configuration")
    config["purpose"] = "formal_dogs_dn4" if config["experiment"] == "sceg_dogs_dn4" else "formal_dogs_baseline"
    config["checkpoint_source"] = str(args.resume.resolve()) if args.resume else None
    config["checkpoint_source_sha256"] = sha256_file(args.resume) if args.resume else None
    gate = Path("experiments/sceg_debug/result.json")
    if json.loads(gate.read_text()).get("status") != "passed":
        raise RuntimeError("Existing CUB equivalence/overfit gate is not passed")
    config["gates"] = {"overfit_result": str(gate), "overfit_sha256": sha256_file(gate),
                       "official_inactive_exception_accepted": True,
                       "strict_all_parameter_gradient_passed": False}
    expected_path = Path("experiments")/config["experiment"]/"runs"/args.run_id
    reuse = bool(args.resume and args.resume.resolve().parent == expected_path.resolve())
    if args.evaluate_only and not reuse:
        raise ValueError("--evaluate-only requires --resume <same run>/latest.pth")
    run = DogsRun(config, args.run_id, reuse_existing=reuse)
    if config["experiment"] == "sceg_dogs_dn4":
        run.update_metrics(process_id=os.getpid(), screen_session=os.environ.get("STY"),
                           full_training_completed=bool(run.metrics.get("full_training_completed", False)),
                           full_evaluation_completed=bool(run.metrics.get("full_evaluation_completed", False)))
    exit_code = 0
    try:
        data_record = check_prepared(config)
        write_json(run.path/"data_manifest.json", {
            "prepared_manifest": config["data_protocol"]["prepared_manifest"],
            "sha256": sha256_file(config["data_protocol"]["prepared_manifest"]),
            "split_manifest": config["data_protocol"]["split_manifest"],
            "split_sha256": data_record["split_manifest_sha256"],
            "integrity_check": "inventory + size/mtime, no repeat image hashing"})
        seed_everything(config["seed"], deterministic=config["runtime"]["deterministic"],
                        allow_tf32=config["runtime"]["allow_tf32"],
                        warn_only=config["runtime"]["deterministic_warn_only"])
        torch.set_num_threads(config["runtime"]["cpu_threads"])
        model = SCEG(ModelConfig.from_dict(config["model"])).to(config["runtime"]["device"])
        # Add timings around the existing evaluator; no sampler/forward changes.
        import trainers.trainer as trainer_module
        original_evaluate = trainer_module.evaluate_episodes
        test_checkpoint_hash = None
        def timed_evaluate(*evaluation_args, **evaluation_kwargs):
            nonlocal test_checkpoint_hash
            started = time.perf_counter()
            result = original_evaluate(*evaluation_args, **evaluation_kwargs)
            result["evaluation_seconds"] = time.perf_counter()-started
            if config["experiment"] == "sceg_dogs_dn4":
                result["rng_after"] = evaluation_rng()
                if result["split"] == "test":
                    test_checkpoint_hash = test_checkpoint_hash or sha256_file(run.path/"checkpoint.pth")
                    result.update(checkpoint_sha256=test_checkpoint_hash, best_epoch=run.metrics["best_epoch"])
            return result
        trainer_module.evaluate_episodes = timed_evaluate
        train_started = time.perf_counter()
        try:
            if args.evaluate_only:
                payload = torch.load(args.resume, map_location="cpu", weights_only=False)
                if payload["completed_epochs"] != config["train"]["epochs"]:
                    raise ValueError("Evaluation-only requires complete formal training")
                restore_rng(payload["rng"])
                run.update_metrics(full_training_completed=True)
            else:
                train(model, config, run, args.resume)
        finally:
            trainer_module.evaluate_episodes = original_evaluate
        if not args.evaluate_only:
            five_way_seconds = sum(json.loads((run.path/f"test_5way_{shot}shot.json").read_text())[
                "evaluation_seconds"] for shot in (1, 5))
            if config["experiment"] == "sceg_dogs_dn4":
                training_seconds = sum(json.loads(line)["epoch_seconds"]
                    for line in (run.path/"epoch_history.jsonl").read_text().splitlines())
            else:
                training_seconds = time.perf_counter()-train_started-five_way_seconds
            run.update_metrics(training_with_validation_seconds=training_seconds)
        complete_suite(model, config, run)
        run.update_metrics(test_seconds=sum(item["evaluation_seconds"] for item in run.metrics["evaluation"]))
        checkpoint_records(run)
    except RuntimeError as error:
        blocked = str(error).startswith("BLOCKED:")
        run.update_metrics(status="blocked" if blocked else "failed", error=str(error),
                           full_training_completed=bool(run.metrics.get("full_training_completed", False)),
                           full_evaluation_completed=False)
        run.logger.exception("Dogs execution stopped")
        exit_code = 2 if blocked else 1
    except BaseException as error:
        run.update_metrics(status="failed", error=f"{type(error).__name__}: {error}")
        run.publish_result()
        raise
    run.update_metrics(elapsed_seconds=time.perf_counter()-run.started,
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated()
                       if torch.cuda.is_available() else None)
    run.publish_result()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
