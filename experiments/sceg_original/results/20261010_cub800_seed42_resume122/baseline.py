"""Experiment-only recording adapter around the existing tools.train entrypoint.

No model, loss, sampler, augmentation, optimizer or scheduler is replaced.
Phase stops happen only after the existing trainer saves a complete epoch.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
import yaml
import tools.train as entry
from trainers.trainer import train as existing_train, dropblock_counters
from utils.experiment import ExperimentRun, write_json
from utils.reproducibility import rng_state

PHASE_A = ROOT / "experiments/sceg_original/runs/20261009_cub800_seed42_phase_a_verified"
STOP_AFTER = None
MANY_WAY_PREFLIGHT = False


class PhaseComplete(Exception):
    pass


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_protected():
    for name in ("protected_source_manifest.json", "existing_artifact_hashes.json"):
        expected = json.loads((PHASE_A / name).read_text())
        changed = [path for path, digest in expected.items() if sha(ROOT / path) != digest]
        if changed:
            raise RuntimeError(f"Protected files changed: {changed}")


def link_checkpoint(source, target):
    temporary = target.with_suffix(target.suffix + ".link")
    temporary.symlink_to(source.relative_to(target.parent))
    temporary.replace(target)


class RecordedRun(ExperimentRun):
    def __init__(self, config, run_id=None):
        self.epoch_started = time.monotonic()
        self.epoch_records = []
        super().__init__(config, run_id)
        self.config = config
        shutil.copy2(self.path / "config.yaml", self.path / "config_resolved.yaml")
        for name in ("data_manifest.json", "image_hashes.json", "phase_a_checks.json"):
            shutil.copy2(PHASE_A / name, self.path / name)
        write_json(self.path / "execution_adapter.json", {
            "path": str(Path(__file__).relative_to(ROOT)), "sha256": sha(__file__),
            "entrypoint": "tools.train.main -> trainers.trainer.train (unchanged)",
            "stop_after_completed_epoch": STOP_AFTER, "phase_a": str(PHASE_A.relative_to(ROOT)),
            "canonical_checkpoint_policy": "only rolling last and validation-selected best; no epoch archives",
            "rng_policy": "one current rng_state.pt; complete RNG also stored in last checkpoint",
            "family_result_policy": "never overwrite existing experiments/sceg_original/result.json"})
        logging.captureWarnings(True)

    def checkpoint(self, payload):
        # latest.pth already covers provisional epochs; only retain a selected best.
        if payload["best_epoch"] is None:
            return
        super().checkpoint(payload)
        if not (self.path / "checkpoint_best.pth").is_symlink():
            link_checkpoint(self.path / "checkpoint.pth", self.path / "checkpoint_best.pth")

    def update_metrics(self, **values):
        super().update_metrics(**values)
        if "completed_epochs" not in values:
            return
        epoch = values["completed_epochs"]
        if not (self.path / "checkpoint_last.pth").is_symlink():
            link_checkpoint(self.path / "latest.pth", self.path / "checkpoint_last.pth")
        rng_path = self.path / "rng_state.pt.part"
        torch.save(rng_state(), rng_path)
        rng_path.replace(self.path / "rng_state.pt")
        validation_path = self.path / f"validation_epoch_{epoch}.json"
        validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
        record = {
            "epoch": epoch, "step": values["step"], "episodes": values["epoch_episodes"],
            "train_loss": values["train_loss"], "train_episode_accuracy": values["train_accuracy"],
            "learning_rate_used": self.config["train"]["lr"] * self.config["train"]["scheduler_gamma"] **
                ((epoch - 1) // self.config["train"]["scheduler_step"]),
            "learning_rate_next_epoch": values["learning_rate"],
            "validation_accuracy": None if validation is None else validation["accuracy"],
            "validation_confidence_interval": None if validation is None else validation["confidence_interval"],
            "validation_status": "not_scheduled_official_every20" if validation is None else "completed",
            "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_max_reserved_bytes": torch.cuda.max_memory_reserved(),
            "elapsed_seconds": time.monotonic() - self.epoch_started,
            "checkpoint_completed_epochs": epoch,
            "checkpoint_policy": "last_and_best_only",
        }
        write_json(self.path / f"epoch_{epoch:04d}.json", record)
        self.epoch_records.append(record)
        with (self.path / "epoch_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        super().update_metrics(epoch_history=self.epoch_records)
        self.logger.info("epoch=%d episodes=%d loss=%.6f train_acc=%.4f lr_used=%.6g val_acc=%s val_ci=%s gpu_allocated=%d elapsed=%.2fs",
            epoch, record["episodes"], record["train_loss"], record["train_episode_accuracy"],
            record["learning_rate_used"], record["validation_accuracy"], record["validation_confidence_interval"],
            record["gpu_max_allocated_bytes"], record["elapsed_seconds"])
        self.epoch_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        if STOP_AFTER is None and epoch == self.config["train"]["epochs"]:
            write_json(self.path / "phase_c_checks.json", {
                "status": "passed", "completed_epochs": epoch, "full_training_completed": True,
                "best_epoch": values["best_epoch"], "best_validation": values["best_validation"],
                "checkpoint_best_sha256": sha(self.path / "checkpoint_best.pth"),
                "formal_testing_status": "next: existing trainer automatically evaluates best checkpoint 5-way 1/5-shot"})
            self.logger.info("Phase C completed: %d epochs; best_epoch=%s best_val=%.4f. Phase D: 5-way tests begin.",
                             epoch, values["best_epoch"], values["best_validation"])
        if STOP_AFTER is not None and epoch >= STOP_AFTER:
            raise PhaseComplete()

    def publish_result(self):
        summary = dict(self.metrics)
        summary.update(run_id=self.path.name,
                       commit=(self.path / "git_commit.txt").read_text().strip(),
                       config_sha256=sha(self.path / "config.yaml"))
        write_json(self.path / "results.json", summary)
        verify_protected()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def profile_forward(model, config, run):
    # Evaluation mode + forked RNG + restored official eval-incremented counters.
    # This records FLOPs without perturbing the training state or BN statistics.
    counters = dropblock_counters(model)
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            images = torch.zeros((200, 3, 84, 84), device=config["runtime"]["device"])
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                        record_shapes=True, with_flops=True) as profile:
                model(images)
            flops = sum(event.flops for event in profile.key_averages())
            write_json(run.path / "flops.json", {
                "input_shape": list(images.shape), "mode": "eval", "forward_flops": flops,
                "per_image_flops_average": flops / 200,
                "scope": "PyTorch profiler counted supported convolution and matrix multiplication operations; unsupported elementwise/reduction/pooling FLOPs excluded",
                "multiply_add_convention": "2 FLOPs per multiply-add", "backward_included": False})
    finally:
        for name, module in model.named_modules():
            if name in counters:
                module.num_batches_tracked = counters[name]
        model.train(was_training)


def recorded_train(model, config, run, resume=None):
    before = parameter_hash(model) if resume is None or STOP_AFTER is not None else None
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        write_json(run.path / "resume_input.json", {
            "source": str(resume.resolve()), "sha256": sha(resume),
            "completed_epochs": payload["completed_epochs"], "step": payload["step"],
            "scheduler_last_epoch": payload["scheduler"]["last_epoch"],
            "optimizer_lr": payload["optimizer"]["param_groups"][0]["lr"],
            "rng_restored_by": "existing trainers.trainer.restore_rng",
            "dropblock_counters": payload["dropblock_counters"]})
        previous_run = resume.resolve().parent
        previous_metrics = json.loads((previous_run / "metrics.json").read_text())
        run.epoch_records = [record for record in previous_metrics.get("epoch_history", [])
                             if record["epoch"] <= payload["completed_epochs"]]
        ExperimentRun.update_metrics(run, completed_epochs=payload["completed_epochs"],
            step=payload["step"], best_epoch=payload["best_epoch"],
            best_validation=payload["best_validation"],
            initial_completed_epochs=payload["completed_epochs"],
            resumed_from=str(previous_run), epoch_history=run.epoch_records)
        write_json(run.path / "continuation.json", {
            "previous_run": str(previous_run.relative_to(ROOT)),
            "initial_completed_epochs": payload["completed_epochs"],
            "initial_optimizer_steps": payload["step"],
            "prior_epoch_metrics": str((previous_run / "metrics.json").relative_to(ROOT)),
            "current_epoch_metrics": "metrics.json",
            "checkpoint_reference": "resume_input.json"})
        if (previous_run / "flops.json").exists():
            shutil.copy2(previous_run / "flops.json", run.path / "flops.json")
        if payload["best_epoch"] is not None:
            # The unchanged trainer copies the prior best into checkpoint.pth.
            link_checkpoint(run.path / "checkpoint.pth", run.path / "checkpoint_best.pth")
        del payload
    if resume is None:
        profile_forward(model, config, run)
        assert parameter_hash(model) == before, "FLOPs profiling changed parameters"
    run.epoch_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        existing_train(model, config, run, resume)
    except PhaseComplete:
        checkpoint = run.path / "checkpoint_last.pth"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        loaded = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        assert loaded.keys() == payload["model_state_dict"].keys()
        assert all(torch.equal(loaded[name], tensor) for name, tensor in payload["model_state_dict"].items())
        changed = parameter_hash(model) != before
        assert changed and payload["optimizer"]["state"]
        if MANY_WAY_PREFLIGHT:
            many_way_preflight(model, config, run)
        write_json(run.path / "dry_run_checks.json", {
            "status": "passed", "forward_backward_optimizer_checkpoint": True,
            "checkpoint_load_state_matches_live_model": True, "parameters_updated": changed,
            "completed_epochs": payload["completed_epochs"], "step": payload["step"],
            "full_training_completed": False,
            "loss": run.metrics["train_loss"], "resume": resume is not None,
            "strict_all_parameter_gradient_passed": False,
            "official_inactive_parameters_exception": "accepted; existing active-gradient checks retained"})
        run.update_metrics(status="dry_run_completed", full_training_completed=False,
                           purpose="resume_check" if resume else "one_epoch_dry_run")


def many_way_preflight(model, config, run):
    counters, was_training = dropblock_counters(model), model.training
    model.eval()
    records = []
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            for shot in (1, 5):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                images = torch.zeros((50 * (shot + 16), 3, 84, 84), device=config["runtime"]["device"])
                output = model(images, way=50, shot=shot)
                assert output.scores.shape == (800, 50) and torch.isfinite(output.scores).all()
                records.append({"way": 50, "shot": shot, "query": 16,
                    "batch_images": len(images), "forward_finite": True,
                    "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_max_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "accuracy": None, "scope": "synthetic inference capacity check; not formal evaluation"})
                del output, images
        write_json(run.path / "many_way_preflight.json", records)
    finally:
        for name, module in model.named_modules():
            if name in counters:
                module.num_batches_tracked = counters[name]
        model.train(was_training)


def entry_record_json(path, value):
    path = Path(path)
    if path.name == "data_manifest.json":
        # Preserve the richer Phase A manifest already copied by RecordedRun.
        existing = json.loads(path.read_text())
        assert existing["hash"]["tree_sha256"] == value["tree_sha256"]
        write_json(path.parent / "physical_copy_provenance.json", value)
    else:
        write_json(path, value)


def main():
    global STOP_AFTER, MANY_WAY_PREFLIGHT
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="experiments/sceg_original/configs/cub.yaml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--many-way-preflight", action="store_true")
    args = parser.parse_args()
    STOP_AFTER = args.stop_after_epoch
    MANY_WAY_PREFLIGHT = args.many_way_preflight
    assert json.loads((PHASE_A / "phase_a_checks.json").read_text())["status"] == "passed"
    verify_protected()
    entry.ExperimentRun = RecordedRun
    entry.train = recorded_train
    entry.write_json = entry_record_json
    sys.argv = ["tools.train", "--config", args.config, "--run-id", args.run_id, "--accept-official-inactive"]
    if args.resume:
        sys.argv += ["--resume", str(args.resume)]
    entry.main()


if __name__ == "__main__":
    main()
