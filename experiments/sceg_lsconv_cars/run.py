"""Cars-only fixed LSConv experiment; local Cars trainer and historical episode replay."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from datasets import make_dataset
from evaluators.fewshot_eval import evaluate_episodes
from trainers.trainer import train
from utils.config import load_config
from utils.experiment import environment_info, write_json
from utils.reproducibility import seed_worker
from experiments.sceg_cars_author.run import configure as setup, config_hash, validate_result
from .model import build_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RUN = "20261019_cars800_seed42"
SETTINGS = ((5, 1), (5, 5), (49, 1), (49, 5))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration(run_id=DEFAULT_RUN):
    spec = yaml.safe_load((HERE/"config.yaml").read_text())
    baseline = Path(spec["baseline_root"])/spec["baseline_run"]
    config = load_config(baseline/"config.yaml")
    template = load_config(ROOT/"experiments/sceg_cars_author/configs/cars.yaml")
    for key in ("model", "train", "loss", "data", "runtime", "eval", "seed", "official_commit"):
        if config[key] != template[key]:
            raise ValueError(f"Cars template differs from historical effective recipe: {key}")
    config.update(experiment="sceg_lsconv_cars", run_id=run_id,
                  replacement=spec["replacement"], execution=spec["execution"])
    assert config["train"]["epochs"] == 800 and config["train"]["validation_every"] == 20
    assert config["loss"] == {"lamda": 36.0, "inter": .05, "intra": .65}
    assert config["model"]["dataset"] == "stanford_car" and config["model"]["base_classes"] == 130
    assert not config["runtime"]["post_training_evaluation"]
    return config, spec


def source_identity():
    paths = [p for directory in ("models", "losses", "datasets", "utils", "trainers", "evaluators")
             for p in (ROOT/directory).rglob("*.py")]
    paths += [ROOT/"experiments/sceg_cars_author/run.py", ROOT/"experiments/sceg_cars_author/configs/cars.yaml"]
    paths += [p for p in HERE.iterdir() if p.suffix in {".py", ".yaml", ".sh"}]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


class LogHandler(logging.Handler):
    def __init__(self, run):
        super().__init__()
        self.run = run

    def emit(self, record):
        self.run.event("progress", message=record.getMessage())


class Run:
    def __init__(self, run_id=DEFAULT_RUN, create=False, repair_precheck=False):
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("Invalid run id")
        self.path = HERE/"runs"/run_id
        self.config, self.spec = configuration(run_id)
        if create:
            repair = None
            if repair_precheck:
                previous = json.loads((self.path/"run_info.json").read_text())
                assert previous["stages"]["precheck"] == "failed" and previous["stages"]["training"] == "pending"
                assert not list(self.path.glob("*.pth")) and not list(self.path.glob("test_*.json"))
                assert not any(v.get("status") == "passed" for v in previous.get("precheck", {}).values() if isinstance(v, dict)), "Reuse passed checks; do not repeat them"
                repair = {"error": previous["error"], "source_sha256": previous["source_sha256"],
                          "precheck": previous["precheck"], "reason": "Pre-training implementation repair; no GPU update or passed check repeated"}
            else:
                self.path.mkdir(parents=True, exist_ok=False)
            baseline = Path(self.spec["baseline_root"])/self.spec["baseline_run"]
            def git(*args):
                return subprocess.check_output(["rtk", "proxy", "git", *args], cwd=ROOT, text=True)
            env = environment_info()
            env["packages"]["triton"] = importlib.metadata.version("triton")
            self.info = {"status": "implemented", "config": self.config, "specification": self.spec,
                "command": shlex.join(sys.orig_argv), "cwd": str(ROOT),
                "commit": git("rev-parse", "HEAD").strip(), "git_diff": git("diff", "HEAD"),
                "git_status": git("status", "--porcelain"),
                "new_source": {str(p.relative_to(ROOT)): p.read_text() for p in sorted(HERE.iterdir())
                               if p.suffix in {".py", ".yaml", ".md", ".sh"}},
                "source_sha256": source_identity(), "environment": env,
                "baseline_config": load_config(baseline/"config.yaml"),
                "baseline_config_file_sha256": sha256(baseline/"config.yaml"),
                "baseline_report_reference": str(baseline/"report.json"),
                "baseline_report_sha256": sha256(baseline/"report.json"),
                "data_manifest_reference": str(Path(self.spec["baseline_root"])/self.config["data_protocol"]["prepared"]),
                "stages": {"implementation": "completed", "precheck": "pending", "training": "pending",
                           "formal_tests": "pending", "statistics": "pending"},
                "changes": {"algorithm": "Only SC.self_s -> fixed LSConv(320), preserving module internal residual",
                    "execution": "Detached epoch accumulation, sparse full-state last/best, separate replay evaluation, chunk128 encoding",
                    "numerical_or_protocol": "none; inherited FP32 Cars recipe, original sampler and validation schedule"}}
            if repair is not None:
                self.info["precheck_repair"] = repair
            self.save()
        else:
            self.info = json.loads((self.path/"run_info.json").read_text())
            if self.info["config"] != self.config or self.info["source_sha256"] != source_identity():
                raise ValueError("Frozen configuration/source mismatch; refusing changed execution code")
        self.metrics = {}
        self.logger = logging.getLogger("lsconv."+run_id)
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.addHandler(LogHandler(self))

    def save(self):
        write_json(self.path/"run_info.json", self.info)

    def event(self, event, **values):
        value = {"time": time.time(), "event": event, **values}
        line = json.dumps(value, allow_nan=False)
        with (self.path/"metrics.jsonl").open("a") as stream:
            stream.write(line+"\n")
        print(line, flush=True)

    def update_metrics(self, **values):
        self.metrics.update(values)
        self.event("epoch" if "epoch_episodes" in values else "status", **values)

    def checkpoint(self, payload):
        assert payload["checkpoint_role"] == "best_validation"
        partial = self.path/"checkpoint.pth.part"
        torch.save(payload, partial)
        partial.replace(self.path/"checkpoint.pth")


def historical_result(run, dataset, way, shot):
    reference = run.info["historical_results"][f"{way}way_{shot}shot"]
    path = Path(reference["path"])
    assert sha256(path) == reference["sha256"], f"Historical task hash mismatch: {path}"
    result = json.loads(path.read_text())
    paths = [str(Path(p).relative_to(dataset.root)) for p, _ in dataset.samples]
    validate_result(result, run.info["baseline_config"], way, shot,
                    run.spec["baseline_best_sha256"], paths)
    # Confirm ImageFolder numeric labels agree with recorded relative-path class names.
    assert all(dataset.classes[label] == Path(relative).parts[0]
               for relative, label in zip(paths, dataset.targets))
    return result, reference


def validate_new_result(run, result, way, shot):
    ref = run.info["historical_results"][f"{way}way_{shot}shot"]
    expected = {"protocol_id": "sceg_author_cars_v1", "run_id": run.config["run_id"],
        "config_sha256": config_hash(run.config), "seed": 42, "dataset": "stanford_car", "split": "test",
        "way": way, "shot": shot, "query": 16, "episodes": 2000,
        "checkpoint_sha256": run.info["best_sha256"], "indices_reference": ref["path"],
        "indices_reference_sha256": ref["sha256"],
        "source_sha256": run.info["source_sha256"],
        "split_sha256": run.config["data_protocol"]["split_manifest_sha256"]}
    for key, value in expected.items():
        assert result.get(key) == value, f"Formal result identity mismatch: {key}"
    values = np.asarray(result["episode_accuracies"], dtype=np.float64)
    assert values.shape == (2000,) and np.isfinite(values).all() and ((values >= 0)&(values <= 100)).all()
    assert abs(float(values.mean())-result["accuracy"]) < 1e-10
    assert abs(float(1.96*values.std(ddof=0)/np.sqrt(2000))-result["confidence_interval"]) < 1e-10
    return values


def checkpoint(run, name, role):
    payload = torch.load(run.path/name, map_location="cpu", weights_only=False)
    assert payload["config"] == run.config and payload["source_sha256"] == run.info["source_sha256"]
    assert payload["checkpoint_role"] == role
    assert all(k in payload for k in ("optimizer", "scheduler", "rng", "dropblock_counters", "step"))
    if role == "best_validation":
        assert payload["completed_epochs"] == payload["best_epoch"] == run.info["best_epoch"]
    return payload


def completed_training(run):
    assert run.info["stages"]["training"] == "completed"
    events = [json.loads(line) for line in (run.path/"metrics.jsonl").read_text().splitlines()]
    # A native resume may redo at most 19 unsaved epochs. Latest event per epoch is authoritative.
    epochs = {v["completed_epochs"]: v for v in events if v["event"] == "epoch"}
    assert set(epochs) == set(range(1, 801)), "800 complete epochs required"
    validations = [v for v in epochs.values() if v["validation_accuracy"] is not None]
    assert len(validations) == 40 and all(v["completed_epochs"] % 20 == 0 and v["validation_episodes"] == 1000 for v in validations)
    best = max(validations, key=lambda v: (v["validation_accuracy"], -v["completed_epochs"]))
    assert best["completed_epochs"] == run.info["best_epoch"]
    return epochs


def evaluate(run):
    completed_training(run)
    last = checkpoint(run, "latest.pth", "last")
    assert last["completed_epochs"] == 800
    del last
    best_hash = sha256(run.path/"checkpoint.pth")
    assert best_hash == run.info["best_sha256"]
    payload = checkpoint(run, "checkpoint.pth", "best_validation")
    model = build_model(run.config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for name, module in model.named_modules():
        if name in payload["dropblock_counters"]:
            module.num_batches_tracked = payload["dropblock_counters"][name]
    del payload
    model.to(run.config["runtime"]["device"]).eval()
    run.info["status"] = "evaluating"
    run.info["stages"]["formal_tests"] = "running"
    run.save()
    dataset = make_dataset(run.config, "test")
    for way, shot in SETTINGS:
        out = run.path/f"test_{way}way_{shot}shot.json"
        historical, reference = historical_result(run, dataset, way, shot)
        if out.exists():
            validate_new_result(run, json.loads(out.read_text()), way, shot)
            del historical
            continue
        loader = DataLoader(dataset, batch_sampler=historical["episode_indices"],
            num_workers=run.config["data"]["workers"], pin_memory=False, worker_init_fn=seed_worker)
        torch.cuda.reset_peak_memory_stats()
        result = evaluate_episodes(model, run.config, "test", way, shot, 2000,
            encode_chunk_size=128, logger=run.logger, loader=loader, record_indices=False)
        result.update(protocol_id="sceg_author_cars_v1", run_id=run.config["run_id"],
            config_sha256=config_hash(run.config), source_sha256=run.info["source_sha256"],
            split_sha256=run.config["data_protocol"]["split_manifest_sha256"],
            checkpoint_sha256=best_hash, indices_reference=reference["path"],
            indices_reference_sha256=reference["sha256"],
            pairing="exact historical episode_indices, paths, labels and support/query ordering validated",
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), encode_chunk_size=128)
        validate_new_result(run, result, way, shot)
        write_json(out, result)
        run.event("formal_test_completed", way=way, shot=shot, accuracy=result["accuracy"],
                  seconds=result["elapsed_seconds"], peak_cuda_bytes=result["peak_cuda_bytes"])
        del historical, loader, result
    run.info["stages"]["formal_tests"] = "completed"
    run.save()


def statistics(run):
    epochs = completed_training(run)
    assert run.info["stages"]["formal_tests"] == "completed"
    assert sha256(run.path/"checkpoint.pth") == run.info["best_sha256"]
    dataset = make_dataset(run.config, "test")
    rows = []
    for way, shot in SETTINGS:
        result = json.loads((run.path/f"test_{way}way_{shot}shot.json").read_text())
        values = validate_new_result(run, result, way, shot)
        historical, _ = historical_result(run, dataset, way, shot)
        baseline = np.asarray(historical["episode_accuracies"], dtype=np.float64)
        difference = values-baseline
        gain = float(difference.mean())
        paired_half = float(1.96*difference.std(ddof=0)/np.sqrt(2000))
        rows.append({"way": way, "shot": shot, "new_accuracy_percent": float(values.mean()),
            "new_ci95_half": float(1.96*values.std(ddof=0)/np.sqrt(2000)),
            "baseline_accuracy_percent": float(baseline.mean()),
            "baseline_ci95_half": float(1.96*baseline.std(ddof=0)/np.sqrt(2000)),
            "gain_pp": gain, "paired_ci95_bounds_pp": [gain-paired_half, gain+paired_half],
            "test_seconds": result["elapsed_seconds"], "peak_cuda_bytes": result["peak_cuda_bytes"]})
        del historical, result
    report = {"status": "completed", "evaluations": rows, "best_epoch": run.info["best_epoch"],
        "best_sha256": run.info["best_sha256"], "epochs": 800, "validations": 40,
        "timing_seconds": {key: sum(v[key] for v in epochs.values())
                           for key in ("train_seconds", "validation_seconds", "save_seconds")},
        "training_wall_seconds": run.info["training_wall_seconds"],
        "training_peak_cuda_bytes": run.info["training_peak_cuda_bytes"],
        "statistics": "CPU float64; population std ddof=0; 1.96/sqrt(2000); paired episode differences",
        "scope": "Single training seed42; no cross-seed stability or internal mechanism causal claim",
        "limitations": "Inherited Cars mirror provenance; lambda36 is completed-baseline value, not an author-confirmed full recipe. Timing is not a controlled LSConv speedup comparison.",
        "artifact_bytes_before_report": sum(p.stat().st_size for p in run.path.iterdir() if p.is_file())}
    write_json(run.path/"results.json", report)
    run.info["stages"]["statistics"] = "completed"
    run.info["status"] = "completed"
    run.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "evaluate", "statistics"), required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = Run(args.run_id)
    if args.mode != "statistics":
        setup(run.config)
    run.event("invocation", command=shlex.join(sys.orig_argv))
    try:
        if args.mode == "train":
            assert run.info["stages"]["precheck"] == "completed"
            assert run.info["stages"]["training"] != "completed", "Training already completed"
            last = run.path/"latest.pth"
            assert last.exists() == args.resume, "Use --resume only with existing last"
            if args.resume:
                saved = checkpoint(run, "latest.pth", "last")
                run.metrics["best_epoch"] = saved["best_epoch"]
                del saved
            run.info["status"] = "training"
            run.info["stages"]["training"] = "running"
            run.save()
            start = time.perf_counter()
            model = build_model(run.config).to(run.config["runtime"]["device"])
            torch.cuda.reset_peak_memory_stats()
            def validation(model, config, split, way, shot, episodes):
                result = evaluate_episodes(model, config, split, way, shot, episodes,
                    encode_chunk_size=128, logger=run.logger, record_indices=False)
                result.pop("episode_accuracies")
                return result
            train(model, run.config, run, last if args.resume else None,
                  lightweight=True, evaluator=validation)
            run.info["training_wall_seconds"] = run.info.get("training_wall_seconds", 0)+time.perf_counter()-start
            run.info["training_peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
            run.info["best_epoch"] = run.metrics["best_epoch"]
            run.info["best_sha256"] = sha256(run.path/"checkpoint.pth")
            run.info["stages"]["training"] = "completed"
            run.info["status"] = "training_completed"
            completed_training(run)
            run.save()
        elif args.mode == "evaluate":
            evaluate(run)
        else:
            statistics(run)
    except BaseException as error:
        run.info["status"] = "stopped"
        run.info["error"] = f"{type(error).__name__}: {error}"
        run.save()
        raise


if __name__ == "__main__":
    main()
