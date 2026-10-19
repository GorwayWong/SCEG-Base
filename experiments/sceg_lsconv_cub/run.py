"""Single CUB migration; reuse training, sampling, loss and episode evaluator."""
import argparse
import hashlib
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
from utils.reproducibility import seed_everything, seed_worker
from .model import build_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RUN = "20261019_cub800_seed42"


def configuration():
    spec = yaml.safe_load((HERE/"config.yaml").read_text())
    baseline = Path(spec["baseline_root"])
    config = load_config(baseline/spec["baseline_config"])
    historical = load_config(baseline/spec["baseline_resolved"])
    for key in ("model", "train", "loss", "data", "runtime", "eval", "seed", "official_commit"):
        if config[key] != historical[key]:
            raise ValueError(f"Effective CUB recipe differs from completed run: {key}")
    config["experiment"] = "sceg_lsconv_cub"
    config["data"]["root"] = str((baseline/config["data"]["root"]).resolve())
    config["replacement"] = spec["replacement"]
    config["execution"] = spec["execution"]
    return config, spec


def setup(config):
    seed_everything(config["seed"], deterministic=config["runtime"]["deterministic"],
        allow_tf32=config["runtime"]["allow_tf32"],
        warn_only=config["runtime"]["deterministic_warn_only"])
    torch.set_num_threads(config["runtime"]["cpu_threads"])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    paths = []
    for folder in ("models", "losses", "datasets", "utils", "trainers", "evaluators"):
        paths += list((ROOT/folder).rglob("*.py"))
    paths += list(HERE.glob("*.py")) + [HERE/"config.yaml"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


class _LogHandler(logging.Handler):
    def __init__(self, run):
        super().__init__()
        self.run = run

    def emit(self, record):
        self.run.event("progress", message=record.getMessage())


class Run:
    def __init__(self, run_id=DEFAULT_RUN, create=False):
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("Invalid run id")
        self.path = HERE/"runs"/run_id
        self.config, self.spec = configuration()
        if create and not self.path.exists():
            self.path.mkdir(parents=True)
            baseline = Path(self.spec["baseline_root"])
            def git(*args):
                return subprocess.check_output(["rtk", "proxy", "git", *args], cwd=ROOT, text=True)
            self.info = {"status": "implemented", "config": self.config,
                "specification": self.spec, "command": shlex.join(sys.orig_argv),
                "cwd": str(ROOT), "commit": git("rev-parse", "HEAD").strip(),
                "git_diff": git("diff", "HEAD"), "git_status": git("status", "--porcelain"),
                "new_source": {str(p.relative_to(ROOT)): p.read_text()
                               for p in sorted(HERE.iterdir()) if p.suffix in {".py", ".yaml", ".md"}},
                "source_sha256": source_identity(), "environment": environment_info(),
                "baseline": json.loads((baseline/self.spec["baseline_results"]).read_text()),
                "data_manifest_reference": str(baseline/self.spec["data_manifest"]),
                "stages": {"implementation": "completed", "precheck": "pending", "training": "pending",
                           "formal_tests": "pending", "statistics": "pending"},
                "changes": {"algorithm": "Only SC.self_s -> LSNet LSConv(320); official internal residual/BN unchanged",
                    "numerical": "FP32, no AMP/TF32; no numerical recipe changes",
                    "execution": "Detached epoch accumulation; no activation/gradient scans; last every20 and final; separate tests; eval chunks128"}}
            self.save()
        else:
            self.info = json.loads((self.path/"run_info.json").read_text())
            if self.info["config"] != self.config or self.info["source_sha256"] != source_identity():
                raise ValueError("Run configuration/source conflict; do not silently resume changed code")
        self.metrics = {}
        self.logger = logging.getLogger("lsconv."+run_id)
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(_LogHandler(self))

    def save(self):
        write_json(self.path/"run_info.json", self.info)

    def event(self, event, **values):
        value = {"time": time.time(), "event": event, **values}
        with (self.path/"metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False)+"\n")
        print(json.dumps(value, allow_nan=False), flush=True)

    def update_metrics(self, **values):
        self.metrics.update(values)
        self.event("epoch" if "epoch_episodes" in values else "status", **values)

    def checkpoint(self, payload):
        partial = self.path/"checkpoint.pth.part"
        torch.save(payload, partial)
        partial.replace(self.path/"checkpoint.pth")


def validation(model, config, split, way, shot, episodes):
    result = evaluate_episodes(model, config, split, way, shot, episodes,
        encode_chunk_size=config["execution"]["encode_chunk_size"], record_indices=False)
    result.pop("episode_accuracies")
    return result


def replay_loader(config, dataset, historical, way, shot):
    paths = [str(Path(p).relative_to(dataset.root)) for p, _ in dataset.samples]
    if historical["sample_index_paths"] != paths:
        raise ValueError("Historical test sample identity/order differs; pairing unavailable")
    indices = historical["episode_indices"]
    assert len(indices) == 2000 and len(historical["episode_accuracies"]) == 2000
    labels = np.asarray(dataset.targets)
    for ids in indices:
        assert len(ids) == way*(shot+16) and len(set(ids)) == len(ids)
        support = labels[ids[:way*shot]].reshape(way, shot)
        query = labels[ids[way*shot:]].reshape(way, 16)
        assert (support == support[:, :1]).all() and (query == support[:, :1]).all()
        assert len(set(support[:, 0])) == way
    return DataLoader(dataset, batch_sampler=indices, num_workers=config["data"]["workers"],
        pin_memory=False, worker_init_fn=seed_worker)


def evaluate(run):
    if run.info["stages"]["training"] != "completed":
        raise RuntimeError("Formal testing requires completed 800-epoch training")
    config = run.config
    best = run.path/"checkpoint.pth"
    best_hash = run.info.get("best_sha256") or sha256(best)
    run.info["best_sha256"] = best_hash
    run.info["status"] = "evaluating"
    run.info["stages"]["formal_tests"] = "running"
    run.save()
    model = build_model(config).to(config["runtime"]["device"])
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["config"] == config
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for name, module in model.named_modules():
        if name in payload["dropblock_counters"]:
            module.num_batches_tracked = payload["dropblock_counters"][name]
    del payload
    dataset = make_dataset(config, "test")
    for baseline_result in run.info["baseline"]["evaluations"]:
        way, shot = baseline_result["way"], baseline_result["shot"]
        out = run.path/f"test_{way}way_{shot}shot.json"
        if out.exists():
            done = json.loads(out.read_text())
            assert done["checkpoint_sha256"] == best_hash and done["episodes"] == 2000
            continue
        reference = Path(run.spec["baseline_root"])/baseline_result["raw_result"]
        assert sha256(reference) == baseline_result["raw_result_sha256"]
        historical = json.loads(reference.read_text())
        loader = replay_loader(config, dataset, historical, way, shot)
        torch.cuda.reset_peak_memory_stats()
        result = evaluate_episodes(model, config, "test", way, shot, 2000,
            encode_chunk_size=config["execution"]["encode_chunk_size"], logger=run.logger,
            loader=loader, record_indices=False)
        result.update(checkpoint_sha256=best_hash, indices_reference=str(reference),
            indices_reference_sha256=baseline_result["raw_result_sha256"],
            pairing="exact historical episode_indices and sample_index_paths verified",
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            encode_chunk_size=config["execution"]["encode_chunk_size"])
        write_json(out, result)
        run.event("formal_test_completed", **{k: v for k, v in result.items() if k != "episode_accuracies"})
        del historical, loader
    run.info["stages"]["formal_tests"] = "completed"
    run.save()
    statistics(run)


def statistics(run):
    rows = []
    for baseline_result in run.info["baseline"]["evaluations"]:
        way, shot = baseline_result["way"], baseline_result["shot"]
        result = json.loads((run.path/f"test_{way}way_{shot}shot.json").read_text())
        historical = json.loads(Path(result["indices_reference"]).read_text())
        values = np.asarray(result["episode_accuracies"], dtype=np.float64)
        baseline = np.asarray(historical["episode_accuracies"], dtype=np.float64)
        assert len(values) == len(baseline) == 2000 and np.isfinite(values).all()
        mean, half = float(values.mean()), float(1.96*values.std(ddof=0)/np.sqrt(len(values)))
        assert abs(mean-result["accuracy"]) < 1e-10
        assert abs(half-result["confidence_interval"]) < 1e-10
        difference = values-baseline
        gain, paired_half = float(difference.mean()), float(1.96*difference.std(ddof=0)/np.sqrt(len(values)))
        rows.append({"way": way, "shot": shot, "accuracy_percent": mean, "ci95_half": half,
            "baseline_percent": float(baseline.mean()), "gain_pp": gain,
            "paired_ci95_half_pp": paired_half, "paired_ci95_bounds_pp": [gain-paired_half, gain+paired_half],
            "evaluation_seconds": result["elapsed_seconds"], "peak_cuda_bytes": result["peak_cuda_bytes"]})
    write_json(run.path/"results.json", {"status": "completed", "evaluations": rows,
        "best_sha256": run.info["best_sha256"], "statistics_cpu_recomputed": True,
        "scope": "One CUB seed42 SC.self_s replacement; no cross-training-seed or internal-mechanism attribution"})
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
    setup(run.config)
    run.event("invocation", command=shlex.join(sys.orig_argv))
    if args.mode == "train":
        assert run.info["stages"]["precheck"] == "completed"
        if run.info["stages"]["training"] == "completed":
            raise RuntimeError("Training already completed; do not retrain")
        last = run.path/"latest.pth"
        if last.exists() != args.resume:
            raise RuntimeError("Existing last requires --resume; --resume requires existing last")
        run.info["status"] = "training"
        run.info["stages"]["training"] = "running"
        run.save()
        start = time.perf_counter()
        try:
            model = build_model(run.config).to(run.config["runtime"]["device"])
            torch.cuda.reset_peak_memory_stats()
            train(model, run.config, run, last if args.resume else None,
                  lightweight=True, evaluator=validation, automatic_test=False)
        except BaseException as error:
            run.info["status"] = "stopped"
            run.info["error"] = f"{type(error).__name__}: {error}"
            run.save()
            raise
        run.info["training_seconds"] = run.info.get("training_seconds", 0)+time.perf_counter()-start
        run.info["training_peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
        run.info["best_epoch"] = run.metrics["best_epoch"]
        run.info["stages"]["training"] = "completed"
        run.info["status"] = "training_completed"
        run.save()
    elif args.mode == "evaluate":
        evaluate(run)
    else:
        statistics(run)


if __name__ == "__main__":
    main()
