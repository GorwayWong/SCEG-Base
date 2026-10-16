"""Author Cars orchestration. Training and episode mathematics stay in public modules."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from datasets import make_loader
from datasets.folder import make_dataset
from evaluators import evaluate_episodes
from losses import IntraInterLoss
from models import SCEG
from trainers.trainer import train, restore_rng, dropblock_counters
from utils.config import load_config, ModelConfig
from utils.diagnostics import assert_finite, gradient_summary
from utils.experiment import ExperimentRun, write_json
from utils.reproducibility import seed_everything, rng_state

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs/cars.yaml"
SETTINGS = ((5, 1), (5, 5), (49, 1), (49, 5))
PROTOCOL = "sceg_author_cars_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def configuration():
    config = load_config(CONFIG)
    prep = read_json(REPO / config['data_protocol']['prepared'])
    for key in ('split_manifest_sha256', 'sample_manifest_sha256'):
        config['data_protocol'][key] = prep[key]
    return config


def source_identity():
    paths = [p for directory in ("models", "losses", "datasets", "utils", "trainers", "evaluators")
             for p in (REPO / directory).rglob("*.py")]
    paths += [p for p in HERE.iterdir() if p.suffix in {".py", ".md"}]
    paths += list((HERE / 'configs').glob('*.yaml'))
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(paths)}


def configure(config):
    r = config["runtime"]
    if r["amp"] or r["allow_tf32"] or config["seed"] != 42:
        raise ValueError("This run is frozen to seed42 FP32, AMP/TF32 disabled")
    torch.set_num_threads(r["cpu_threads"])
    seed_everything(config["seed"], deterministic=r["deterministic"],
                    allow_tf32=r["allow_tf32"], warn_only=r["deterministic_warn_only"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")


def light_data_check(config):
    """Read preparation metadata and enumerate loader paths; never decode/hash images."""
    prepared_path = REPO / config["data_protocol"]["prepared"]
    prep = read_json(prepared_path)
    split_path = prepared_path.parent / prep["split_manifest"]
    samples_path = prepared_path.parent / prep["sample_manifest"]
    split = read_json(split_path)
    source_path = Path(split["source_record"])
    source = read_json(source_path)
    author = source_path.parent / source["files"]["cars"]["path"]
    assert prep["status"] == "data_prepared" and prep["data_ready"]
    assert prep["protocol_id"] == split["protocol_id"] == PROTOCOL
    assert source["source_kind"] == "author_via_email_user_supplied"
    assert source["author_supplied_complete_training_recipe"] is False
    assert config["data"]["root"] == prep["root"]
    assert config["model"]["base_classes"] == 130
    assert config["model"]["dataset"] == "stanford_car"
    assert config["data"]["pre_resized"] is True
    refs = {}
    for key, path in (("author_split_sha256", author), ("split_manifest_sha256", split_path),
                      ("sample_manifest_sha256", samples_path)):
        actual = digest(path)
        assert actual == prep[key] == config["data_protocol"][key], (key, actual)
        refs[key] = {"path": str(path), "sha256": actual}
    assert refs["author_split_sha256"]["sha256"] == source["files"]["cars"]["sha256"]
    datasets = {}
    for name, count, images in (("train", 130, 10747), ("val", 17, 1410), ("test", 49, 4028)):
        ds = make_dataset(config, name)
        assert ds.class_to_idx == split["class_to_idx"][name]
        assert len(ds.classes) == count and len(ds) == images
        transforms = [type(t).__name__ for t in ds.transform.transforms]
        if name != "train":
            assert transforms == ["ToTensor", "Normalize"]
        datasets[name] = {"root": str(ds.root), "classes": count, "images": len(ds),
                          "transforms": transforms}
    for name, path in (("prepared", prepared_path), ("source", source_path)):
        refs[name] = {"path": str(path), "sha256": digest(path)}
    provenance = prepared_path.parent / 'provenance.json'
    assert digest(provenance) == prep['provenance_sha256']
    refs['provenance'] = {'path': str(provenance), 'sha256': digest(provenance)}
    return {"protocol_id": PROTOCOL, "references": refs, "datasets": datasets,
            "scope": "metadata hashes + ImageFolder paths/mapping; prior full image validation reused"}


def model_check(config, data_check):
    parity = read_json(HERE / 'checks/official_parity.json')
    assert parity['status'] == 'passed' and parity['official_commit'] == config['official_commit']
    assert parity['loss_config'] == config['loss'] and parity['base_classes'] == 130
    for relative, sha in parity['replica_sources'].items():
        assert digest(REPO / relative) == sha
    configure(config)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = SCEG(ModelConfig.from_dict(config["model"])).to(config["runtime"]["device"])
    model.train()
    loader = make_loader(config, "train")
    iterator = iter(loader)
    images, labels = next(iterator)
    del iterator, loader
    images, labels = images.cuda(), labels.cuda()
    assert 0 <= labels.min() and labels.max() < 130
    assert list(images.shape) == [200, 3, 84, 84]
    support = labels[:50].reshape(10, 5)
    query = labels[50:].reshape(10, 15)
    assert (support == support[:, :1]).all() and (query == support[:, :1]).all()
    assert support[:, 0].unique().numel() == 10
    output = model(images)
    assert [list(x.shape) for x in output] == [[150, 10]]*3 + [[200, 130]]
    assert_finite(output, "real_episode")
    torch.testing.assert_close(output.scores, output[0]+output[1]+output[2], rtol=0, atol=0)
    criterion = IntraInterLoss(10, **config["loss"]).cuda()
    global_criterion = IntraInterLoss(130, **config["loss"]).cuda()
    targets = torch.arange(10, device="cuda").repeat_interleave(15)
    terms = [criterion(x, targets) for x in output[:3]] + [global_criterion(output[3], labels)]
    assert_finite(terms, "four_losses")
    optimizer = torch.optim.SGD(model.parameters(), **{k: config["train"][k]
                               for k in ("lr", "momentum", "nesterov", "weight_decay")})
    optimizer.zero_grad()
    sum(terms).backward()
    gradients = gradient_summary(model)
    assert set(gradients["missing_parameters"]) == model.expected_inactive_parameters()
    before = model.class_representative.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.class_representative.weight)
    assert criterion.lamda.item() == global_criterion.lamda.item() == 36.0
    losses = [float(x.item()) for x in terms]
    del output, terms, optimizer, criterion, global_criterion, images, labels, before
    validation = evaluate_episodes(model, config, "val", 5, 5, 3)
    assert validation["episodes"] == 3
    assert all(len(set(indices)) == len(indices) for indices in validation["episode_indices"])
    result = {"status": "passed", "created_at": utc(), "config_sha256": config_hash(config),
              "source_identity": source_identity(), "data_check": data_check,
              "official_parity_sha256": digest(HERE / 'checks/official_parity.json'),
              "global_shape": [200, 130], "query_shapes": [[150, 10]]*3,
              "losses": losses, "gradients": gradients, "validation_episodes": 3,
              "validation_accuracy": validation["accuracy"],
              "seconds": time.perf_counter()-started,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
              "weights_saved": False, "formal_epochs": 0}
    (HERE / "checks").mkdir(exist_ok=True)
    write_json(HERE / "checks" / "model_check.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"data_check", "source_identity"}}, indent=2))


def json_rng():
    state = rng_state()
    n = state["numpy"]
    return {"python": state["python"], "numpy": [n[0], n[1].tolist(), *n[2:]],
            "torch": state["torch"].tolist(), "cuda": [x.tolist() for x in state["cuda"]]}


def restore_json_rng(state):
    py = state["python"]
    n = state["numpy"]
    restore_rng({"python": (py[0], tuple(py[1]), py[2]),
                 "numpy": (n[0], np.array(n[1], dtype=np.uint32), *n[2:]),
                 "torch": torch.tensor(state["torch"], dtype=torch.uint8),
                 "cuda": [torch.tensor(x, dtype=torch.uint8) for x in state["cuda"]]})


def restore_counters(model, counters):
    for name, module in model.named_modules():
        if name in counters:
            module.num_batches_tracked = counters[name]


class AuthorRun(ExperimentRun):
    """Keep the public run recorder; add append-only epoch/event records."""
    def __init__(self, config, *args, **kwargs):
        self.author_config = config
        super().__init__(config, *args, **kwargs)

    def update_metrics(self, **values):
        if values.get('status') == 'running':
            values['status'] = 'evaluating' if getattr(self, 'metrics', {}).get('training_completed') else 'training'
        if values.get("completed_epochs", 0) > 0:
            with (self.path / "epochs.jsonl").open("a") as f:
                f.write(json.dumps({"recorded_at": utc(), **values}, allow_nan=False)+"\n")
        super().update_metrics(**values)
        if values.get("completed_epochs", 0) > 0 and values["completed_epochs"] % 20 == 0:
            write_report(self, self.author_config)


def event(run, kind, **values):
    with (run.path / "events.jsonl").open("a") as f:
        f.write(json.dumps({"at": utc(), "event": kind, **values})+"\n")


def checkpoint_identity(payload, config):
    if payload["config"] != config:
        raise ValueError("Checkpoint is not from this run and frozen configuration")
    if payload["config"]["data_protocol"]["protocol_id"] != PROTOCOL:
        raise ValueError("Wrong checkpoint protocol")
    required = {"optimizer", "scheduler", "rng", "dropblock_counters", "completed_epochs"}
    if not required.issubset(payload):
        raise ValueError("Incomplete epoch checkpoint")


def validate_result(result, config, way, shot, checkpoint_sha, sample_paths=None):
    expected = {"protocol_id": PROTOCOL, "run_id": config["run_id"], "seed": 42,
                "dataset": "stanford_car", "split": "test", "way": way, "shot": shot,
                "query": 16, "episodes": 2000, "checkpoint_sha256": checkpoint_sha,
                "split_sha256": config["data_protocol"]["split_manifest_sha256"],
                "config_sha256": config_hash(config)}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Invalid evaluation identity: {key}")
    values = np.asarray(result["episode_accuracies"], dtype=float)
    if values.shape != (2000,) or not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
        raise ValueError("Invalid episode accuracies")
    paths = result["sample_index_paths"]
    if sample_paths is not None and paths != sample_paths:
        raise ValueError("Dataset path/index mapping changed")
    episodes = result["episode_indices"]
    if len(episodes) != 2000 or len(result["episode_classes"]) != 2000:
        raise ValueError("Incomplete episode index/class records")
    for indices, recorded_classes in zip(episodes, result["episode_classes"]):
        if len(indices) != way*(shot+16) or len(set(indices)) != len(indices):
            raise ValueError("Repeated or missing support/query indices")
        if any(type(i) is not int or i < 0 or i >= len(paths) for i in indices):
            raise ValueError("Invalid sample index")
        classes = [Path(paths[i]).parts[0] for i in indices]
        support = np.array(classes[:way*shot]).reshape(way, shot)
        query = np.array(classes[way*shot:]).reshape(way, 16)
        if not ((support == support[:, :1]).all() and (query == support[:, :1]).all()
                and len(set(support[:, 0])) == way and list(support[:, 0]) == recorded_classes):
            raise ValueError("Episode class/order mismatch")
    mean = float(values.mean())
    ci = float(1.96*np.sqrt(values.var(ddof=0)/len(values)))
    if not np.isclose(mean, result["accuracy"], atol=1e-10, rtol=0) or not np.isclose(
            ci, result["confidence_interval"], atol=1e-10, rtol=0):
        raise ValueError("Reported statistics disagree with raw episodes")
    if "rng_after" not in result or "dropblock_after" not in result:
        raise ValueError("Missing evaluation continuation state")
    return {"way": way, "shot": shot, "query": 16, "episodes": len(values),
            "accuracy": mean, "confidence_interval": ci, "seconds": result["seconds"],
            "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
            "checkpoint_sha256": checkpoint_sha, "statistics_recomputed": True}


def evaluate_suite(model, config, run):
    device = config["runtime"]["device"]
    latest = torch.load(run.path / "latest.pth", map_location="cpu", weights_only=False)
    checkpoint_identity(latest, config)
    if latest["completed_epochs"] != 800:
        raise ValueError("Formal evaluation requires all 800 completed epochs")
    best_path = run.path / "checkpoint.pth"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    checkpoint_identity(best, config)
    if best["completed_epochs"] != latest["best_epoch"] or best["best_epoch"] != latest["best_epoch"]:
        raise ValueError("Best checkpoint does not match latest validation selection")
    sha = digest(best_path)
    frozen_sha = run.metrics.get("best_sha256")
    if frozen_sha is not None and frozen_sha != sha:
        raise ValueError("Selected best changed after evaluation started")
    model.load_state_dict(best["model_state_dict"], strict=True)
    # Match public trainer: best weights, training-end RNG and non-state_dict counters.
    restore_counters(model, latest["dropblock_counters"])
    restore_rng(latest["rng"])
    best_epoch = latest["best_epoch"]
    del best, latest
    model.to(device).eval()
    run.update_metrics(training_completed=True, full_training_completed=True,
                       full_evaluation_completed=False, best_sha256=sha, best_epoch=best_epoch,
                       status='evaluating')
    ds = make_dataset(config, "test")
    paths = [str(Path(p).relative_to(ds.root)) for p, _ in ds.samples]
    boundaries_path = run.path / "evaluation_boundaries.json"
    boundaries = read_json(boundaries_path) if boundaries_path.exists() else {}
    summaries = []
    for way, shot in SETTINGS:
        key = f"{way}way_{shot}shot"
        path = run.path / f"test_{key}.json"
        result = None
        if path.exists():
            try:
                result = read_json(path)
            except json.JSONDecodeError:
                event(run, "incomplete_result_retry", setting=key)
        if result is not None:
            summary = validate_result(result, config, way, shot, sha, paths)
            restore_json_rng(result["rng_after"])
            restore_counters(model, result["dropblock_after"])
            event(run, "reuse_complete_evaluation", setting=key)
        else:
            if key not in boundaries:
                boundaries[key] = {"rng": json_rng(), "counters": dropblock_counters(model),
                                   "checkpoint_sha256": sha}
                write_json(boundaries_path, boundaries)
            boundary = boundaries[key]
            if boundary["checkpoint_sha256"] != sha:
                raise ValueError("Boundary checkpoint mismatch")
            restore_json_rng(boundary["rng"])
            restore_counters(model, boundary["counters"])
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            event(run, "evaluation_started", setting=key)
            result = evaluate_episodes(model, config, "test", way, shot, 2000)
            result.update(protocol_id=PROTOCOL, run_id=config["run_id"], checkpoint_sha256=sha,
                          split_sha256=config["data_protocol"]["split_manifest_sha256"],
                          config_sha256=config_hash(config), seconds=time.perf_counter()-started,
                          peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                          rng_after=json_rng(), dropblock_after=dropblock_counters(model))
            result["episode_classes"] = [[Path(paths[i]).parts[0] for i in indices[:way*shot:shot]]
                                         for indices in result["episode_indices"]]
            summary = validate_result(result, config, way, shot, sha, paths)
            write_json(path, result)
            event(run, "evaluation_completed", setting=key, **summary)
        summaries.append(summary)
        run.update_metrics(evaluation=summaries, status="running")
        run.logger.info("test=%s accuracy=%.4f ci=%.4f", key, summary["accuracy"], summary["confidence_interval"])
    if digest(best_path) != sha:
        raise ValueError("Checkpoint changed during evaluation")
    run.update_metrics(status="completed", training_completed=True, full_evaluation_completed=True,
                       evaluation=summaries, completed_at=utc(), metric_scope="held-out test episodes")


def snapshot(run):
    write_json(run.path / "execution_sources.json", source_identity())
    # Public tracked edits are already in ExperimentRun.git_diff.patch. Save new task
    # files once; git diff alone cannot represent these untracked files.
    for relative in source_identity():
        path = REPO / relative
        if not path.is_relative_to(HERE):
            continue
        tracked = subprocess.run(["rtk", "proxy", "git", "ls-files", "--error-unmatch", str(path)],
                                 cwd=REPO, capture_output=True).returncode == 0
        if not tracked:
            target = run.path / "source_snapshot" / path.relative_to(HERE)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def verify_source(run):
    if read_json(run.path / "execution_sources.json") != source_identity():
        raise ValueError("Execution source changed; restore the recorded source before continuing this run")


def commands(run_id):
    prefix = f"rtk proxy env CUDA_VISIBLE_DEVICES=0 {shlex.quote(sys.executable)} -m experiments.sceg_cars_author.run --run-id {shlex.quote(run_id)}"
    return {"train": prefix+" --mode pipeline", "resume": prefix+" --mode pipeline --resume",
            "evaluate_only": prefix+" --mode evaluate", "status": prefix+" --mode status"}


def write_report(run, config):
    metrics = run.metrics
    rows = {}
    for way, shot in SETTINGS:
        path = run.path / f"test_{way}way_{shot}shot.json"
        rows[f"{way}way_{shot}shot"] = None
        if path.exists() and metrics.get("best_sha256"):
            try:
                rows[f"{way}way_{shot}shot"] = validate_result(read_json(path), config, way, shot, metrics["best_sha256"])
            except (ValueError, KeyError, TypeError):
                pass
    if metrics.get("status") == "completed" and any(x is None for x in rows.values()):
        raise ValueError("Cannot report completed without four verified results")
    if metrics.get('status') == 'completed' and (metrics.get('completed_epochs') != 800 or
            not metrics.get('training_completed') or not metrics.get('full_evaluation_completed')):
        raise ValueError('Cannot report completed before all training and evaluation')
    epochs = [json.loads(x) for x in (run.path / "epochs.jsonl").read_text().splitlines()] if (run.path / "epochs.jsonl").exists() else []
    report = {"status": metrics["status"], "protocol_id": PROTOCOL,
              "training_completed": metrics.get("training_completed", False),
              "full_evaluation_completed": metrics.get("full_evaluation_completed", False),
              "completed_epochs": metrics.get("completed_epochs", 0), "best_epoch": metrics.get("best_epoch"),
              "best_sha256": metrics.get("best_sha256"), "results": rows,
              "train_validation_checkpoint_seconds": sum(x.get("epoch_seconds", 0) for x in epochs),
              "train_seconds": sum(x.get('train_seconds', 0) for x in epochs),
              "validation_seconds": sum(x.get('validation_seconds', 0) for x in epochs),
              "peak_train_cuda_memory_bytes": max([x.get("peak_cuda_memory_bytes", 0) for x in epochs] or [0]),
              "commands": commands(config["run_id"]), "updated_at": utc(),
              "initialization": "random; no pretrained weights", "training_seeds": [42],
              "class_counts": [130, 17, 49], "image_counts": [10747, 1410, 4028],
              "ci_scope": "episode sampling conditional on one trained model; not multi-training-seed stability",
              "failure_and_resume_record": "events.jsonl", "source": "execution_sources.json + git_commit.txt + git_diff.patch + source_snapshot/",
              "recipe": "source_snapshot/recipe.md or repository recipe.md at the recorded version",
              "latest_sha256": digest(run.path / 'latest.pth') if metrics['status'] == 'completed' else None,
              "disk_bytes": sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file()),
              "uncertainty": "Cars complete command and actual original temperature unconfirmed; lambda36 is generic parser default; mirror pixels not verified against original archive"}
    write_json(run.path / "report.json", report)
    lines = ["# Cars original SCEG — author split", "", f"Status: **{report['status']}**; epochs: {report['completed_epochs']}/800.",
             f"Training completed: {report['training_completed']}; full evaluation completed: {report['full_evaluation_completed']}.",
             "", "Protocol: sceg_author_cars_v1. Classes 130/17/49; original images 10747/1410/4028 (16185 total).",
             "Original annotation class IDs (1..196) map through author membership to zero-padded folders; ImageFolder train labels are 0..129; episode labels are 0..way-1. Original train/test filename prefixes prevent collisions.",
             "Random initialization, seed42, FP32, no AMP/TF32; 4 workers, 2 CPU threads. Prepared train/val/test records remain read-only.",
             f"Best epoch: {report['best_epoch']}; best SHA256: {report['best_sha256']}.", "",
             "| Setting | Accuracy (%) | 95% CI half-width (pp) | Episodes | Seconds | Peak allocated bytes |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, value in rows.items():
        lines.append(f"| {key} | null | null | null | null | null |" if value is None else
                     f"| {key} | {value['accuracy']:.6f} | {value['confidence_interval']:.6f} | {value['episodes']} | {value['seconds']:.2f} | {value['peak_cuda_memory_bytes']} |")
    lines += ["", "CI = 1.96 × sqrt(var(episode_accuracy, ddof=0) / 2000), in percentage points. All available summaries are recomputed from raw arrays.",
              "One trained model: this CI describes episode sampling only, not variation across training seeds.",
              "", f"Recorded training + validation + checkpoint time: {report['train_validation_checkpoint_seconds']:.2f}s; peak train allocated GPU bytes: {report['peak_train_cuda_memory_bytes']}.",
              f"Training compute/data time: {report['train_seconds']:.2f}s; validation time: {report['validation_seconds']:.2f}s.",
              "Interrupted partial epochs are not included in epoch totals; invocation starts, failures and resumes are in events.jsonl.",
              "", "Recipe provenance: recipe.md. Cars intra=0.65 explicitly uses the paper value through the official parameter interface; fixed lambda36 is the generic parser default, not a confirmed Cars-specific training temperature. The Cars g_st gate still executes and updates BN but its output is discarded. No model formulas changed.",
              f"Run disk bytes (including two checkpoints when present): {report['disk_bytes']}. Raw-image mirror equivalence to the original Stanford archive is unverified; full images were not cropped/re-encoded for training.",
              "Source: git_commit.txt + git_diff.patch for tracked files, source_snapshot/ for new entry files, execution_sources.json for loaded source identity; environment.json and command.txt record the actual invocation.",
              "Input fingerprints: data_check.json, referencing the original prep manifests without duplicating their sample list.",
              "Raw tests bind the same best SHA and split SHA, with class names, sample indices, relative paths and episode accuracy; evaluation_boundaries.json preserves retries' RNG.",
              "This task changes only the author entry and optional public trainer recording/finalization. CUB, ProKeR and archives are not written. No ProKeR gain is computed.",
              "", "Commands (cwd: "+str(REPO)+"):", "", "```bash", *report["commands"].values(), "```", ""]
    (run.path / "report.md").write_text("\n".join(lines))
    if epochs:
        # Keep directly plottable loss/validation curves without another per-epoch file.
        (run.path / "curves.csv").write_text("epoch,train_loss,train_accuracy,validation_accuracy,validation_ci,seconds,learning_rate\n" +
            "\n".join(",".join(str(e.get(k, "")) if e.get(k) is not None else "" for k in
                ("completed_epochs", "train_loss", "train_accuracy", "validation_accuracy", "validation_ci", "epoch_seconds", "learning_rate")) for e in epochs)+"\n")


@contextmanager
def exclusive(run_id):
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Invalid run id")
    with (HERE / f".{run_id}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This run already has an active process") from exc
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "pipeline", "evaluate", "status"), required=True)
    parser.add_argument("--run-id", default="cars800_seed42_author_v1")
    parser.add_argument("--resume", action="store_true", help="Only this run's latest complete epoch")
    args = parser.parse_args()
    if args.resume and args.mode != "pipeline":
        parser.error("--resume is only valid with --mode pipeline")
    config = configuration()
    if Path.cwd().resolve() != REPO:
        raise ValueError(f'Run from the isolated Cars working directory: {REPO}')
    if args.mode == "status":
        path = HERE / "runs" / args.run_id
        print(json.dumps({"run": str(path), **read_json(path / "metrics.json")}, indent=2))
        return
    with exclusive(args.run_id):
        if args.mode == "check":
            model_check(config, light_data_check(config))
            return
        configure(config)
        reuse = args.resume or args.mode == "evaluate"
        config["run_id"] = args.run_id
        if reuse:
            run_path = HERE / "runs" / args.run_id
            saved_config = load_config(run_path / "config.yaml")
            if config != saved_config or read_json(run_path / "execution_sources.json") != source_identity():
                raise ValueError("Run config/source mismatch; restore recorded version before continuation")
            if read_json(run_path / 'metrics.json').get('status') == 'completed':
                print(json.dumps({'status': 'completed', 'action': 'preserved; no recomputation', 'run': str(run_path)}))
                return
            payload = torch.load(run_path / 'latest.pth', map_location='cpu', weights_only=False)
            checkpoint_identity(payload, config)
            if args.mode == 'evaluate' and payload['completed_epochs'] != 800:
                raise ValueError('Evaluate-only requires epoch800; no run files changed')
            del payload
            data_check = read_json(run_path / "data_check.json")
            # Metadata hashes are cheap; do not repeat full data-preparation validation.
            for ref in data_check["references"].values():
                if digest(ref["path"]) != ref["sha256"]:
                    raise ValueError("Prepared metadata changed")
        else:
            check = read_json(HERE / "checks" / "model_check.json")
            original_config = {k: v for k, v in config.items() if k != "run_id"}
            if check["status"] != "passed" or check["config_sha256"] != config_hash(original_config) or check["source_identity"] != source_identity():
                raise ValueError("Model check is missing or does not match the frozen execution source")
            data_check = check["data_check"]
        run = AuthorRun(config, args.run_id, reuse_existing=reuse)
        if not reuse:
            snapshot(run)
            write_json(run.path / "data_check.json", data_check)
            write_json(run.path / "model_check.json", check)
            write_json(run.path / 'official_parity.json', read_json(HERE / 'checks/official_parity.json'))
            write_json(run.path / 'test_results.json', read_json(HERE / 'checks/test_results.json'))
            run.update_metrics(training_completed=False, full_evaluation_completed=False, completed_epochs=0,
                               started_at=utc(), initialization="random", protocol_id=PROTOCOL)
        event(run, "invocation", mode=args.mode, resume=args.resume, pid=os.getpid(),
              command=shlex.join(getattr(sys, "orig_argv", [sys.executable, *sys.argv])))
        write_report(run, config)
        run.publish_result()
        try:
            verify_source(run)
            model = SCEG(ModelConfig.from_dict(config["model"])).to(config["runtime"]["device"])
            resume_path = None
            completed = 0
            if reuse:
                resume_path = run.path / "latest.pth"
                payload = torch.load(resume_path, map_location="cpu", weights_only=False)
                checkpoint_identity(payload, config)
                completed = payload["completed_epochs"]
                del payload
            if args.mode == "pipeline" and completed < 800:
                torch.cuda.reset_peak_memory_stats()
                event(run, "training_started" if not reuse else "training_resumed", completed_epochs=completed)
                train(model, config, run, resume=resume_path)
                event(run, "training_completed")
            elif completed != 800:
                raise ValueError("Evaluate-only cannot enter training; latest must contain epoch800")
            verify_source(run)
            evaluate_suite(model, config, run)
            write_report(run, config)
            run.publish_result()
        except BaseException as exc:
            event(run, "failure", error=repr(exc), traceback=traceback.format_exc())
            run.update_metrics(status="failed", error=repr(exc))
            write_report(run, config)
            run.publish_result()
            raise


if __name__ == "__main__":
    main()
