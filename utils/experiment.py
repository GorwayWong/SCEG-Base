"""Exclusive run directories, source fingerprints and complete experiment records."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import torch
import yaml


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def environment_info():
    versions = {}
    for package in ("torch", "torchvision", "numpy", "scipy", "Pillow", "PyYAML", "pytest", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "packages": versions,
            "torch_build": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": [{"name": torch.cuda.get_device_name(index),
                      "memory_bytes": torch.cuda.get_device_properties(index).total_memory}
                     for index in range(torch.cuda.device_count())]}


def source_manifest(root):
    files = {}
    for directory in ("models", "losses", "datasets", "utils", "tools", "trainers", "evaluators", "configs", "tests"):
        for path in sorted((root/directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml"}:
                files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


class ExperimentRun:
    def __init__(self, config, run_id=None):
        self.root = Path(__file__).resolve().parents[1]
        experiment = config["experiment"]
        if not experiment.isidentifier():
            raise ValueError("Experiment name must be an isolated directory identifier")
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("Invalid run id")
        self.path = self.root/"experiments"/experiment/"runs"/run_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        def git(*args):
            return subprocess.run(["rtk", "proxy", "git", *args], cwd=self.root,
                                  capture_output=True, text=True)
        commit_result = git("rev-parse", "HEAD")
        (self.path/"git_commit.txt").write_text(
            commit_result.stdout.strip()+"\n" if commit_result.returncode == 0 else "UNCOMMITTED\n")
        (self.path/"git_diff.patch").write_text(git("diff", "HEAD").stdout)
        (self.path/"git_status.txt").write_text(git("status", "--porcelain").stdout)
        (self.path/"command.txt").write_text(shlex.join([sys.executable, *sys.argv])+"\n")
        write_json(self.path/"environment.json", environment_info())
        write_json(self.path/"source_manifest.json", source_manifest(self.root))
        self.logger = logging.getLogger(f"sceg.{experiment}.{run_id}")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        for handler in (logging.FileHandler(self.path/"train.log"), logging.StreamHandler()):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.metrics = {"seed": config["seed"], "dataset": config["model"]["dataset"],
                        "way": config["model"]["train_way"], "shot": config["model"]["train_shot"],
                        "accuracy": None, "confidence_interval": None, "status": "running"}
        self.update_metrics()
        self.logger.info("run=%s", self.path)

    def update_metrics(self, **values):
        self.metrics.update(values)
        write_json(self.path/"metrics.json", self.metrics)

    def checkpoint(self, payload):
        partial = self.path/"checkpoint.pth.part"
        torch.save(payload, partial)
        partial.replace(self.path/"checkpoint.pth")
