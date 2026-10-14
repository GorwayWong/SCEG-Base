"""Small contract regressions; no dataset rebuilding or training simulations."""
from copy import deepcopy
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.sceg_dogs_author import run as author
from models import SCEGOutput
from utils.config import load_config
from utils.reproducibility import seed_everything
import trainers.trainer as trainer


def configuration():
    cfg = load_config(author.CONFIG)
    cfg["run_id"] = "unit_author"
    return cfg


def result_fixture():
    cfg = configuration()
    paths = [f"breed{c}/{i}.png" for c in range(5) for i in range(17)]
    indices = [c*17 for c in range(5)] + [c*17+i for c in range(5) for i in range(1, 17)]
    result = {"protocol_id": author.PROTOCOL, "run_id": cfg["run_id"], "seed": 42,
              "dataset": "stanford_dog", "split": "test", "way": 5, "shot": 1,
              "query": 16, "episodes": 2000, "checkpoint_sha256": "best-hash",
              "split_sha256": cfg["data_protocol"]["split_manifest_sha256"],
              "config_sha256": author.config_hash(cfg), "episode_accuracies": [50.0]*2000,
              "sample_index_paths": paths, "episode_indices": [indices]*2000,
              "episode_classes": [[f"breed{c}" for c in range(5)]]*2000,
              "accuracy": 50.0, "confidence_interval": 0.0, "seconds": 1.0,
              "peak_cuda_memory_bytes": 0, "rng_after": author.json_rng(), "dropblock_after": {}}
    return cfg, result, paths


def test_result_identity_statistics_and_disjointness():
    cfg, result, paths = result_fixture()
    assert author.validate_result(result, cfg, 5, 1, "best-hash", paths)["statistics_recomputed"]
    for key, value in (("run_id", "old_run"), ("checkpoint_sha256", "old_best"),
                       ("split_sha256", "dn4"), ("query", 15), ("accuracy", 99.0),
                       ("episode_accuracies", [50.0]*1999)):
        broken = {**result, key: value}
        with pytest.raises(ValueError):
            author.validate_result(broken, cfg, 5, 1, "best-hash", paths)
    broken = deepcopy(result)
    broken["episode_indices"][0] = [broken["episode_indices"][0][0]]*85
    with pytest.raises(ValueError, match="support/query"):
        author.validate_result(broken, cfg, 5, 1, "best-hash", paths)


def test_rng_roundtrip_preserves_existing_sampling_stream():
    seed_everything(42, warn_only=True)
    state = json.loads(json.dumps(author.json_rng()))
    expected = (np.random.permutation(30), torch.rand(4))
    author.restore_json_rng(state)
    np.testing.assert_array_equal(np.random.permutation(30), expected[0])
    torch.testing.assert_close(torch.rand(4), expected[1], rtol=0, atol=0)


def test_checkpoint_rejects_other_run_and_partial_state():
    cfg = configuration()
    payload = {"config": cfg, "optimizer": {}, "scheduler": {}, "rng": {},
               "dropblock_counters": {}, "completed_epochs": 800}
    author.checkpoint_identity(payload, cfg)
    with pytest.raises(ValueError):
        author.checkpoint_identity(payload, {**cfg, "run_id": "another"})
    del payload["rng"]
    with pytest.raises(ValueError, match="Incomplete"):
        author.checkpoint_identity(payload, cfg)


def test_trainer_options_preserve_default_math_and_defer_completion(tmp_path, monkeypatch):
    cfg = configuration()
    cfg["model"].update(train_way=3, train_shot=1, train_query=1, base_classes=3)
    cfg["train"].update(epochs=1, validation_every=1)
    cfg["eval"].update(way=3, validation_shot=1, shots=[1])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3)*.1)
        def forward(self, images):
            scores = self.weight.tanh()
            return SCEGOutput(scores, scores, scores, scores.repeat(2, 1))
        def expected_inactive_parameters(self):
            return set()

    class Run:
        def __init__(self, path):
            self.path, self.metrics = path, {}
            path.mkdir()
            self.logger = logging.getLogger(str(path))
        def update_metrics(self, **values):
            self.metrics.update(values)
        def checkpoint(self, payload):
            torch.save(payload, self.path / "checkpoint.pth")

    class Loader(list):
        dataset = SimpleNamespace(classes=list(range(3)))

    monkeypatch.setattr(trainer, "make_loader", lambda *a, **k: Loader([(torch.zeros(6, 1), torch.arange(3).repeat(2))]))
    calls = []
    def evaluate(model, config, split, way, shot, episodes):
        calls.append(split)
        return {"accuracy": 10.0, "confidence_interval": 0.0, "way": way, "shot": shot}
    monkeypatch.setattr(trainer, "evaluate_episodes", evaluate)
    default_cfg = deepcopy(cfg)
    for key in ("record_activations", "record_epoch_timing", "post_training_evaluation"):
        default_cfg["runtime"].pop(key)
    default_model, default_run = Model(), Run(tmp_path / "default")
    trainer.train(default_model, default_cfg, default_run)
    assert calls == ["val", "test"] and default_run.metrics["status"] == "completed"
    assert (default_run.path / "activations_epoch_1.json").exists()
    calls.clear()
    author_model, author_run = Model(), Run(tmp_path / "author")
    trainer.train(author_model, cfg, author_run)
    assert calls == ["val"]
    assert author_run.metrics["status"] == "running"
    assert author_run.metrics["training_completed"] and not author_run.metrics["full_evaluation_completed"]
    assert not list(author_run.path.glob("activations_epoch_*.json"))
    assert author_run.metrics["epoch_seconds"] > 0
    torch.testing.assert_close(default_model.weight, author_model.weight, rtol=0, atol=0)
    assert set(p.name for p in author_run.path.glob("*.pth")) == {"latest.pth", "checkpoint.pth"}


def test_incomplete_report_cannot_claim_completed(tmp_path):
    cfg = configuration()
    fake_run = SimpleNamespace(path=tmp_path, metrics={"status": "completed", "best_sha256": "best-hash"})
    with pytest.raises(ValueError, match="four verified"):
        author.write_report(fake_run, cfg)


def test_run_recorder_isolated_to_author_family(tmp_path, monkeypatch):
    import utils.experiment as experiment
    monkeypatch.setattr(experiment, "__file__", str(tmp_path / "utils" / "experiment.py"))
    cfg = configuration()
    run = author.AuthorRun(cfg, cfg["run_id"])
    run.publish_result()
    assert run.path == tmp_path / "experiments" / "sceg_dogs_author" / "runs" / cfg["run_id"]
    assert (tmp_path / "experiments" / "sceg_dogs_author" / "result.json").exists()
    assert not (tmp_path / "experiments" / "sceg_original").exists()
    with pytest.raises(FileExistsError):
        author.AuthorRun(cfg, cfg["run_id"])
