"""Synthetic unit test of checkpoint selection and scheduler ordering, not a dataset result."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import logging
import torch
from models import SCEGOutput
import trainers.trainer as trainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(3)*0.1)

    def forward(self, images):
        cosine = self.weight.tanh()
        return SCEGOutput(cosine, cosine, cosine, cosine.repeat(2, 1))

    def expected_inactive_parameters(self):
        return set()


class TinyRun:
    def __init__(self, path):
        self.path = path
        self.metrics = {}
        self.logger = logging.getLogger("training-contract-test")

    def checkpoint(self, payload):
        torch.save(payload, self.path/"checkpoint.pth")

    def update_metrics(self, **values):
        self.metrics.update(values)


def test_best_checkpoint_is_restored_and_lr_steps_after_validation(config, tmp_path, monkeypatch):
    config = deepcopy(config)
    config["model"].update(train_way=3, train_shot=1, train_query=1, base_classes=3)
    config["train"].update(epochs=2, scheduler_step=1, validation_every=1)
    config["eval"].update(way=3, validation_shot=1, shots=[1], test_episodes=2)
    class Loader(list):
        dataset = SimpleNamespace(classes=[0, 1, 2])
    loader = Loader([(torch.zeros(6, 1), torch.arange(3).repeat(2))])
    monkeypatch.setattr(trainer, "make_loader", lambda *args, **kwargs: loader)
    snapshots = []
    def evaluate(model, cfg, split, way, shot, episodes):
        model.eval()
        if split == "val":
            snapshots.append(model.weight.detach().clone())
            accuracy = 100.0 if len(snapshots) == 1 else 0.0
        else:
            torch.testing.assert_close(model.weight, snapshots[0])
            accuracy = 100.0
        return {"seed": 42, "dataset": "synthetic_unit_test", "way": way, "shot": shot,
                "accuracy": accuracy, "confidence_interval": 0.0}
    monkeypatch.setattr(trainer, "evaluate_episodes", evaluate)
    model, run = TinyModel(), TinyRun(tmp_path)
    trainer.train(model, config, run)
    best = torch.load(tmp_path/"checkpoint.pth", weights_only=False)
    latest = torch.load(tmp_path/"latest.pth", weights_only=False)
    assert best["completed_epochs"] == best["best_epoch"] == 1
    assert latest["completed_epochs"] == 2
    assert best["optimizer"]["param_groups"][0]["lr"] == config["train"]["lr"]*config["train"]["scheduler_gamma"]
    assert latest["scheduler"]["last_epoch"] == 2
    assert len(best["optimizer"]["param_groups"][0]["params"]) == 1  # loss lambdas excluded
    assert run.metrics["full_training_completed"] is True
    torch.testing.assert_close(model.weight, best["model_state_dict"]["weight"])
