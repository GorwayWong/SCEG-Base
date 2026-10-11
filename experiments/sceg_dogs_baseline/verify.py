"""At most two TOTAL synthetic epochs; no Dogs research result or guessed class split.

All image fixtures and diagnostic weights are deleted after verification. Raw Dogs
readability is audited separately by prepare --audit-raw, without extracting images.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from datasets import make_dataset
from datasets.samplers import TrainingEpisodeSampler, EvaluationEpisodeSampler
from datasets.transforms import image_transform
from evaluators import evaluate_episodes
from models import SCEG
from trainers.trainer import train, restore_rng
from tools.prepare_data import sha256_file
from utils.config import load_config, ModelConfig
from utils.experiment import write_json
from utils.reproducibility import seed_everything, rng_state
from .prepare import BASE, COUNTS
from .run import DogsRun, checkpoint_records


class EpochOneSaved(Exception):
    pass


class StopAfterOne(DogsRun):
    def update_metrics(self, **values):
        super().update_metrics(**values)
        if values.get("completed_epochs") == 1:
            raise EpochOneSaved("Intentional diagnostic interruption after complete epoch1")


def protected_inventory():
    """Metadata inventory, avoiding another hash of historical images/weights."""
    root = Path("/root/projects/aris/sceg_fsfg")
    digest = hashlib.sha256()
    count = 0
    for relative in ("experiments/sceg_original", "data/frn", "configs", "models", "docs/paper"):
        for path in sorted((root/relative).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                stat = path.stat()
                digest.update(json.dumps([str(path.relative_to(root)), stat.st_size,
                                          stat.st_mtime_ns, stat.st_ino]).encode())
                count += 1
    return {"root": str(root), "files": count, "metadata_sha256": digest.hexdigest(),
            "method": "relative path + size + mtime_ns + inode; no image rehash"}


def fixture(root):
    generator = np.random.RandomState(7)
    for split, count in COUNTS.items():
        for category in range(count):
            folder = root/split/f"synthetic_{split}_{category:03d}"
            folder.mkdir(parents=True)
            for index in range(20 if split == "train" else 21):
                Image.fromarray(generator.randint(0, 256, (84, 84, 3), dtype=np.uint8)).save(
                    folder/f"{index:03d}.png")
    crop = transforms.Compose([transforms.Resize(92), transforms.CenterCrop(84)])
    for split in ("val", "test"):
        for path in sorted((root/split).rglob("*.png")):
            target = root/(split+"_pre")/path.relative_to(root/split)
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image:
                crop(image.convert("RGB")).save(target)


def check_order(dataset, indices, way, shot, query):
    labels = np.array(dataset.targets)[indices]
    support = labels[:way*shot].reshape(way, shot)
    queries = labels[way*shot:].reshape(way, query)
    assert len(set(indices)) == len(indices)
    assert len(set(support[:, 0])) == way
    assert (support == support[:, :1]).all() and (queries == support[:, :1]).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    before = protected_inventory()
    started = time.perf_counter()
    config = load_config(BASE/"configs/dogs.yaml")
    config["purpose"] = "synthetic_technical_smoke_only_not_Dogs_baseline"
    config["train"]["epochs"] = 2
    config["train"]["validation_every"] = 1
    config["eval"]["validation_episodes"] = 2
    config["eval"]["test_episodes"] = 2
    config["checkpoint_source"] = None
    config["checkpoint_source_sha256"] = None
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    seed_everything(42, allow_tf32=False, warn_only=True)
    device = config["runtime"]["device"]
    runs = []
    with tempfile.TemporaryDirectory(prefix="dogs_synthetic_smoke_") as directory:
        config["data"]["root"] = directory
        fixture(Path(directory))
        datasets = {split: make_dataset(config, split) for split in COUNTS}
        for split, dataset in datasets.items():
            assert len(dataset.classes) == COUNTS[split]
            assert dataset.class_to_idx == {name: i for i, name in enumerate(sorted(dataset.classes))}
        class_sets = [set(dataset.classes) for dataset in datasets.values()]
        assert sum(map(len, class_sets)) == len(set.union(*class_sets)) == 120
        episodes = list(TrainingEpisodeSampler(datasets["train"], 10, (5, 15)))
        assert len(episodes) == 7  # fixture exhaustion only; no formal epoch budget set
        assert len(set(sum(episodes, []))) == 1400
        for episode in episodes:
            check_order(datasets["train"], episode, 10, 5, 15)
        for way in (5, 30):
            for shot in (1, 5):
                sampler = EvaluationEpisodeSampler(datasets["test"], way, shot, 2, 16)
                for episode in sampler:
                    check_order(datasets["test"], episode, way, shot, 16)
        val_image = sorted((Path(directory)/"val").rglob("*.png"))[0]
        pre_image = Path(directory)/"val_pre"/val_image.relative_to(Path(directory)/"val")
        with Image.open(val_image) as raw, Image.open(pre_image) as pre:
            assert torch.equal(image_transform(config, False, False)(raw.convert("RGB")),
                               image_transform(config, False, True)(pre.convert("RGB")))
        # Reset randomness independently of fixture creation and sampler inspection.
        seed_everything(42, allow_tf32=False, warn_only=True)
        model = SCEG(ModelConfig.from_dict(config["model"])).to(device)
        assert model.class_representative.weight.shape == (70, 640)
        observed = {}
        def output_witness(module, inputs, output):
            observed.setdefault("global_shape", list(output.global_cosine.shape))
            observed.setdefault("episode_shape", list(output.scores.shape))
        hook = model.register_forward_hook(output_witness)
        first = StopAfterOne(copy.deepcopy(config), args.run_id+"_epoch1")
        runs.append(first)
        try:
            train(model, config, first)
        except EpochOneSaved:
            first.update_metrics(status="diagnostic_interrupted", full_training_completed=False,
                                 metric_scope="synthetic fixture only")
        hook.remove()
        checkpoint_records(first)
        payload = torch.load(first.path/"latest.pth", map_location="cpu", weights_only=False)
        assert payload["completed_epochs"] == 1 and payload["step"] == 7
        assert payload["scheduler"]["last_epoch"] == 1 and payload["optimizer"]["state"]
        assert payload["best_epoch"] == 1
        restore_rng(payload["rng"])
        witness = [float(np.random.rand()), float(torch.rand(1))]
        restore_rng(payload["rng"])
        assert witness == [float(np.random.rand()), float(torch.rand(1))]
        # Existing trainer restores model, optimizer, scheduler, counters and RNG.
        resumed_config = copy.deepcopy(config)
        resumed_config["checkpoint_source"] = str(first.path/"latest.pth")
        resumed_config["checkpoint_source_sha256"] = sha256_file(first.path/"latest.pth")
        resumed = DogsRun(resumed_config, args.run_id+"_resume2")
        runs.append(resumed)
        resumed_model = SCEG(ModelConfig.from_dict(config["model"])).to(device)
        train(resumed_model, resumed_config, resumed, first.path/"latest.pth")
        checkpoint_records(resumed)
        final_payload = torch.load(resumed.path/"latest.pth", map_location="cpu", weights_only=False)
        assert final_payload["completed_epochs"] == 2 and final_payload["step"] == 14
        assert final_payload["scheduler"]["last_epoch"] == 2
        assert all(final_payload["dropblock_counters"][key] > value
                   for key, value in payload["dropblock_counters"].items())
        checks = {"fixture_scope": config["purpose"], "formal_training_executed": False,
                  "technical_epochs_total": 2, "class_counts": COUNTS,
                  "support_first_and_sample_nonoverlap": True,
                  "global_head_shape": [70, 640], "first_train_output": observed,
                  "forward_backward_active_gradient_check": "passed by unchanged trainer",
                  "accepted_inactive_parameters": sorted(model.expected_inactive_parameters()),
                  "online_and_pre_resized_eval_tensors_equal": True,
                  "resume_completed_epochs": 2, "resume_steps": 14,
                  "rng_replay_checked": True, "scheduler_and_momentum_restored": True,
                  "dropblock_counters_continued": True, "formal_accuracy": None,
                  "evaluation": [], "protected_before": before}
        best = torch.load(resumed.path/"checkpoint.pth", map_location="cpu", weights_only=False)
        resumed_model.load_state_dict(best["model_state_dict"], strict=True)
        for way in (5, 30):
            for shot in (1, 5):
                if way == 5:
                    path = resumed.path/f"test_{way}way_{shot}shot.json"
                    result = json.loads(path.read_text())
                else:
                    result = evaluate_episodes(resumed_model, config, "test", way, shot, 2)
                    write_json(resumed.path/f"test_{way}way_{shot}shot.json", result)
                values = np.array(result["episode_accuracies"])
                assert result["episodes"] == len(result["episode_indices"]) == len(values) == 2
                assert result["accuracy"] == float(values.mean())
                assert result["confidence_interval"] == float(1.96*np.sqrt(values.var()/2))
                checks["evaluation"].append({"way": way, "shot": shot, "episodes": 2,
                                             "indices_and_population_CI_verified": True})
        checks["elapsed_seconds"] = time.perf_counter()-started
        checks["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        checks["protected_after"] = protected_inventory()
        assert checks["protected_after"] == before
        checks["historical_CUB_and_untracked_paper_unchanged"] = True
        write_json(resumed.path/"technical_checks.json", checks)
        # No synthetic weights are needed for scientific claims or formal resume.
        for run in runs:
            removed = []
            for path in run.path.glob("*.pth"):
                removed.append({"name": path.name, "bytes": path.stat().st_size})
                path.unlink()
            write_json(run.path/"diagnostic_cleanup.json", {"weights_removed": removed,
                                                         "reason": "regenerable synthetic fixtures"})
            run.update_metrics(status="technical_checks_passed", accuracy=None,
                               confidence_interval=None, full_training_completed=False,
                               full_evaluation_completed=False,
                               metric_scope="synthetic technical smoke only; not Dogs accuracy")
        print(json.dumps({"technical_checks": str(resumed.path/"technical_checks.json"),
                          "elapsed_seconds": checks["elapsed_seconds"],
                          "gpu_peak_allocated_bytes": checks["gpu_peak_allocated_bytes"]}))


if __name__ == "__main__":
    main()
