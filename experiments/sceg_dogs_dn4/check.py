"""One real Dogs train forward/backward and two validation episodes; no checkpoints."""
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys
import time

from PIL import Image
import torch
from datasets import make_dataset, make_loader
from datasets.transforms import image_transform
from evaluators import evaluate_episodes
from experiments.sceg_dogs_baseline.prepare import check_prepared
from losses import IntraInterLoss
from models import SCEG
from utils.config import load_config, ModelConfig
from utils.diagnostics import assert_finite, gradient_summary
from utils.experiment import write_json
from utils.reproducibility import seed_everything
from .prepare import BASE


def main():
    started = time.perf_counter()
    config = load_config(BASE/"configs/dogs.yaml")
    record = check_prepared(config)
    seed_everything(config["seed"], deterministic=config["runtime"]["deterministic"],
                    allow_tf32=config["runtime"]["allow_tf32"],
                    warn_only=config["runtime"]["deterministic_warn_only"])
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    equal_inputs = []
    for part in ("val", "test"):
        raw_config = deepcopy(config)
        raw_config["data"]["pre_resized"] = False
        raw = make_dataset(raw_config, part)
        pre = make_dataset(config, part)
        assert raw.class_to_idx == pre.class_to_idx
        for path, label in raw.samples[:3]:
            pre_path = Path(config["data"]["root"])/(part+"_pre")/Path(path).parent.name/(Path(path).stem+".png")
            with Image.open(path) as image:
                online = raw.transform(image.convert("RGB"))
            with Image.open(pre_path) as image:
                offline = pre.transform(image.convert("RGB"))
            assert torch.equal(online, offline)
            equal_inputs.append(str(Path(path).relative_to(config["data"]["root"])))
    model = SCEG(ModelConfig.from_dict(config["model"])).to(config["runtime"]["device"])
    loader = make_loader(config, "train")
    assert len(loader.dataset.classes) == 70
    assert loader.dataset.class_to_idx == record["classes"]["train"]["class_to_idx"]
    images, labels = next(iter(loader))
    way, shot, query = (config["model"][key] for key in ("train_way", "train_shot", "train_query"))
    support = labels[:way*shot].reshape(way, shot)
    queries = labels[way*shot:].reshape(way, query)
    assert (support == support[:, :1]).all() and (queries == support[:, :1]).all()
    assert support[:, 0].unique().numel() == way and labels.min() >= 0 and labels.max() < 70
    images, labels = images.to(config["runtime"]["device"]), labels.to(config["runtime"]["device"])
    model.train()
    output = model(images)
    assert output.global_cosine.shape == (200, 70) and output.scores.shape == (150, 10)
    assert_finite(output, "real_dogs.forward")
    local = torch.arange(way, device=images.device).repeat_interleave(query)
    criterion = IntraInterLoss(way, **config["loss"]).to(images.device)
    global_criterion = IntraInterLoss(70, **config["loss"]).to(images.device)
    loss = sum(criterion(cosine, local) for cosine in output[:3])+global_criterion(output.global_cosine, labels)
    assert_finite(loss, "real_dogs.loss")
    loss.backward()
    gradients = gradient_summary(model)
    assert set(gradients["missing_parameters"]) == model.expected_inactive_parameters()
    validation = evaluate_episodes(model, config, "val", 5, 5, 2)
    result = {"status": "passed", "data": "real DN4 Stanford Dogs", "training_episodes": 1,
              "validation_episodes": validation["episodes"], "formal_training_state_reused": False,
              "global_head_shape": list(model.class_representative.weight.shape),
              "global_output_shape": list(output.global_cosine.shape),
              "episode_output_shape": list(output.scores.shape), "support_first": True,
              "global_labels_match_ImageFolder": True, "loss": loss.item(),
              "gradient_summary": gradients, "original_pre_input_tensor_equal": equal_inputs,
              "elapsed_seconds": time.perf_counter()-started,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "command": shlex.join(sys.orig_argv),
              "historical_checks": "../sceg_dogs_baseline/REPORT.md (synthetic smoke/resume), ../../docs/cub_reproduction_report.md"}
    write_json(BASE/"technical_checks.json", result)
    print(json.dumps({key: result[key] for key in ("status", "loss", "global_output_shape", "episode_output_shape", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
