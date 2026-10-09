"""Inspect a local SCEG checkpoint, summary, parameters, FLOPs and activations."""
import argparse
import json
from pathlib import Path
import torch
from models import SCEG
from utils.config import ModelConfig, load_config
from utils.diagnostics import parameter_summary, estimate_flops, activation_statistics
from utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", default="configs/sceg_original.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if payload is not None and "config" in payload:
        config = payload["config"]
    seed_everything(config["seed"], warn_only=config["runtime"]["deterministic_warn_only"])
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    settings = ModelConfig.from_dict(config["model"])
    model = SCEG(settings).to(args.device).eval()
    if payload is not None:
        model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    batch = settings.train_way*(settings.train_shot+1)
    images = torch.randn(batch, settings.input_channels, settings.image_size,
                         settings.image_size, device=args.device)
    with torch.no_grad(), activation_statistics(model) as activations:
        model(images)
    print(json.dumps({"parameters": parameter_summary(model),
                      "flops": estimate_flops(model, images, settings.train_way, settings.train_shot),
                      "activations": activations}, indent=2))


if __name__ == "__main__":
    main()
