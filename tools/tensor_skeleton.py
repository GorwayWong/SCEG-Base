"""Phase1 random episode; no training/data/optimizer dependency."""
import argparse
import json
import torch
from models import SCEG
from utils.config import load_config, ModelConfig
from utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sceg_original.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"], **{key: config["runtime"][key]
                                    for key in ("deterministic", "allow_tf32")})
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    model = SCEG(ModelConfig.from_dict(config["model"])).to(args.device).eval()
    verification = config["verification"]
    way, shot, query = (verification[key] for key in ("way", "shot", "query"))
    images = torch.randn(way*(shot+query), config["model"]["input_channels"],
                         config["model"]["image_size"], config["model"]["image_size"],
                         device=args.device)
    with torch.no_grad():
        # Eval microbatching is an engineering memory adapter; training never splits BN batches.
        chunks = [model.encode(chunk) for chunk in images.split(verification["microbatch"])]
        embeddings = tuple(torch.cat([chunk[stage] for chunk in chunks]) for stage in range(3))
        output = model.classify(embeddings, way, shot)
    print(json.dumps({"input": list(images.shape),
        "embeddings": [list(embedding.shape) for embedding in embeddings],
        "output": [list(cosine.shape) for cosine in output],
        "finite": all(torch.isfinite(cosine).all().item() for cosine in output)}, indent=2))


if __name__ == "__main__":
    main()
