"""Official trainers/eval.py protocol, with explicit raw episode accuracies and CI."""
from pathlib import Path
import numpy as np
import torch
from datasets import make_loader


@torch.no_grad()
def evaluate_episodes(model, config, split, way, shot, episodes):
    if config["eval"]["query"] != 16:
        raise ValueError("Official evaluation sampler requires actual query=16 (D05)")
    if episodes <= 0:
        raise ValueError("Evaluation requires at least one episode")
    model.eval()
    loader = make_loader(config, split, way=way, shot=shot, episodes=episodes)
    device = next(model.parameters()).device
    targets = torch.arange(way, device=device).repeat_interleave(config["eval"]["query"])
    accuracies = []
    for images, global_labels in loader:
        support_labels = global_labels[:way*shot].reshape(way, shot)
        query_labels = global_labels[way*shot:].reshape(way, config["eval"]["query"])
        assert (support_labels == support_labels[:, :1]).all()
        assert (query_labels == support_labels[:, :1]).all(), "Support/query class order changed"
        assert support_labels[:, 0].unique().numel() == way
        output = model(images.to(device), way=way, shot=shot)
        assert output.scores.shape[0] == targets.numel()
        predictions = output.scores.argmax(dim=1)
        accuracies.append(100*(predictions == targets).sum().item()/targets.numel())
    values = np.array(accuracies)
    return {"seed": config["seed"], "dataset": config["model"]["dataset"], "split": split,
            "way": way, "shot": shot, "query": config["eval"]["query"],
            "episodes": len(accuracies), "accuracy": float(values.mean()),
            "confidence_interval": float(config["eval"]["confidence_multiplier"]*
                                           np.sqrt(values.var()/len(values))),
            "confidence_interval_type": "95% normal half-width, population variance ddof=0",
            "episode_accuracies": accuracies,
            "episode_indices": loader.batch_sampler.episode_indices,
            "sample_index_paths": [str(Path(path).relative_to(loader.dataset.root))
                                   for path, _ in loader.dataset.samples]}
