"""Official trainers/eval.py protocol, with explicit raw episode accuracies and CI."""
from pathlib import Path
import time
import numpy as np
import torch
from datasets import make_loader


@torch.no_grad()
def encode_in_chunks(model, images, chunk_size):
    """Reuse Cars-author helper: CPU chunks, ordered features, one logical counter tick."""
    assert not model.training
    device = next(model.parameters()).device
    counters = [(m, m.num_batches_tracked) for m in model.modules() if hasattr(m, "ramp_batches")]
    parts = [model.encode(chunk.to(device)) for chunk in images.split(chunk_size)]
    for module, before in counters:
        module.num_batches_tracked = before + 1
    return tuple(torch.cat(branch, dim=0) for branch in zip(*parts))


@torch.no_grad()
def evaluate_episodes(model, config, split, way, shot, episodes, *,
                      encode_chunk_size=None, logger=None, loader=None, record_indices=True):
    if config["eval"]["query"] != 16:
        raise ValueError("Official evaluation sampler requires actual query=16 (D05)")
    if episodes <= 0:
        raise ValueError("Evaluation requires at least one episode")
    model.eval()
    if loader is None:
        loader = make_loader(config, split, way=way, shot=shot, episodes=episodes)
    device = next(model.parameters()).device
    targets = torch.arange(way, device=device).repeat_interleave(config["eval"]["query"])
    accuracies = []
    started = time.perf_counter()
    for images, global_labels in loader:
        support_labels = global_labels[:way*shot].reshape(way, shot)
        query_labels = global_labels[way*shot:].reshape(way, config["eval"]["query"])
        assert (support_labels == support_labels[:, :1]).all()
        assert (query_labels == support_labels[:, :1]).all(), "Support/query class order changed"
        assert support_labels[:, 0].unique().numel() == way
        if encode_chunk_size is None:
            output = model(images.to(device), way=way, shot=shot)
        else:
            output = model.classify(encode_in_chunks(model, images, encode_chunk_size), way, shot)
        assert output.scores.shape[0] == targets.numel()
        predictions = output.scores.argmax(dim=1)
        accuracies.append(100*(predictions == targets).sum().item()/targets.numel())
        if logger is not None and len(accuracies) % 100 == 0:
            logger.info("eval %dway_%dshot episodes=%d/%d elapsed=%.2fs", way, shot,
                        len(accuracies), episodes, time.perf_counter()-started)
    values = np.array(accuracies)
    result = {"seed": config["seed"], "dataset": config["model"]["dataset"], "split": split,
            "way": way, "shot": shot, "query": config["eval"]["query"],
            "episodes": len(accuracies), "accuracy": float(values.mean()),
            "confidence_interval": float(config["eval"]["confidence_multiplier"]*
                                           np.sqrt(values.var()/len(values))),
            "confidence_interval_type": "95% normal half-width, population variance ddof=0",
            "episode_accuracies": accuracies, "elapsed_seconds": time.perf_counter()-started}
    if record_indices:
        result.update(episode_indices=loader.batch_sampler.episode_indices,
            sample_index_paths=[str(Path(path).relative_to(loader.dataset.root))
                               for path, _ in loader.dataset.samples])
    return result
