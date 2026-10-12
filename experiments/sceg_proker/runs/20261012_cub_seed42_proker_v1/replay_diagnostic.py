"""Inspect the sole historical/cache disagreement; never change selected parameters."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from datasets import make_dataset
from models import SCEG
from utils.config import load_config, ModelConfig
from utils.experiment import write_json
from utils.reproducibility import seed_everything
from experiments.sceg_proker.head import VisualEpisodeHead

run = Path(__file__).resolve().parent
config = load_config(run/"config.yaml")
settings = config["proker"]
history = json.loads(Path(settings["historical_results"]["50way_1shot"]).read_text())
result = json.loads((run/"test_50way_1shot.json").read_text())
changed = np.flatnonzero(np.asarray(result["methods"]["baseline"]["episode_accuracies"])-
                         np.asarray(history["episode_accuracies"]))
assert changed.tolist() == [1171]
seed_everything(42, warn_only=True, allow_tf32=False)
torch.set_num_threads(2)
device = config["runtime"]["device"]
cache = torch.load(json.loads((run/"cache_reference.json").read_text())["path"],
                   map_location="cpu", weights_only=False)
indices = history["episode_indices"][1171]
dataset = make_dataset(config, "test")
images = torch.stack([dataset[i][0] for i in indices]).to(device)
payload = torch.load(settings["checkpoint"], map_location="cpu", weights_only=False)
model = SCEG(ModelConfig.from_dict(config["model"])).to(device).eval().requires_grad_(False)
model.load_state_dict(payload["model_state_dict"], strict=True)
del payload
torch.cuda.reset_peak_memory_stats(device)
torch.cuda.synchronize(device)
start = time.perf_counter()
with torch.inference_mode():
    cached = cache["splits"]["test"]["features"][indices].to(device)
    original_cached = model.classify(tuple(cached[:, s] for s in range(3)), 50, 1).scores
    full_episode = model.encode(images)  # Exact original evaluator batch size: 850.
    original_online = model.classify(full_episode, 50, 1).scores
    head = VisualEpisodeHead(config["model"]).fit(cached[:50], torch.arange(50, device=device), 50)
    f0 = head.baseline(cached[50:])
    prediction_changed = torch.nonzero(original_cached.argmax(1) != original_online.argmax(1)).flatten()
    truth = torch.arange(50, device=device).repeat_interleave(16)
    details = []
    for index in prediction_changed.tolist():
        details.append({"query_offset": index, "truth": int(truth[index]),
                        "cached_prediction": int(original_cached[index].argmax()),
                        "original_batch_prediction": int(original_online[index].argmax()),
                        "cached_top2_scores": original_cached[index].topk(2).values.tolist(),
                        "original_batch_top2_scores": original_online[index].topk(2).values.tolist()})
torch.cuda.synchronize(device)
record = {"episode_zero_based": 1171, "status": "diagnosed",
          "command": sys.orig_argv, "seconds": time.perf_counter()-start,
          "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
          "cached_original_accuracy": 100*(original_cached.argmax(1) == truth).sum().item()/800,
          "full_episode_original_accuracy": 100*(original_online.argmax(1) == truth).sum().item()/800,
          "historical_accuracy": history["episode_accuracies"][1171],
          "f0_equals_cached_original_div3": bool(torch.equal(f0, original_cached/3)),
          "div3_prediction_differences": int(((original_cached/3).argmax(1) != original_cached.argmax(1)).sum()),
          "maximum_original_score_difference": float((original_online-original_cached).abs().max()),
          "changed_predictions": details,
          "interpretation": "Batch-dependent float32 encoding differences at a tied decision; all paired comparisons use the same cached features. No hyperparameter changes."}
assert record["full_episode_original_accuracy"] == record["historical_accuracy"]
write_json(run/"replay_diagnostic.json", record)
print(json.dumps(record, indent=2))
