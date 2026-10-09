"""Isolated formal evaluation of an explicit checkpoint. Never uses the debug checkpoint."""
import argparse
import hashlib
from pathlib import Path
import torch
from models import SCEG
from evaluators import evaluate_episodes
from utils.config import load_config, ModelConfig
from utils.experiment import ExperimentRun, write_json
from utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/sceg_original/config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--way", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "config" in payload and payload["config"]["experiment"] != config["experiment"]:
        raise ValueError("Checkpoint experiment differs from evaluation configuration")
    if args.way is not None:
        config["eval"]["way"] = args.way
    config["purpose"] = "formal_evaluation"
    config["checkpoint_source"] = str(args.checkpoint.resolve())
    seed_everything(config["seed"], warn_only=config["runtime"]["deterministic_warn_only"],
                    allow_tf32=config["runtime"]["allow_tf32"])
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    model = SCEG(ModelConfig.from_dict(config["model"])).to(config["runtime"]["device"])
    model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    run = ExperimentRun(config, args.run_id)
    try:
        import shutil
        shutil.copy2(args.checkpoint, run.path/"checkpoint.pth")
        write_json(run.path/"checkpoint_manifest.json",
            {"source": str(args.checkpoint.resolve()), "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()})
        results = []
        for shot in config["eval"]["shots"]:
            result = evaluate_episodes(model, config, "test", config["eval"]["way"], shot,
                                      config["eval"]["test_episodes"])
            write_json(run.path/f"test_{result['way']}way_{shot}shot.json", result)
            results.append({key: value for key, value in result.items()
                            if key not in {"episode_indices", "sample_index_paths", "episode_accuracies"}})
        write_json(run.path/"evaluation.json", results)
        run.update_metrics(status="completed", evaluation=results, metric_scope="held-out test episodes",
            way=results[0]["way"], shot=results[0]["shot"], accuracy=results[0]["accuracy"],
            confidence_interval=results[0]["confidence_interval"])
        run.publish_result()
    except BaseException as error:
        run.update_metrics(status="failed", error=f"{type(error).__name__}: {error}")
        run.publish_result()
        raise


if __name__ == "__main__":
    main()
