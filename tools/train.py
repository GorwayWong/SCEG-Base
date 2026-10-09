"""Phase5 entrypoint, refuses training before overfit or unresolved strict-gradient gate."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from models import SCEG
from utils.config import load_config, ModelConfig
from utils.reproducibility import seed_everything
from utils.experiment import ExperimentRun, write_json
from trainers.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/sceg_original/config.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--accept-official-inactive", action="store_true",
                        help="Explicitly accept the documented official inactive-parameter exception (D01/D02)")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["experiment"] == "sceg_debug":
        raise ValueError("Debug runs use tools.overfit, not the original trainer")
    gate_path = Path("experiments/sceg_debug/result.json")
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "passed":
        raise RuntimeError("Overfit gate failed/unverified; full training forbidden")
    if not args.accept_official_inactive:
        raise RuntimeError("Strict all-parameter gradient gate failed in official code; explicit exception required")
    config["gates"] = {"overfit_result": str(gate_path), "overfit_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
                       "official_inactive_exception_accepted": True,
                       "strict_all_parameter_gradient_passed": False}
    if config["eval"]["query"] != 16:
        raise ValueError("Official formal evaluation uses query16")
    if config["train"]["optimizer"] != "sgd" or config["train"]["scheduler"] != "StepLR":
        raise ValueError("This entrypoint implements the released run.sh recipe")
    for split in ("train", "val_pre", "test_pre"):
        if not (Path(config["data"]["root"])/split).is_dir():
            raise FileNotFoundError(f"Missing prepared official split {split}")
    seed_everything(config["seed"], deterministic=config["runtime"]["deterministic"],
                    allow_tf32=config["runtime"]["allow_tf32"],
                    warn_only=config["runtime"]["deterministic_warn_only"])
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    run = ExperimentRun(config, args.run_id)
    try:
        manifest = Path("data/provenance/copied_datasets.json")
        if not manifest.exists():
            raise FileNotFoundError("Physical dataset copy audit missing")
        dataset_records = json.loads(manifest.read_text())
        data_root = Path(config["data"]["root"]).resolve()
        matches = [record for record in dataset_records.values()
                   if Path(record["destination"]).resolve() == data_root]
        if len(matches) != 1 or not (matches[0].get("verified_equal") or matches[0].get("verified_images_match_source")):
            raise ValueError("Configured dataset lacks a matching physical-copy provenance audit")
        from tools.prepare_data import tree_manifest
        actual = tree_manifest(data_root)
        if actual["tree_sha256"] != matches[0]["tree_sha256"]:
            raise ValueError("Dataset changed since copy audit")
        write_json(run.path/"data_manifest.json", matches[0])
        prepared = Path("data/provenance/pre_resized.json")
        write_json(run.path/"pre_resized_manifest.json", json.loads(prepared.read_text()))
        model = SCEG(ModelConfig.from_dict(config["model"])).to(config["runtime"]["device"])
        train(model, config, run, args.resume)
        run.publish_result()
    except BaseException as error:
        run.update_metrics(status="failed", error=f"{type(error).__name__}: {error}")
        run.publish_result()
        run.logger.exception("training terminated")
        raise


if __name__ == "__main__":
    main()
