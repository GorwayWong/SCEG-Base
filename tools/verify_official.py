"""Isolated official-code oracle; does not vendor or modify the reference.

Usage: python -m tools.verify_official --reference /tmp/sceg-official-reference
Checks complete state_dict compatibility, outputs, and missing-gradient sets
for both SC branches. Original imports stay in this tool's isolated process.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--gradients", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reference = Path(args.reference).resolve()
    # Avoid namespace collisions with the official un-packaged models/utils folders.
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != root]
    sys.path.insert(0, str(reference))
    import torch
    from models.SCEG import SCEG as OfficialSCEG
    config_module = load_module("reproduction_config", root/"utils/config.py")
    config = config_module.load_config(root/"configs/sceg_original.yaml")
    reproduction = load_module("reproduction_models", root/"models/__init__.py", package=True)
    seed_module = load_module("reproduction_seed", root/"utils/reproducibility.py")
    seed_module.seed_everything(config["seed"], warn_only=config["runtime"]["deterministic_warn_only"])
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    records = []
    for dataset in ("cub_raw", "stanford_car"):
        values = dict(config["model"], dataset=dataset, train_way=3, train_shot=1, train_query=1)
        official = OfficialSCEG(SimpleNamespace(dataset=dataset, train_way=3,
            train_shot=1, train_query_shot=1, base_class=values["base_classes"])).to(device)
        replica = reproduction.SCEG(config_module.ModelConfig.from_dict(values)).to(device)
        assert set(replica.state_dict()) == set(official.state_dict())
        replica.load_state_dict(official.state_dict(), strict=True)
        images = torch.randn(6, 3, values["image_size"], values["image_size"], device=device)
        official.eval(); replica.eval()
        with torch.no_grad():
            expected = official(images)
            actual = replica(images)
        errors = [(left-right).abs().max().item() for left, right in zip(expected, actual)]
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, atol=config["verification"]["parity_atol"],
                                        rtol=config["verification"]["parity_rtol"])
        record = {"dataset": dataset, "state_keys": len(official.state_dict()),
                  "eval_max_abs_errors": errors}
        if args.gradients:
            if not torch.cuda.is_available():
                raise RuntimeError("Official train/loss oracle requires CUDA")
            official_loss = load_module("official_loss", reference/"loss.py")
            replica_loss = load_module("replica_loss", root/"losses/intra_inter_loss.py")
            target_episode = torch.arange(3, device=device)
            target_global = target_episode.repeat(2)
            missing_by_model = []
            gradients = []
            train_outputs = []
            for model, loss_type in ((official, official_loss.Intra_Inter_Loss),
                                     (replica, replica_loss.IntraInterLoss)):
                seed_module.seed_everything(config["seed"], warn_only=config["runtime"]["deterministic_warn_only"])
                model.train()
                prediction = model(images)
                criterion = loss_type(3, **config["loss"]).to(device)
                global_criterion = loss_type(values["base_classes"], **config["loss"]).to(device)
                total_loss = sum(criterion(score, target_episode) for score in prediction[:3])
                total_loss = total_loss+global_criterion(prediction[3], target_global)
                total_loss.backward()
                missing_by_model.append({name for name, parameter in model.named_parameters()
                                         if parameter.grad is None})
                gradients.append({name: parameter.grad.detach().clone()
                                  for name, parameter in model.named_parameters()
                                  if parameter.grad is not None})
                train_outputs.append(prediction)
            assert missing_by_model[0] == missing_by_model[1] == replica.expected_inactive_parameters()
            for left, right in zip(*train_outputs):
                torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
            gradient_error = max((gradients[0][name]-gradients[1][name]).abs().max().item()
                                 for name in gradients[0])
            # exp/log loss magnifies last-bit cosine differences; report max absolute gradient error.
            record.update(missing_parameters=sorted(missing_by_model[0]),
                          gradient_max_abs_error=gradient_error,
                          active_parameters=len(gradients[0]))
        records.append(record)
        del official, replica
    print(json.dumps({"device": device, "records": records, "passed": True}, indent=2))


if __name__ == "__main__":
    main()
