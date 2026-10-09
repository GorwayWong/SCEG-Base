"""Phase4 only: four real CUB images, no augmentation/dropout/weight decay, 200 steps.

The same four originals are reused as support AND query (8 tensor rows) for
memorization debugging only. This deliberate leakage is not a few-shot evaluation.
Uses all three episodic losses and the global loss with four class representatives.
"""
import argparse
import hashlib
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from models import SCEG
from losses import IntraInterLoss
from utils.config import load_config, ModelConfig
from utils.reproducibility import seed_everything, rng_state
from utils.experiment import ExperimentRun, write_json
from utils.diagnostics import (assert_finite, activation_statistics,
                               gradient_summary, parameter_summary, estimate_flops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug_overfit.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = ModelConfig.from_dict(config["model"])
    assert settings.dropout == config["train"]["weight_decay"] == 0
    assert not config["data"]["augmentation"]
    assert config["debug"]["unique_images"] == settings.train_way == settings.base_classes
    assert settings.train_shot == settings.train_query == 1
    seed_everything(config["seed"], deterministic=config["runtime"]["deterministic"],
                    allow_tf32=config["runtime"]["allow_tf32"],
                    warn_only=config["runtime"]["deterministic_warn_only"])
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    device = torch.device(config["runtime"]["device"])
    run = ExperimentRun(config, args.run_id)
    try:
        data_root = Path(config["data"]["root"])/"train"
        classes = sorted(path for path in data_root.iterdir() if path.is_dir())[:settings.base_classes]
        assert len(classes) == settings.base_classes
        preprocess = transforms.Compose([
            transforms.Resize(config["data"]["eval_resize"]),
            transforms.CenterCrop(config["data"]["image_size"]), transforms.ToTensor(),
            transforms.Normalize(config["data"]["mean"], config["data"]["std"]),
        ])
        originals, provenance = [], []
        for class_index, class_path in enumerate(classes):
            image_path = next(path for path in sorted(class_path.iterdir()) if path.is_file())
            assert not image_path.is_symlink(), "Debug data must be a physical copy"
            with Image.open(image_path) as image:
                originals.append(preprocess(image.convert("RGB")))
            provenance.append({"path": str(image_path), "class_id": class_path.name,
                               "debug_class": class_index,
                               "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
        write_json(run.path/"data_manifest.json", {"images": provenance,
            "unique_images": len(originals), "support_query_reuse": True,
            "not_a_fewshot_metric": True})
        images = torch.stack(originals).repeat(2, 1, 1, 1).to(device)
        query_labels = torch.arange(settings.train_way, device=device)
        global_labels = query_labels.repeat(2)
        model = SCEG(settings).to(device)
        optimizer = torch.optim.SGD(model.parameters(), **{key: config["train"][key]
            for key in ("lr", "momentum", "nesterov", "weight_decay")})
        criterion = IntraInterLoss(settings.train_way, **config["loss"]).to(device)
        global_criterion = IntraInterLoss(settings.base_classes, **config["loss"]).to(device)
        diagnostics = {"parameters": parameter_summary(model),
                       "flops": estimate_flops(model, images, settings.train_way, settings.train_shot)}
        (run.path/"model_summary.txt").write_text(str(model))
        model.train()
        history = []
        for step in range(1, config["debug"]["steps"]+1):
            optimizer.zero_grad()
            with activation_statistics(model) as activations:
                output = model(images)
            assert_finite(output, f"step{step}.output")
            loss = sum(criterion(score, query_labels) for score in output[:3])
            loss = loss+global_criterion(output.global_cosine, global_labels)
            assert_finite(loss, f"step{step}.loss")
            loss.backward()
            gradients = gradient_summary(model)
            assert set(gradients["missing_parameters"]) == model.expected_inactive_parameters()
            optimizer.step()
            accuracy = (output.scores.argmax(1) == query_labels).float().mean().item()*100
            global_accuracy = (output.global_cosine.argmax(1) == global_labels).float().mean().item()*100
            history.append({"step": step, "loss": loss.item(), "accuracy": accuracy,
                            "global_accuracy": global_accuracy, "gradient_norm": gradients["global_norm"]})
            if step == 1 or step % config["debug"]["log_every"] == 0:
                run.logger.info("step=%d loss=%.8f episode_acc=%.2f global_acc=%.2f grad_norm=%.6f",
                    step, loss.item(), accuracy, global_accuracy, gradients["global_norm"])
                run.update_metrics(step=step, accuracy=accuracy, global_accuracy=global_accuracy, loss=loss.item())
        # Evaluate the post-update training-mode graph; eval-mode behavior is separately reported.
        with torch.no_grad(), activation_statistics(model) as final_activations:
            final_output = model(images)
            final_loss = sum(criterion(score, query_labels) for score in final_output[:3])
            final_loss = final_loss+global_criterion(final_output.global_cosine, global_labels)
            final_accuracy = (final_output.scores.argmax(1) == query_labels).float().mean().item()*100
            final_global_accuracy = (final_output.global_cosine.argmax(1) == global_labels).float().mean().item()*100
            model.eval()
            eval_output = model(images)
            eval_accuracy = (eval_output.scores.argmax(1) == query_labels).float().mean().item()*100
            eval_global_accuracy = (eval_output.global_cosine.argmax(1) == global_labels).float().mean().item()*100
        passed = (final_loss.item() <= config["debug"]["max_final_loss"]
                  and min(final_accuracy, final_global_accuracy) >= config["debug"]["minimum_accuracy"])
        diagnostics.update(final_activations=final_activations, final_gradients=gradients)
        write_json(run.path/"diagnostics.json", diagnostics)
        write_json(run.path/"history.json", history)
        run.checkpoint({"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": config, "steps": config["debug"]["steps"], "rng": rng_state(),
                        "dropblock_counters": {name: module.num_batches_tracked
                            for name, module in model.named_modules() if hasattr(module, "ramp_batches")}})
        run.update_metrics(status="passed" if passed else "failed", overfit_passed=passed,
            accuracy=final_accuracy, global_accuracy=final_global_accuracy, loss=final_loss.item(),
            eval_accuracy=eval_accuracy, eval_global_accuracy=eval_global_accuracy,
            strict_all_parameter_gradient_passed=False, metric_scope="four-image memorization only",
            theoretical_positive_loss_floor=4*torch.nn.functional.softplus(
                torch.tensor(-config["loss"]["lamda"]*(1-config["loss"]["intra"]))).item())
        run.logger.info("overfit_passed=%s final_loss=%.8f run=%s", passed, final_loss.item(), run.path)
        if not passed:
            raise SystemExit("Overfit gate FAILED; full training forbidden")
    except BaseException as error:
        if run.metrics["status"] == "running":
            run.update_metrics(status="failed", error=f"{type(error).__name__}: {error}")
        run.logger.exception("run terminated")
        raise


if __name__ == "__main__":
    main()
