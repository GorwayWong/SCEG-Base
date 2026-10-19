"""SCEG_train.py/trainer.py-equivalent single-GPU loop, with auditable checkpoints.

Four equal loss terms, fixed loss lambda excluded from optimizer, source
weighted-depletion epochs, validation every20, best-val checkpoint, StepLR.
Engineering additions: logs/finite checks/complete resume state; no AMP,
gradient accumulation, clipping, early stopping, or alternative method.
"""
from pathlib import Path
import numpy as np
import random
import time
import torch
from datasets import make_loader
from evaluators import evaluate_episodes
from losses import IntraInterLoss
from utils.reproducibility import rng_state
from utils.diagnostics import assert_finite, gradient_summary, parameter_summary, activation_statistics
from utils.experiment import write_json


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["cuda"]])


def dropblock_counters(model):
    return {name: module.num_batches_tracked for name, module in model.named_modules()
            if hasattr(module, "ramp_batches")}


def train(model, config, run, resume=None, *, lightweight=False, evaluator=evaluate_episodes):
    settings = config["model"]
    optimizer = torch.optim.SGD(model.parameters(), **{key: config["train"][key]
        for key in ("lr", "momentum", "weight_decay", "nesterov")})
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=config["train"]["scheduler_step"], gamma=config["train"]["scheduler_gamma"])
    loader = make_loader(config, "train")
    assert len(loader.dataset.classes) == settings["base_classes"]
    device = next(model.parameters()).device
    target_query = torch.arange(settings["train_way"], device=device).repeat_interleave(settings["train_query"])
    first_epoch, step, best_validation, best_epoch = 0, 0, 0.0, None
    if resume is not None:
        payload = torch.load(resume, map_location=device, weights_only=False)
        for key in ("model", "train", "loss", "data", "seed", "official_commit", *(["replacement", "execution", "run_id"] if lightweight else [])):
            if payload["config"][key] != config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng(payload["rng"])
        for name, module in model.named_modules():
            if name in payload["dropblock_counters"]:
                module.num_batches_tracked = payload["dropblock_counters"][name]
        first_epoch, step = payload["completed_epochs"], payload["step"]
        best_validation, best_epoch = payload["best_validation"], payload["best_epoch"]
        if best_epoch is not None:
            best_path = Path(resume).parent/"checkpoint.pth"
            if not best_path.exists():
                raise FileNotFoundError("Resume needs the corresponding best-validation checkpoint")
            import shutil
            if best_path.resolve() != (run.path/"checkpoint.pth").resolve():
                shutil.copy2(best_path, run.path/"checkpoint.pth")
        del payload
    if not lightweight:
        (run.path/"model_summary.txt").write_text(str(model))
        write_json(run.path/"parameter_summary.json", parameter_summary(model))
    for epoch in range(first_epoch, config["train"]["epochs"]):
        epoch_started = time.perf_counter()
        progress_at = epoch_started
        epoch_lr = optimizer.param_groups[0]["lr"]
        model.train()
        # Recreate both loss modules each epoch exactly as official default_train.
        criterion = IntraInterLoss(settings["train_way"], **config["loss"]).to(device)
        global_criterion = IntraInterLoss(settings["base_classes"], **config["loss"]).to(device)
        loss_sum, accuracy_sum, batches = 0.0, 0.0, 0
        for images, global_labels in loader:
            images, global_labels = images.to(device), global_labels.to(device)
            if batches == 0 and not lightweight and config.get("runtime", {}).get("record_activations", True):
                with activation_statistics(model) as activations:
                    output = model(images)
                write_json(run.path/f"activations_epoch_{epoch+1}.json", activations)
            else:
                output = model(images)
            assert_finite(output, f"epoch{epoch+1}.forward")
            total_loss = sum(criterion(cosine, target_query) for cosine in output[:3])
            total_loss = total_loss+global_criterion(output.global_cosine, global_labels)
            assert_finite(total_loss, f"epoch{epoch+1}.loss")
            optimizer.zero_grad()
            total_loss.backward()
            if not lightweight:
                gradients = gradient_summary(model)
                assert set(gradients["missing_parameters"]) == model.expected_inactive_parameters()
            optimizer.step()
            accuracy = 100*(output.scores.detach().argmax(1) == target_query).sum()/target_query.numel()
            if lightweight:
                loss_sum = loss_sum + total_loss.detach()
                accuracy_sum = accuracy_sum + accuracy
            else:
                loss_sum += total_loss.item(); accuracy_sum += accuracy.item()
            batches += 1; step += 1
            if lightweight:
                del output, total_loss, images, global_labels, accuracy
                if time.perf_counter()-progress_at >= 60:
                    run.logger.info("train epoch=%d/%d episodes=%d elapsed=%.1fs", epoch+1,
                                    config["train"]["epochs"], batches, time.perf_counter()-epoch_started)
                    progress_at = time.perf_counter()
        if batches == 0:
            raise RuntimeError("No complete official training episodes")
        validation = None
        improved = False
        if lightweight and device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter()-epoch_started
        validation_started = time.perf_counter()
        if (epoch+1) % config["train"]["validation_every"] == 0:
            validation = evaluator(model, config, "val", config["eval"]["way"],
                config["eval"]["validation_shot"], config["eval"]["validation_episodes"])
            if not lightweight:
                write_json(run.path/f"validation_epoch_{epoch+1}.json", validation)
            if validation["accuracy"] > best_validation:
                best_validation, best_epoch = validation["accuracy"], epoch+1
                improved = True
            model.train()
        validation_seconds = time.perf_counter()-validation_started if validation is not None else 0.0
        # Step occurs after validation, exactly as official trainer.py:248.
        scheduler.step()
        save_started = time.perf_counter()
        if not lightweight or validation is not None or epoch+1 == config["train"]["epochs"]:
            payload = {"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "config": config, "completed_epochs": epoch+1,
                       "step": step, "rng": rng_state(), "dropblock_counters": dropblock_counters(model),
                       "best_validation": best_validation, "best_epoch": best_epoch}
            if lightweight:
                payload["source_sha256"] = run.info["source_sha256"]
                payload["checkpoint_role"] = "last"
            partial = run.path/"latest.pth.part"
            torch.save(payload, partial); partial.replace(run.path/"latest.pth")
            if improved or (best_epoch is None and not lightweight):
                if lightweight:
                    payload["checkpoint_role"] = "best_validation"
                run.checkpoint(payload)
            del payload
        save_seconds = time.perf_counter()-save_started
        if lightweight:
            loss_sum, accuracy_sum = loss_sum.item(), accuracy_sum.item()
        timing = {}
        if config.get("runtime", {}).get("record_epoch_timing", False):
            timing = {"epoch_seconds": time.perf_counter()-epoch_started,
                      "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                      "save_seconds": save_seconds, "learning_rate_used": epoch_lr,
                      "validation_accuracy": validation["accuracy"] if validation else None,
                      "validation_ci": validation["confidence_interval"] if validation else None,
                      "validation_episodes": validation["episodes"] if validation else None,
                      "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                          if device.type == "cuda" else 0}
        run.update_metrics(completed_epochs=epoch+1, step=step, epoch_episodes=batches,
            train_loss=loss_sum/batches, train_accuracy=accuracy_sum/batches,
            **({} if lightweight else {"gradient_norm": gradients["global_norm"]}), learning_rate=scheduler.get_last_lr()[0],
            best_validation=best_validation, best_epoch=best_epoch,
            checkpoint_role="best_validation" if best_epoch is not None else "provisional_last", **timing)
        if not lightweight:
            run.logger.info("epoch=%d/%d episodes=%d loss=%.6f train_acc=%.4f best_val=%.4f lr=%.6g",
                epoch+1, config["train"]["epochs"], batches, loss_sum/batches,
                accuracy_sum/batches, best_validation, optimizer.param_groups[0]["lr"])
    if best_epoch is None:
        raise RuntimeError("No best-validation checkpoint selected; do not report a paper result")
    if not config.get("runtime", {}).get("post_training_evaluation", True):
        run.update_metrics(training_completed=True, full_training_completed=True,
                           full_evaluation_completed=False, status="running")
        return
    payload = torch.load(run.path/"checkpoint.pth", map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    results = []
    for shot in config["eval"]["shots"]:
        result = evaluate_episodes(model, config, "test", config["eval"]["way"], shot,
                                  config["eval"]["test_episodes"])
        write_json(run.path/f"test_{result['way']}way_{shot}shot.json", result)
        results.append({key: value for key, value in result.items()
                        if key not in {"episode_indices", "sample_index_paths", "episode_accuracies", "rng_after"}})
    run.update_metrics(status="completed", full_training_completed=True, evaluation=results,
        way=results[0]["way"], shot=results[0]["shot"], accuracy=results[0]["accuracy"],
        confidence_interval=results[0]["confidence_interval"], metric_scope="held-out test episodes")
