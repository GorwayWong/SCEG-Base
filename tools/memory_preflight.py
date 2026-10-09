"""One full-size random episode forward/backward; zero optimizer updates, no performance claim."""
import argparse
import torch
from models import SCEG
from losses import IntraInterLoss
from utils.config import load_config, ModelConfig
from utils.reproducibility import seed_everything
from utils.experiment import ExperimentRun, write_json
from utils.diagnostics import assert_finite, gradient_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/sceg_preflight/config.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_config(args.config)
    config["experiment"] = "sceg_preflight"
    config["purpose"] = "random full-episode memory/gradient preflight; zero optimizer steps"
    seed_everything(config["seed"], warn_only=config["runtime"]["deterministic_warn_only"])
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    settings = ModelConfig.from_dict(config["model"])
    run = ExperimentRun(config, args.run_id)
    device = config["runtime"]["device"]
    try:
        model = SCEG(settings).to(device).train()
        batch = settings.train_way*(settings.train_shot+settings.train_query)
        images = torch.randn(batch, settings.input_channels, settings.image_size,
                             settings.image_size, device=device)
        query_labels = torch.arange(settings.train_way, device=device).repeat_interleave(settings.train_query)
        global_labels = torch.cat((torch.arange(settings.train_way, device=device).repeat_interleave(settings.train_shot),
                                   query_labels))
        write_json(run.path/"data_manifest.json", {"type": "torch.randn", "seed": config["seed"],
                                                   "shape": list(images.shape), "not_a_dataset_result": True})
        torch.cuda.reset_peak_memory_stats()
        output = model(images)
        criterion = IntraInterLoss(settings.train_way, **config["loss"]).to(device)
        global_criterion = IntraInterLoss(settings.base_classes, **config["loss"]).to(device)
        loss = sum(criterion(cosine, query_labels) for cosine in output[:3])
        loss = loss+global_criterion(output.global_cosine, global_labels)
        assert_finite(loss, "full_episode.loss")
        loss.backward()
        gradients = gradient_summary(model)
        assert set(gradients["missing_parameters"]) == model.expected_inactive_parameters()
        run.checkpoint({"model_state_dict": model.state_dict(), "config": config,
                        "optimizer_updates": 0, "untrained_weights": True})
        run.update_metrics(status="completed", forward_backward_passed=True, optimizer_updates=0,
            full_training_completed=False, batch_size=batch, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), gradient_norm=gradients["global_norm"],
            strict_all_parameter_gradient_passed=False, metric_scope="resource/gradient preflight only")
        run.publish_result()
        run.logger.info("full_episode batch=%d peak_allocated_GiB=%.4f zero optimizer steps",
                        batch, run.metrics["peak_allocated_bytes"]/(1024**3))
    except BaseException as error:
        run.update_metrics(status="failed", error=f"{type(error).__name__}: {error}")
        run.publish_result()
        raise


if __name__ == "__main__":
    main()
