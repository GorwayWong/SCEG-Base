"""One bounded precheck process; no random checkpoints and no test-set tuning."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time

import torch
import torch.nn.functional as F
from datasets import make_dataset, make_loader
from evaluators.fewshot_eval import encode_in_chunks
from losses import IntraInterLoss
from trainers.trainer import dropblock_counters
from .model import build_model
from .run import Run, DEFAULT_RUN, ROOT, setup


def command(*args):
    try:
        p = subprocess.run(["rtk", "proxy", *args], capture_output=True, text=True, timeout=20)
        return p.stdout.strip() if p.returncode == 0 else "unknown: "+p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return "unknown: "+str(error)


def resources(run):
    manifest = json.loads(Path(run.info["data_manifest_reference"]).read_text())
    counts = {}
    for split, count, images in (("train", 100, 5903), ("val", 50, 2927), ("test", 50, 2958)):
        dataset = make_dataset(run.config, split)
        assert dataset.classes == manifest[split+"_classes"]
        assert len(dataset.classes) == count and len(dataset) == images
        counts[split] = {"classes": count, "images": images, "identity": "matches existing manifest"}
    free = shutil.disk_usage(run.path).free
    # 17M FP32 parameters: last with momentum ~135MB, best ~68MB;
    # atomic last+best replacements add ~203MB. Reserve 1GiB total incl logs/index refs.
    assert free >= 1024**3, "Insufficient disk for two weights and atomic-write peak"
    processes = command("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader")
    assert not processes or processes.startswith("unknown"), "Another CUDA compute process is running: "+processes
    return {"splits": counts, "train_episode_images": 200, "maximum_eval_images": 1050,
        "gpu": command("nvidia-smi", "--query-gpu=name,memory.used,memory.free,memory.total", "--format=csv,noheader"),
        "gpu_processes": processes, "wsl_memory": command("free", "-m"),
        "host_memory": command("powershell.exe", "-NoProfile", "-Command",
            "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json"),
        "disk_free_bytes": free, "disk_reserved_budget_bytes": 1024**3,
        "dataset_policy": "metadata and class IDs only; reused existing decoded/hash manifest"}


def operator_check(config):
    from .ska import SKA
    device = config["runtime"]["device"]
    x = torch.randn(2, 320, 10, 10, device=device, requires_grad=True)
    w = torch.randn(2, 40, 9, 10, 10, device=device, requires_grad=True)
    upstream = torch.randn_like(x)  # Shared, explicitly nonuniform for both backward paths.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    y = SKA()(x, w)
    actual_grad = torch.autograd.grad(y, (x, w), upstream)
    torch.cuda.synchronize()
    first = time.perf_counter()-started
    ref_x = x.detach().clone().requires_grad_()
    ref_w = w.detach().clone().requires_grad_()
    padded = F.pad(ref_x, (1, 1, 1, 1))
    index = torch.arange(320, device=device) % 40
    reference = sum(padded[:, :, kh:kh+10, kw:kw+10]*ref_w[:, index, kh*3+kw]
                    for kh in range(3) for kw in range(3))
    reference_grad = torch.autograd.grad(reference, (ref_x, ref_w), upstream)
    atol, rtol = config["verification"]["parity_atol"], config["verification"]["parity_rtol"]
    errors = {}
    for name, actual, expected in zip(("forward", "input_gradient", "kernel_gradient"),
                                    (y, *actual_grad), (reference, *reference_grad)):
        errors[name] = float((actual-expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    # Full tensor includes all four zero-padded borders; wrong c//8 sharing must fail.
    wrong = sum(padded[:, :, kh:kh+10, kw:kw+10]*ref_w[:, torch.arange(320, device=device)//8, kh*3+kw]
                for kh in range(3) for kw in range(3))
    assert not torch.allclose(y, wrong, atol=atol, rtol=rtol)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(3):
        value = SKA()(x, w)
        torch.autograd.grad(value, (x, w), upstream)
    torch.cuda.synchronize()
    return {"status": "passed", "shape": [2, 320, 10, 10], "dynamic_kernel": [2, 40, 9, 10, 10],
        "atol": atol, "rtol": rtol, "max_abs_errors": errors,
        "modulo40_and_zero_padding_verified": True,
        "first_compile_forward_backward_seconds": first,
        "steady_forward_backward_seconds": (time.perf_counter()-started)/3,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated()}


def training_check(config):
    setup(config)
    baseline = build_model(config, enabled=False)
    after_baseline = torch.get_rng_state().clone()
    baseline_count = sum(p.numel() for p in baseline.parameters())
    setup(config)
    model = build_model(config)
    assert torch.equal(torch.get_rng_state(), after_baseline)
    unchanged = baseline.state_dict()
    for name, tensor in model.state_dict().items():
        if not name.startswith("SC.self_s."):
            assert torch.equal(tensor, unchanged[name]), name
    del unchanged, baseline
    model.to(config["runtime"]["device"]).train()
    optimizer = torch.optim.SGD(model.parameters(), **{key: config["train"][key]
        for key in ("lr", "momentum", "weight_decay", "nesterov")})
    criterion = IntraInterLoss(10, **config["loss"]).to(config["runtime"]["device"])
    global_criterion = IntraInterLoss(100, **config["loss"]).to(config["runtime"]["device"])
    targets = torch.arange(10, device=config["runtime"]["device"]).repeat_interleave(15)
    before = {n: p.detach().clone() for n, p in model.SC.self_s.named_parameters()}
    iterator = iter(make_loader(config, "train"))
    torch.cuda.reset_peak_memory_stats()
    losses, elapsed = [], []
    for update in range(5):
        images, labels = next(iterator)
        assert images.shape == (200, 3, 84, 84)
        support = labels[:50].reshape(10, 5)
        assert (support == support[:, :1]).all()
        assert (labels[50:].reshape(10, 15) == support[:, :1]).all()
        images, labels = images.to(config["runtime"]["device"]), labels.to(config["runtime"]["device"])
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(images)
        assert all(item.shape == (150, 10) for item in output[:3])
        assert output.global_cosine.shape == (200, 100)
        loss = sum(criterion(cosine, targets) for cosine in output[:3])
        loss = loss + global_criterion(output.global_cosine, labels)
        assert torch.isfinite(loss), "Nonfinite training loss"
        optimizer.zero_grad()
        loss.backward()
        if update == 0:
            missing = {n for n, p in model.named_parameters() if p.grad is None}
            assert missing == model.expected_inactive_parameters(), missing
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.SC.self_s.parameters())
        optimizer.step()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start)
        losses.append(float(loss.detach()))
        del output, loss, images, labels
    updated = [n for n, p in model.SC.self_s.named_parameters() if not torch.equal(p, before[n])]
    assert len(updated) == len(before), "LSConv parameter did not update"
    result = {"status": "passed", "updates": 5, "warmup_updates": 2, "losses": losses,
        "warmup_seconds": elapsed[:2], "steady_update_seconds": elapsed[2:],
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "all_lsconv_parameters_updated": True,
        "baseline_parameters": baseline_count, "parameters": sum(p.numel() for p in model.parameters()),
        "unchanged_initialization_and_cpu_rng": "exactly matched baseline", "inactive_parameters": sorted(missing)}
    del model, optimizer, iterator, before, criterion, global_criterion
    gc.collect()
    return result


@torch.no_grad()
def maximum_check(config):
    model = build_model(config).to(config["runtime"]["device"]).eval()
    iterator = iter(make_loader(config, "val", way=50, shot=5, episodes=1))
    images, _ = next(iterator)
    assert images.shape == (1050, 3, 84, 84) and images.device.type == "cpu"
    counters = dropblock_counters(model)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    embeddings = encode_in_chunks(model, images, config["execution"]["encode_chunk_size"])
    assert all(value.shape == (1050, 640) for value in embeddings)
    output = model.classify(embeddings, 50, 5)
    assert output.scores.shape == (800, 50) and torch.isfinite(output.scores).all()
    assert dropblock_counters(model) == {name: value+1 for name, value in counters.items()}
    torch.cuda.synchronize()
    return {"status": "passed", "images": 1050, "split": "validation", "chunk": 128,
        "scores_shape": [800, 50], "seconds": time.perf_counter()-start,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "logical_dropblock_increment": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    args = parser.parse_args()
    config = __import__("experiments.sceg_lsconv_cub.run", fromlist=["configuration"]).configuration()[0]
    setup(config)
    run = Run(args.run_id, create=True)
    if run.info["stages"]["precheck"] == "completed":
        raise RuntimeError("Precheck already passed; do not repeat unaffected checks")
    started = time.perf_counter()
    checks = run.info.setdefault("precheck", {})
    try:
        checks["resources"] = resources(run)
        run.save()
        for name, function in (("operator", operator_check), ("training_update", training_check), ("maximum_eval", maximum_check)):
            checks[name] = function(config)
            run.event("precheck", check=name, **checks[name])
            run.save()
        run.info["stages"]["precheck"] = "completed"
        run.info["status"] = "ready"
    except BaseException as error:
        run.info["stages"]["precheck"] = "failed"
        run.info["status"] = "stopped"
        run.info["error"] = f"{type(error).__name__}: {error}"
        run.event("precheck_failed", error=run.info["error"])
        raise
    finally:
        checks["total_seconds"] = time.perf_counter()-started
        run.save()


if __name__ == "__main__":
    main()
