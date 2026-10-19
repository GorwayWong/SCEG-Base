"""One bounded Cars precheck, isolated from the formal seed42 process."""
import argparse
import hashlib
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import torch
from datasets import make_dataset, make_loader
from evaluators.fewshot_eval import encode_in_chunks
from losses import IntraInterLoss
from trainers.trainer import dropblock_counters
from .model import build_model
from .run import Run, DEFAULT_RUN, HERE, ROOT, SETTINGS, setup, sha256, historical_result


def command(*args):
    result = subprocess.run(["rtk", "proxy", *args], capture_output=True, text=True, timeout=20)
    return result.stdout.strip() if result.returncode == 0 else "unavailable: "+result.stderr.strip()


def memory():
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return {"wsl": {k: info[k].strip() for k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")},
            "swap_pages_in": int(vm["pswpin"]), "swap_pages_out": int(vm["pswpout"]),
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def resources_and_sources(run):
    prep_path = Path(run.info["data_manifest_reference"])
    prep = json.loads(prep_path.read_text())
    split_path = prep_path.parent/prep["split_manifest"]
    split = json.loads(split_path.read_text())
    assert prep["full_decode_passed"] and prep["data_ready"]
    assert prep["protocol_id"] == split["protocol_id"] == "sceg_author_cars_v1"
    assert prep["root"] == run.config["data"]["root"]
    assert sha256(split_path) == run.config["data_protocol"]["split_manifest_sha256"]
    assert sha256(prep_path.parent/prep["sample_manifest"]) == run.config["data_protocol"]["sample_manifest_sha256"]
    counts = {}
    for name, classes, images in (("train", 130, 10747), ("val", 17, 1410), ("test", 49, 4028)):
        dataset = make_dataset(run.config, name)
        assert len(dataset.classes) == classes and len(dataset) == images
        assert dataset.class_to_idx == split["class_to_idx"][name]
        counts[name] = {"classes": classes, "images": images}
    baseline = Path(run.spec["baseline_root"])/run.spec["baseline_run"]
    report = json.loads((baseline/"report.json").read_text())
    assert report["status"] == "completed" and report["best_sha256"] == run.spec["baseline_best_sha256"]
    assert sha256(baseline/"checkpoint.pth") == report["best_sha256"]
    # Read one historical task array at a time, then retain only its parseable reference.
    run.info["historical_results"] = {}
    for way, shot in SETTINGS:
        path = baseline/f"test_{way}way_{shot}shot.json"
        run.info["historical_results"][f"{way}way_{shot}shot"] = {"path": str(path), "sha256": sha256(path)}
        historical, _ = historical_result(run, dataset, way, shot)
        assert abs(historical["accuracy"]-report["results"][f"{way}way_{shot}shot"]["accuracy"]) < 1e-10
        del historical
    cub_root = Path(run.spec["module_root"])
    for path, commit in ((ROOT, run.spec["baseline_commit"]), (Path(run.spec["baseline_root"]), run.spec["baseline_commit"]),
                         (cub_root, run.spec["module_commit"])):
        assert command("git", "-C", str(path), "rev-parse", "HEAD") == commit
    evidence_path = cub_root/run.spec["module_evidence"]
    evidence = json.loads(evidence_path.read_text())
    module_identity = {}
    for filename in ("model.py", "lsconv.py", "ska.py"):
        relative = "experiments/sceg_lsconv_cub/"+filename
        assert sha256(HERE/filename) == sha256(cub_root/relative)
        snapshot = evidence["new_source"][relative]
        assert hashlib.sha256(snapshot.encode()).hexdigest() == evidence["source_sha256"][relative]
        # CUB's final commit strips trailing whitespace/empty EOF lines only.
        normalize = lambda text: "\n".join(line.rstrip() for line in text.splitlines()).rstrip()
        assert normalize(snapshot) == normalize((HERE/filename).read_text()), filename
        module_identity[filename] = {"fixed_commit_and_local_sha256": sha256(HERE/filename),
            "validated_snapshot_sha256": evidence["source_sha256"][relative],
            "difference": "none except trailing whitespace/empty EOF lines"}
    current, previous = run.info["environment"], evidence["environment"]
    for key in ("python", "torch_build", "cuda_runtime", "gpu", "platform"):
        assert current[key] == previous[key], f"Operator evidence environment changed: {key}"
    for package in ("torch", "torchvision", "triton"):
        assert current["packages"][package] == previous["packages"][package]
    assert evidence["precheck"]["operator"]["status"] == "passed"
    processes = command("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader")
    assert not processes.startswith("unavailable"), processes
    other = [line for line in processes.splitlines() if int(line.split(",")[0]) != os.getpid()]
    assert not other, f"Other CUDA workloads: {other}"
    disk = shutil.disk_usage(run.path)
    assert disk.free >= 1024**3
    return {"status": "passed", "splits": counts, "historical_pairing": "all four sets validated, 2000 each",
        "data_evidence": "Existing full-decode record plus manifest hashes/class mappings reused; no image rehash/preprocessing",
        "operator_reused": {"path": str(evidence_path), "sha256": sha256(evidence_path),
            "check": "precheck.operator", "result": evidence["precheck"]["operator"],
            "module_identity": module_identity,
            "basis": "Identical executable source (trailing whitespace only), operator shape, precision, Python/PyTorch/Triton/CUDA/GPU/platform"},
        "memory_before": memory(), "disk_free_bytes": disk.free,
        "gpu_before": command("nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version", "--format=csv,noheader"),
        "host_memory": command("powershell.exe", "-NoProfile", "-Command",
            "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json"),
        "effective_loader": {"workers": 4, "pin_memory": False, "prefetch_factor": 2, "persistent_workers": False},
        "precision": {"parameter_dtype": "float32", "amp": False, "tf32": False,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}}


def training_check(config):
    setup(config)
    baseline = build_model(config, enabled=False)
    baseline_rng = torch.get_rng_state().clone()
    baseline_count = sum(p.numel() for p in baseline.parameters())
    setup(config)
    model = build_model(config)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    reference = baseline.state_dict()
    for name, value in model.state_dict().items():
        if not name.startswith("SC.self_s."):
            assert torch.equal(value, reference[name]), name
    del reference, baseline
    model.to(config["runtime"]["device"]).train()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), **{key: config["train"][key]
        for key in ("lr", "momentum", "weight_decay", "nesterov")})
    criterion = IntraInterLoss(10, **config["loss"]).to(config["runtime"]["device"])
    global_criterion = IntraInterLoss(130, **config["loss"]).to(config["runtime"]["device"])
    assert not ({id(p) for p in criterion.parameters()} & {id(p) for g in optimizer.param_groups for p in g["params"]})
    targets = torch.arange(10, device=config["runtime"]["device"]).repeat_interleave(15)
    before = {n: p.detach().clone() for n, p in model.SC.self_s.named_parameters()}
    loader = make_loader(config, "train")
    assert loader.num_workers == 4 and loader.pin_memory is False and loader.prefetch_factor == 2
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    losses, elapsed = [], []
    seen_shapes = []
    hook = model.SC.self_s.register_forward_pre_hook(lambda m, args: seen_shapes.append(list(args[0].shape)))
    for update in range(5):
        images, labels = next(iterator)
        assert images.shape == (200, 3, 84, 84)
        support = labels[:50].reshape(10, 5)
        assert (support == support[:, :1]).all() and support[:, 0].unique().numel() == 10
        assert (labels[50:].reshape(10, 15) == support[:, :1]).all()
        images, labels = images.to(config["runtime"]["device"]), labels.to(config["runtime"]["device"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = model(images)
        assert all(item.shape == (150, 10) and torch.isfinite(item).all() for item in output[:3])
        assert output.global_cosine.shape == (200, 130) and torch.isfinite(output.global_cosine).all()
        terms = [criterion(cosine, targets) for cosine in output[:3]]
        terms.append(global_criterion(output.global_cosine, labels))
        loss = sum(terms)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        missing = {n for n, p in model.named_parameters() if p.grad is None}
        assert missing == model.expected_inactive_parameters(), missing
        assert any(n.startswith("SC.g_st.") for n in missing)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.SC.self_s.parameters())
        optimizer.step()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-started)
        losses.append([float(term.detach()) for term in terms])
        del output, terms, loss, images, labels
    hook.remove()
    assert seen_shapes == [[200, 320, 10, 10]]*5
    assert all(not torch.equal(p, before[n]) for n, p in model.SC.self_s.named_parameters())
    result = {"status": "passed", "updates": 5, "warmup_updates": 2, "four_loss_terms": losses,
        "warmup_seconds": elapsed[:2], "steady_update_seconds": elapsed[2:],
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "all_lsconv_parameters_finite_and_updated": True,
        "baseline_parameters": baseline_count, "parameters": sum(p.numel() for p in model.parameters()),
        "unchanged_parameters_and_cpu_rng": "bitwise matched original Cars construction",
        "inactive_parameters": sorted(missing), "stage3_shape": seen_shapes[0],
        "task_score_shapes": [[150, 10]]*3, "global_shape": [200, 130]}
    del optimizer, iterator, loader, before, criterion, global_criterion
    model.zero_grad(set_to_none=True)
    gc.collect()
    return model.eval(), result


@torch.no_grad()
def chunk_check(model, config):
    iterator = iter(make_loader(config, "val", way=10, shot=5, episodes=1))
    images, _ = next(iterator)
    assert images.shape == (210, 3, 84, 84)
    counters = dropblock_counters(model)
    buffers = {n: p.clone() for n, p in model.named_buffers()}
    torch.cuda.reset_peak_memory_stats()
    full = model.encode(images.to(config["runtime"]["device"]))
    full_output = model.classify(full, 10, 5)
    for name, module in model.named_modules():
        if name in counters:
            module.num_batches_tracked = counters[name]
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone()
    chunks = encode_in_chunks(model, images, 128)
    chunk_output = model.classify(chunks, 10, 5)
    differences = {}
    pairs = [(f"embedding_{i}", a, b) for i, (a, b) in enumerate(zip(full, chunks))]
    pairs += [(f"cosine_{i}", a, b) for i, (a, b) in enumerate(zip(full_output, chunk_output))]
    pairs += [("sum_scores", full_output.scores, chunk_output.scores)]
    for name, a, b in pairs:
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        differences[name] = float((a-b).abs().max())
    assert torch.equal(full_output.scores.argmax(1), chunk_output.scores.argmax(1))
    assert dropblock_counters(model) == {n: v+1 for n, v in counters.items()}
    assert all(torch.equal(p, buffers[n]) for n, p in model.named_buffers())
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
    return {"status": "passed", "split": "validation", "way": 10, "shot": 5, "query": 16,
        "images": 210, "chunk_sizes": [128, 82], "atol": 1e-5, "rtol": 1e-5,
        "max_absolute_errors": differences, "predictions_identical": True,
        "buffers_and_rng_unchanged": True, "logical_dropblock_increment": 1,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated()}


@torch.no_grad()
def maximum_check(model, config):
    images = torch.randn(1029, 3, 84, 84, device="cpu")
    counters = dropblock_counters(model)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    embeddings = encode_in_chunks(model, images, 128)
    assert all(value.shape == (1029, 640) for value in embeddings)
    output = model.classify(embeddings, 49, 5)
    assert output.scores.shape == (784, 49) and torch.isfinite(output.scores).all()
    assert dropblock_counters(model) == {n: v+1 for n, v in counters.items()}
    torch.cuda.synchronize()
    return {"status": "passed", "images": 1029, "input": "synthetic CPU tensor; no test images",
        "chunk": 128, "scores_shape": [784, 49], "seconds": time.perf_counter()-started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "logical_dropblock_increment": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--repair-precheck", action="store_true")
    args = parser.parse_args()
    run = Run(args.run_id, create=True, repair_precheck=args.repair_precheck)
    setup(run.config)
    started = time.perf_counter()
    checks = run.info.setdefault("precheck", {})
    try:
        checks["resources_sources_pairing"] = resources_and_sources(run)
        run.event("precheck", check="resources_sources_pairing", status="passed")
        run.save()
        model, checks["training_update"] = training_check(run.config)
        run.event("precheck", check="training_update", **checks["training_update"])
        run.save()
        checks["chunk_boundary"] = chunk_check(model, run.config)
        run.event("precheck", check="chunk_boundary", **checks["chunk_boundary"])
        run.save()
        checks["maximum_eval"] = maximum_check(model, run.config)
        run.event("precheck", check="maximum_eval", **checks["maximum_eval"])
        checks["memory_after"] = memory()
        checks["gpu_after"] = command("nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader")
        checks["memory_units"] = "PyTorch peak allocated bytes is distinct from nvidia-smi whole-card usage; Windows shared GPU memory not measured"
        run.info["stages"]["precheck"] = "completed"
        run.info["status"] = "ready"
    except BaseException as error:
        run.info["stages"]["precheck"] = "failed"
        run.info["status"] = "stopped"
        run.info["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        checks["total_seconds"] = time.perf_counter()-started
        run.save()


if __name__ == "__main__":
    main()
