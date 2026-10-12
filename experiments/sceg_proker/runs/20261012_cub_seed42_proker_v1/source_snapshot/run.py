"""Frozen CUB cache -> validation selection -> locked paired historical replay.

Run from repository root with the EXISTING baseline Python environment.
Stages are restartable; a completed stage is reused, never retuned on test results.
"""
import argparse
import csv
import hashlib
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from datasets import make_dataset
from datasets.samplers import EvaluationEpisodeSampler
from models import SCEG
from utils.config import load_config, ModelConfig
from utils.experiment import ExperimentRun, write_json
from utils.reproducibility import seed_everything
from .head import VisualEpisodeHead

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT/"experiments/sceg_proker"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"accuracy": float(values.mean()),
            "confidence_interval": float(1.96*np.sqrt(values.var(ddof=0)/len(values))),
            "episodes": len(values)}


def snapshot(run_path):
    # git diff does not include this untracked experiment: retain a small complete snapshot.
    destination = run_path/"source_snapshot"
    destination.mkdir(exist_ok=True)
    manifest = {}
    for path in sorted(FAMILY.iterdir()):
        if path.suffix in {".py", ".yaml"}:
            shutil.copy2(path, destination/path.name)
            manifest[path.name] = sha256(path)
    write_json(run_path/"experiment_source_manifest.json", manifest)


def paths(dataset):
    return [str(Path(path).relative_to(dataset.root)) for path, _ in dataset.samples]


def cache_features(config, settings, run):
    checkpoint = Path(settings["checkpoint"])
    actual_sha = sha256(checkpoint)
    if actual_sha != settings["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the verified baseline best")
    datasets = {split: make_dataset(config, split) for split in ("val", "test")}
    identity = {"checkpoint_sha256": actual_sha, "model": config["model"],
                "preprocessing": {key: config["data"][key] for key in
                                  ("image_size", "mean", "std", "pre_resized", "eval_resize")},
                "mapping_sha256": {split: digest_json(paths(ds)) for split, ds in datasets.items()},
                "baseline_source_sha256": json.loads((run.path/"source_manifest.json").read_text())}
    cache_dir = FAMILY/"cache"/digest_json(identity)[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir/"features.pth"
    device = config["runtime"]["device"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SCEG(ModelConfig.from_dict(config["model"])).to(device)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    model.eval().requires_grad_(False)
    del payload
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    bn_before = {name: buffer.clone() for name, buffer in model.named_buffers()
                 if "running_" in name or name.endswith("num_batches_tracked")}
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache["identity"] != identity:
            raise ValueError("Cache identity mismatch")
        run.logger.info("reused feature cache %s", cache_path)
    else:
        cache = {"identity": identity, "splits": {}}
        torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            for split, ds in datasets.items():
                chunks = []
                loader = DataLoader(ds, batch_size=settings["encoding_batch_size"], shuffle=False,
                                    num_workers=config["data"]["workers"], pin_memory=False)
                split_start = time.perf_counter()
                for batch, (images, _) in enumerate(loader):
                    embedding = model.encode(images.to(device))
                    chunks.append(torch.stack(embedding, dim=1).float().cpu())
                    if batch % 20 == 0:
                        run.logger.info("encoding %s %d/%d batches", split, batch+1, len(loader))
                sync(device)
                features = torch.cat(chunks)
                assert features.shape == (len(ds), 3, 640) and torch.isfinite(features).all()
                cache["splits"][split] = {"features": features, "sample_index_paths": paths(ds),
                                          "class_ids": [label for _, label in ds.samples],
                                          "encoding_seconds": time.perf_counter()-split_start}
        cache["encoding_seconds"] = time.perf_counter()-start
        cache["encoding_peak_gpu_bytes"] = torch.cuda.max_memory_allocated(device)
        temporary = cache_path.with_suffix(".pth.part")
        torch.save(cache, temporary)
        temporary.replace(cache_path)
        write_json(cache_dir/"metadata.json", {key: value for key, value in cache.items() if key != "splits"} |
                   {"splits": {split: {key: value for key, value in data.items() if key != "features"}
                               for split, data in cache["splits"].items()}})
    for split, ds in datasets.items():
        assert cache["splits"][split]["sample_index_paths"] == paths(ds)
    for name, buffer in model.named_buffers():
        if name in bn_before:
            assert torch.equal(buffer, bn_before[name]), f"BN updated in eval: {name}"
    write_json(run.path/"cache_reference.json", {"path": str(cache_path), "sha256": sha256(cache_path),
               "encoding_seconds": cache["encoding_seconds"],
               "encoding_peak_gpu_bytes": cache["encoding_peak_gpu_bytes"], "identity": identity,
               "input_checkpoint": str(checkpoint), "input_checkpoint_sha256": actual_sha,
               "data_manifest": settings["data_manifest"], "data_manifest_sha256": sha256(settings["data_manifest"]),
               "feature_shape": "[images,stage2/stage3/stage4,640],float32",
               "frozen_parameters": sum(p.numel() for p in model.parameters()), "bn_unchanged": True})
    online_regression(config, settings, run, model, datasets, cache)
    del model, bn_before
    torch.cuda.empty_cache()
    return cache, datasets


def online_regression(config, settings, run, model, datasets, cache):
    if (run.path/"online_cache_regression.json").exists():
        return
    checks = []
    device = config["runtime"]["device"]
    with torch.inference_mode():
        for key in ("5way_1shot", "5way_5shot", "50way_5shot"):
            original = json.loads(Path(settings["historical_results"][key]).read_text())
            way, shot = original["way"], original["shot"]
            assert original["sample_index_paths"] == cache["splits"]["test"]["sample_index_paths"]
            for episode_number, indices in enumerate(original["episode_indices"][:2]):
                images = torch.stack([datasets["test"][i][0] for i in indices])
                chunks = [model.encode(chunk.to(device)) for chunk in images.split(settings["encoding_batch_size"])]
                online = tuple(torch.cat([chunk[s] for chunk in chunks]) for s in range(3))
                cached = cache["splits"]["test"]["features"][indices].to(device)
                original_scores = model.classify(online, way, shot).scores/3
                labels = torch.arange(way, device=device).repeat_interleave(shot)
                head = VisualEpisodeHead(config["model"]).fit(cached[:way*shot], labels, way)
                cached_scores = head.baseline(cached[way*shot:])
                torch.testing.assert_close(cached_scores, original_scores, atol=1e-5, rtol=1e-5)
                assert torch.equal(cached_scores.argmax(1), original_scores.argmax(1))
                zero = head.proker(cached[way*shot:], 5., 1., 0.)
                assert torch.equal(zero, cached_scores)
                checks.append({"setting": key, "episode": episode_number,
                               "max_score_error": float((cached_scores-original_scores).abs().max()),
                               "predictions_equal": True, "zero_residual_equal": True})
    write_json(run.path/"online_cache_regression.json", {"status": "passed", "checks": checks})
    run.logger.info("online/cache regression passed: %d real episodes", len(checks))


def manifests(config, settings, run, datasets, cache):
    directory = run.path/"episodes"
    directory.mkdir(exist_ok=True)
    output = {}
    for shot in (1, 5):
        for phase, way, count in (("coarse", 5, settings["coarse_episodes"]),
                                  ("coarse", 50, settings["coarse_episodes"]),
                                  ("confirm", 50, settings["confirm_episodes"])):
            key = f"{phase}_{way}way_{shot}shot"
            path = directory/f"{key}.npz"
            if not path.exists():
                np.random.seed(settings["validation_seeds"][key])
                sampler = EvaluationEpisodeSampler(datasets["val"], way, shot, count, query=16)
                np.savez_compressed(path, episode_indices=np.asarray(list(sampler), dtype=np.int32))
            output[key] = np.load(path)["episode_indices"]
    metadata = {}
    for key, source in settings["historical_results"].items():
        original = json.loads(Path(source).read_text())
        assert original["sample_index_paths"] == cache["splits"]["test"]["sample_index_paths"]
        assert original["episodes"] == settings["formal_episodes"] and original["query"] == 16
        array = np.asarray(original["episode_indices"], dtype=np.int32)
        path = directory/f"test_{key}.npz"
        if path.exists():
            assert np.array_equal(array, np.load(path)["episode_indices"])
        else:
            np.savez_compressed(path, episode_indices=array)
        output[f"test_{key}"] = array
        metadata[f"test_{key}"] = {"source": source, "source_sha256": sha256(source),
                                   "replay": "exact saved indices + exact sorted path mapping"}
    for key, indices in output.items():
        split = "test" if key.startswith("test_") else "val"
        way, shot = [int(part.replace("way", "").replace("shot", "")) for part in key.split("_")[1:]]
        labels = np.asarray(cache["splits"][split]["class_ids"])
        assert indices.shape == (2000 if split == "test" else
                                 settings["confirm_episodes"] if key.startswith("confirm") else settings["coarse_episodes"],
                                 way*(shot+16))
        for row in indices:
            support = labels[row[:way*shot]].reshape(way, shot)
            query = labels[row[way*shot:]].reshape(way, 16)
            assert len(np.unique(row)) == len(row) and len(np.unique(support[:, 0])) == way
            assert np.all(support == support[:, :1]) and np.all(query == support[:, :1])
        metadata.setdefault(key, {})
        metadata[key].update({"path": str(directory/f"{key}.npz"), "sha256": sha256(directory/f"{key}.npz"),
                              "split": split, "way": way, "shot": shot, "query": 16, "episodes": len(indices),
                              "mapping": str(FAMILY/"cache"/digest_json(cache["identity"])[:20]/"metadata.json"),
                              "mapping_sha256": cache["identity"]["mapping_sha256"][split],
                              "seed": settings["validation_seeds"].get(key)})
    write_json(run.path/"episode_manifest.json", metadata)
    return output


@torch.inference_mode()
def evaluate_candidates(config, features, indices, way, shot, candidates, run, label):
    device = features.device
    ys = torch.arange(way, device=device).repeat_interleave(shot)
    yq = torch.arange(way, device=device).repeat_interleave(16)
    results = [[] for _ in candidates]
    for number, row in enumerate(indices):
        z = features[row.astype(np.int64)]
        head = VisualEpisodeHead(config["model"]).fit(z[:way*shot], ys, way)
        query = z[way*shot:]
        for j, candidate in enumerate(candidates):
            method, params = candidate["method"], candidate["params"]
            scores = head.baseline(query) if method == "baseline" else getattr(head, method)(query, **params)
            results[j].append(100*(scores.argmax(1) == yq).sum().item()/len(yq))
        if (number+1) % 100 == 0:
            run.logger.info("%s %d/%d episodes, %d candidates", label, number+1, len(indices), len(candidates))
    return [{**candidate, **summary(values), "episode_accuracies": values}
            for candidate, values in zip(candidates, results)]


def candidate_key(candidate):
    return candidate["method"]+json.dumps(candidate["params"], sort_keys=True)


def save_search_table(path, search):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("shot", "phase", "way", "method", "beta", "ridge_lambda",
                                                   "residual_weight", "episodes", "accuracy", "confidence_interval"))
        writer.writeheader()
        for row in search:
            writer.writerow({key: row.get(key, row["params"].get(key)) for key in writer.fieldnames})


def validate(config, settings, run, cache, episodes):
    lock_path = run.path/"locked_parameters.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text())
    features = cache["splits"]["val"]["features"].to(config["runtime"]["device"])
    grid = [{"method": "proker", "params": {"beta": float(b), "ridge_lambda": float(l), "residual_weight": float(w)}}
            for b, l, w in itertools.product(settings["beta"], settings["ridge_lambda"], settings["residual_weight"])]
    ridges = [{"method": "ridge", "params": {"ridge_lambda": float(l)}} for l in settings["linear_ridge_lambda"]]
    search, selections = [], {}
    validation_results = {}
    for shot in (1, 5):
        candidates = [{"method": "baseline", "params": {}}]+grid+ridges
        coarse = {}
        for way in (5, 50):
            key = f"coarse_{way}way_{shot}shot"
            coarse[way] = evaluate_candidates(config, features, episodes[key], way, shot, candidates, run, key)
            validation_results[key] = coarse[way]
            search.extend({"shot": shot, "phase": "coarse", "way": way, **row} for row in coarse[way])
        if settings["local_expansion"]:
            best = max((row for row in coarse[50] if row["method"] == "proker"), key=lambda r: r["accuracy"])["params"]
            expansion = []
            known = {candidate_key(candidate) for candidate in candidates}
            for param, values in (("beta", (best["beta"]*.5, best["beta"]*2)),
                                  ("ridge_lambda", (best["ridge_lambda"]*.5, best["ridge_lambda"]*2)),
                                  ("residual_weight", (.25, 1.5))):
                for value in values:
                    candidate = {"method": "proker", "params": {**best, param: value}}
                    if candidate_key(candidate) not in known:
                        expansion.append(candidate)
                        known.add(candidate_key(candidate))
            assert len(grid)+len(expansion) <= 24
            for way in (5, 50):
                key = f"coarse_{way}way_{shot}shot"
                extra = evaluate_candidates(config, features, episodes[key], way, shot, expansion, run, key+"_local")
                coarse[way].extend(extra)
                search.extend({"shot": shot, "phase": "local", "way": way, **row} for row in extra)
        top = [{"method": "baseline", "params": {}}]
        for method in ("proker", "ridge"):
            # Stable tie-breaking uses grid order, never test data or auxiliary 5-way scores.
            ranking = sorted((row for row in coarse[50] if row["method"] == method),
                             key=lambda row: -row["accuracy"])
            top.extend({"method": row["method"], "params": row["params"]} for row in ranking[:2])
        key = f"confirm_50way_{shot}shot"
        confirmed = evaluate_candidates(config, features, episodes[key], 50, shot, top, run, key)
        validation_results[key] = confirmed
        search.extend({"shot": shot, "phase": "confirm", "way": 50, **row} for row in confirmed)
        selections[str(shot)] = {method: max((row for row in confirmed if row["method"] == method),
                                            key=lambda row: row["accuracy"])["params"]
                                 for method in ("proker", "ridge")}
        selections[str(shot)]["proker_candidates_tested"] = len(coarse[50])-1-len(ridges)
    write_json(run.path/"validation_results.json", validation_results)
    save_search_table(run.path/"search_table.csv", search)
    lock = {"status": "locked_before_test", "selected_at_unix": time.time(), "by_shot": selections,
            "selection": "coarse 50-way top two per method -> independent 1000-episode 50-way confirmation; stable grid tie-break",
            "test_labels_used": False, "same_parameters_for_5way_and_50way": True,
            "validation_episode_manifest_sha256": sha256(run.path/"episode_manifest.json")}
    write_json(lock_path, lock)
    run.logger.info("parameters locked: %s", selections)
    del features
    torch.cuda.empty_cache()
    return lock


@torch.inference_mode()
def formal(config, settings, run, cache, episodes, lock):
    # A durable lock MUST exist before any formal score is measured.
    assert (run.path/"locked_parameters.json").exists()
    features = cache["splits"]["test"]["features"].to(config["runtime"]["device"])
    results = {}
    for way, shot in itertools.product((5, 50), (1, 5)):
        key = f"{way}way_{shot}shot"
        path = run.path/f"test_{key}.json"
        if path.exists():
            results[key] = json.loads(path.read_text())
            continue
        device = features.device
        ys = torch.arange(way, device=device).repeat_interleave(shot)
        yq = torch.arange(way, device=device).repeat_interleave(16)
        params = lock["by_shot"][str(shot)]
        values = {method: [] for method in ("baseline", "ridge", "proker")}
        times = {method: [] for method in values}
        peak = {method: 0 for method in values}
        first = features[episodes[f"test_{key}"][0].astype(np.int64)]
        # Warm up all actual sizes; no labels or accuracy used for warmup.
        warm = VisualEpisodeHead(config["model"]).fit(first[:way*shot], ys, way)
        for _ in range(3):
            warm.baseline(first[way*shot:])
            warm.ridge(first[way*shot:], **params["ridge"])
            warm.proker(first[way*shot:], **params["proker"])
        sync(device)
        del warm, first
        for number, row in enumerate(episodes[f"test_{key}"]):
            z = features[row.astype(np.int64)]
            for method in values:
                torch.cuda.reset_peak_memory_stats(device)
                sync(device)
                start = time.perf_counter()
                head = VisualEpisodeHead(config["model"]).fit(z[:way*shot], ys, way)
                scores = head.baseline(z[way*shot:]) if method == "baseline" else getattr(head, method)(z[way*shot:], **params[method])
                sync(device)
                times[method].append((time.perf_counter()-start)*1000)
                peak[method] = max(peak[method], torch.cuda.max_memory_allocated(device))
                values[method].append(100*(scores.argmax(1) == yq).sum().item()/len(yq))
                del head, scores
            if (number+1) % 100 == 0:
                run.logger.info("formal %s %d/%d", key, number+1, len(episodes[f"test_{key}"]))
                write_json(run.path/"evaluation_progress.json", {"status": "running", "setting": key, "episodes": number+1})
        historical = json.loads(Path(settings["historical_results"][key]).read_text())
        historical_delta = np.asarray(values["baseline"])-np.asarray(historical["episode_accuracies"])
        results[key] = {"way": way, "shot": shot, "query": 16, "status": "complete",
                        "episode_indices": f"episodes/test_{key}.npz", "lock_sha256": sha256(run.path/"locked_parameters.json"),
                        "historical_baseline_replay": {"different_episode_accuracies": int(np.count_nonzero(historical_delta)),
                                                        "max_absolute_difference_pp": float(np.abs(historical_delta).max()),
                                                        "mean_difference_pp": float(historical_delta.mean())}, "methods": {}}
        for method in values:
            differences = np.asarray(values[method])-np.asarray(values["baseline"])
            paired = summary(differences)
            results[key]["methods"][method] = {**summary(values[method]), "episode_accuracies": values[method],
               "gain_pp": paired["accuracy"], "paired_ci_half_width_pp": paired["confidence_interval"],
               "head_ms_per_episode": float(np.mean(times[method])),
               "head_ms_p50": float(np.median(times[method])), "head_ms_p95": float(np.percentile(times[method], 95)),
               "head_total_seconds": float(np.sum(times[method])/1000), "peak_gpu_bytes": peak[method],
               "head_timing": "synchronized wall time including support fit + query scores, excluding encoding/gather/argmax",
               "parameters": {} if method == "baseline" else params[method]}
        write_json(path, results[key])
    write_json(run.path/"results.json", results)
    write_json(run.path/"evaluation_progress.json", {"status": "complete", "groups": 4, "episodes_per_group": 2000})
    return results


def report(run, results, lock, wall_seconds):
    reference = json.loads((run.path/"cache_reference.json").read_text())
    many = [results[f"50way_{shot}shot"]["methods"]["proker"] for shot in (1, 5)]
    few = [results[f"5way_{shot}shot"]["methods"]["proker"] for shot in (1, 5)]
    # Predeclared engineering rule; it assesses the outcome and does not trigger further search.
    if all(row["gain_pp"]+row["paired_ci_half_width_pp"] <= 0 for row in many):
        decision = "stop"
    elif all(row["gain_pp"]-row["paired_ci_half_width_pp"] > 0 for row in many) and min(row["gain_pp"] for row in few) >= -.5:
        decision = "continue"
    else:
        decision = "uncertain"
    lines = ["# SCEG CUB 纯视觉 ProKeR 迁移实验", "", f"状态：代码与必要测试完成；验证选参完成；四组正式评测完成。工程判断：**{decision}**。", "",
             "本轮使用已有 seed42、epoch480 的 SCEG best，全部参数冻结；无重训练、文本、CLIP、外部预训练或测试增强。", "",
             "## 结果", "", "Accuracy (%) ± 原版95% episode CI半宽；增益与成对CI单位为百分点。每组2000 episodes，query=16。", "",
             "|Setting|Method|Accuracy ± CI|Gain ± paired CI|Head ms/episode|Peak GPU MiB|",
             "|---|---|---:|---:|---:|---:|"]
    for key, setting in results.items():
        for method, row in setting["methods"].items():
            lines.append(f"|{key}|{method}|{row['accuracy']:.4f} ± {row['confidence_interval']:.4f}|{row['gain_pp']:+.4f} ± {row['paired_ci_half_width_pp']:.4f}|{row['head_ms_per_episode']:.3f}|{row['peak_gpu_bytes']/2**20:.1f}|")
    lines += ["", "## 方法与来源", "",
              "作者 [ProKeR CVPR2025论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Bendou_ProKeR_A_Kernel_Perspective_on_Few-Shot_Adaptation_of_Large_Vision-Language_CVPR_2025_paper.pdf) 与 [补充](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Bendou_ProKeR_A_Kernel_CVPR_2025_supplemental.pdf) 使用冻结CLIP视觉特征及文本先验，11个基准、1/2/4/8/16-shot；不是本实验的CUB episodic协议。", "",
              "实际读取代码commit `f2ecb3d2a0d7e0993f977e550286b34f06015edb`：[trainer](https://github.com/ybendou/ProKeR/blob/f2ecb3d2a0d7e0993f977e550286b34f06015edb/trainers/proker.py)、[RN50配置](https://github.com/ybendou/ProKeR/blob/f2ecb3d2a0d7e0993f977e550286b34f06015edb/configs/RN50/configs_proker/hyperparameters.json)。其1-shot参数beta=3、lambda=.05；没有作者CUB/5-shot参数。补充逐页视觉图表未另复核，公式与发布代码已核查。", "",
              "本次迁移：每图SCEG三路FC后未归一化z2/z3/z4，各640维；支持特征先按shot平均，再用原版 normalize_episode。f0为原三路scores/3（原预测不变，但残差拟合尺度改变），f0(S)来自当前支持集原型，不用训练基类class_representative。", "",
              "核输入u=concat(F.normalize(z2),F.normalize(z3),F.normalize(z4))/sqrt(3)，K=exp[-beta(1-u·u')]。作者求解 A=solve(I+KSS/lambda,Y-f0(S))，预测 f0(q)+w KqS A。若写成(K+lambda I)^-1，预测必须乘额外lambda，不能静默替换。w是迁移新增参数，不是论文默认参数。", "",
              "线性ridge在同一u上拟合one-hot、无截距：W=U_S^T solve(U_SU_S^T+lambda I,Y)，不加f0。纯归纳：query间不交互，不读取query标签或整体统计构造特征。", "",
              "## 选参与隔离", "",
              "每shot先18组beta={1,5,10}、lambda={.1,1,10}、w={.5,1}，固定200个5-way和200个50-way验证episodes。对50-way粗筛最佳仅做一次6个单轴邻域扩展：beta×{.5,2}、lambda×{.5,2}、w={.25,1.5}，跳过重复，合计最多24组。线性ridge独立搜索lambda={.001,.01,.1,1,10,100}。", "",
              "各头按50-way粗筛取前两名，在另一固定1000个50-way验证episodes上确认；5-way仅辅助记录。同shot两种way共用参数，稳定网格顺序破平分。完整搜索表见search_table.csv，验证逐episode准确率见validation_results.json，锁定参数见locked_parameters.json。测试后不追加搜索。", "",
              "锁定参数：", "", "```json", json.dumps(lock["by_shot"], indent=2), "```", "",
              "独立worktree/分支 experiment/sceg_proker，全部新代码/配置/缓存/结果位于experiments/sceg_proker；历史sceg_original只读，主工作区docs/paper/保留。完整历史索引按路径映射重放，并在此实验重新评价三种方法。", "",
              "## 验证与统计限制", "",
              "必要测试检查作者同输入公式、w=0退化、支持排列不变、query单独/分块一致、线性ridge原/对偶等价和正参数。另执行一次既有cosine代表性回归。6个真实episode在线/缓存scores与预测核验见online_cache_regression.json；BN统计保持不变。", "",
              "原版95% CI半宽=1.96 sqrt(var(ddof=0)/n)；成对CI对同episode百分点差值使用同一公式。只反映固定checkpoint与数据下episode抽样不确定性，不代表多训练seed稳定性；episode复用图像使其并非独立训练重复。", "",
              "历史逐episodebaseline核对：", ""]
    for key, setting in results.items():
        lines.append(f"- {key}: {json.dumps(setting['historical_baseline_replay'])}")
    lines += ["", "## 资源、命令与记录", "",
              f"唯一特征缓存：`{reference['path']}`；验证2927图、测试2958图，每图3×640 float32，含路径映射，不存空间图/episode核/新权重。编码总耗时{reference['encoding_seconds']:.3f}s，编码峰值显存{reference['encoding_peak_gpu_bytes']/2**20:.1f}MiB。本次入口总墙钟{wall_seconds:.3f}s，含验证搜索与正式评测；细分见stage_times.json。", "",
              "头耗时是单GPU同步墙钟，包括支持拟合和query打分，不含编码、cache gather、argmax；统一wrapper支持准备包含三路原型与u，因此baseline计时亦含少量通用准备开销。显存为torch峰值allocated，含常驻缓存、不含其他进程和CUDA驱动保留内存；GPU可同时受外部负载影响。", "",
              f"输入checkpoint：`{reference['input_checkpoint']}`，SHA256 `{reference['input_checkpoint_sha256']}`。已有数据清单直接引用cache_reference.json，未重复哈希全部图片。", "",
              "实际命令见command.txt与invocations.jsonl，环境见environment.json，基础代码版本/差异见git_commit.txt/git_diff.patch/git_status.txt。新实验未跟踪代码保存在source_snapshot/及experiment_source_manifest.json，不能仅用空git diff追溯。", "",
              "```bash", (run.path/"command.txt").read_text().strip(), "```", "",
              "可续跑：同一命令重用run_id并添加 --resume；已有cache、lock及完成的正式组直接复用，中断组按固定索引重测。没有训练恢复权重，因为本轮不训练。", "",
              "原始episode索引位于episodes/*.npz，路径映射引用cache metadata；各test_*.json保存三种方法的逐episode准确率、配对指标和计时。不按episode保存模型或核矩阵。", "",
              "工程规则：两个many-way成对CI上界均≤0则stop；两个下界均>0且few-way最差退化≤.5pp则continue；其他uncertain。它只决定后续工程判断，不触发本轮额外实验。最终判断应结合上表收益、few-way退化和头开销；完整结果可为负，不增加其他模块。", "",
              "未完成项：无本轮要求内的指标；多训练seed及作者原CLIP实验不在范围内。无自动push/merge。"]
    (run.path/"report.md").write_text("\n".join(lines)+"\n")
    return decision


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=FAMILY/"config.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("all", "prepare", "validate", "test"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    config = load_config(settings["baseline_config"])
    config["experiment"] = "sceg_proker"
    config["data"]["root"] = settings["data_root"]
    config["proker"] = settings
    config["purpose"] = "frozen_visual_inductive_inference"
    run_path = FAMILY/"runs"/args.run_id
    if args.resume:
        previous = yaml.safe_load((run_path/"config.yaml").read_text())
        if previous != config:
            raise ValueError("Resolved configuration changed since this run; use a fresh run_id")
        # Reuse the native record class without creating/overwriting the exclusive run.
        import logging
        run = ExperimentRun.__new__(ExperimentRun)
        run.root, run.path = ROOT, run_path
        run.metrics = json.loads((run_path/"metrics.json").read_text())
        run.logger = logging.getLogger("sceg_proker.resume")
        run.logger.setLevel(logging.INFO)
        for handler in (logging.FileHandler(run_path/"train.log"), logging.StreamHandler()):
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            run.logger.addHandler(handler)
    else:
        run = ExperimentRun(config, args.run_id)
    snapshot(run.path)
    with (run.path/"invocations.jsonl").open("a") as stream:
        stream.write(json.dumps({"argv": getattr(sys, "orig_argv", sys.argv), "time": time.time(), "stage": args.stage})+"\n")
    seed_everything(config["seed"], deterministic=config["runtime"]["deterministic"],
                    warn_only=config["runtime"]["deterministic_warn_only"], allow_tf32=False)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    start = time.perf_counter()
    stage_times = json.loads((run.path/"stage_times.json").read_text()) if (run.path/"stage_times.json").exists() else {}
    try:
        run.update_metrics(status="running", accuracy=None, confidence_interval=None,
                           metric_scope="frozen visual CUB inference, existing seed42 checkpoint")
        timer = time.perf_counter()
        cache, datasets = cache_features(config, settings, run)
        episodes = manifests(config, settings, run, datasets, cache)
        stage_times["prepare_seconds"] = time.perf_counter()-timer
        if args.stage == "prepare":
            run.update_metrics(status="prepared")
            return
        timer = time.perf_counter()
        if args.stage == "test" and not (run.path/"locked_parameters.json").exists():
            raise ValueError("Run validation and lock parameters before test")
        lock = validate(config, settings, run, cache, episodes)
        stage_times["validation_seconds"] = time.perf_counter()-timer
        if args.stage == "validate":
            run.update_metrics(status="validation_complete")
            return
        timer = time.perf_counter()
        results = formal(config, settings, run, cache, episodes, lock)
        stage_times["formal_seconds"] = time.perf_counter()-timer
        decision = report(run, results, lock, time.perf_counter()-start)
        run.update_metrics(status="completed", engineering_decision=decision,
                           evaluation={key: {"way": value["way"], "shot": value["shot"],
                                       "methods": {m: {k: v for k, v in row.items() if k != "episode_accuracies"}
                                                   for m, row in value["methods"].items()}}
                                       for key, value in results.items()})
    except BaseException as error:
        run.update_metrics(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(run.path/"stage_times.json", stage_times)
        run.publish_result()


if __name__ == "__main__":
    main()
