"""Replay Dogs author episodes with a frozen model and CUB-locked inference heads.

Run from this worktree with the existing SCEG Python environment. A single CPU
feature cache supports interruption recovery; completed settings are immutable.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from datasets import make_dataset
from models import SCEG
from utils.config import ModelConfig, load_config
from utils.reproducibility import seed_everything
from .head import VisualEpisodeHead

ROOT = Path(__file__).resolve().parents[2]
FAMILY = Path(__file__).resolve().parent
METHODS = ('baseline', 'ridge', 'proker')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def tree_identity(path):
    files = sorted(p for p in path.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and p.suffix != '.pyc')
    return {'files': len(files), 'sha256': digest(
        [(str(p.relative_to(path)), sha256(p)) for p in files])}


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {'accuracy_pct': float(values.mean()),
            'ci95_halfwidth_pp': float(1.96 * values.std(ddof=0) / np.sqrt(len(values)))}


def summarize(rows):
    result = {}
    for method in METHODS:
        values = np.array([float(r[method + '_accuracy_pct']) for r in rows])
        result[method] = statistics(values)
        result[method]['head_mean_ms'] = float(np.mean(
            [float(r[method + '_head_ms']) for r in rows]))
        result[method]['peak_allocated_bytes'] = max(int(r[method + '_peak_bytes']) for r in rows)
        if method != 'baseline':
            delta = values - np.array([float(r['baseline_accuracy_pct']) for r in rows])
            paired = statistics(delta)
            result[method].update(gain_pp=paired['accuracy_pct'],
                                 paired_ci95_halfwidth_pp=paired['ci95_halfwidth_pp'])
    return result


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def resource_check():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                                   '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                        '--format=csv,noheader'], text=True).strip()
    if processes and 'No running' not in processes:
        raise RuntimeError('Existing GPU compute processes: ' + processes)
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0]) * 1024
    disk = shutil.disk_usage(ROOT).free
    if available < 1024**3 or disk < 1024**3:
        raise RuntimeError('Less than 1 GiB available RAM or disk')
    return {'gpu_before': gpu, 'compute_processes_before': processes,
            'available_ram_bytes': available, 'free_disk_bytes': disk}


def inputs(settings):
    dogs = Path(settings['dogs_root'])
    original_run = dogs / settings['dogs_run']
    config = load_config(original_run / 'config.yaml')
    checkpoint = original_run / 'checkpoint.pth'
    assert sha256(checkpoint) == settings['checkpoint_sha256'], 'Checkpoint identity changed'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == settings['dogs_commit']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=dogs, text=True).strip() == settings['dogs_commit']
    proker = Path(settings['proker_root'])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=proker, text=True).strip() == settings['proker_commit']
    original_head = proker / 'experiments/sceg_proker/head.py'
    assert sha256(original_head) == sha256(FAMILY / 'head.py') == settings['proker_head_sha256']
    lock_path = proker / settings['parameter_lock']
    assert sha256(lock_path) == settings['parameter_lock_sha256']
    lock = json.loads(lock_path.read_text())
    assert not lock['test_labels_used']
    for shot, params in settings['parameters'].items():
        assert all(params[m] == lock['by_shot'][str(shot)][m] for m in ('ridge', 'proker'))
    assert config['runtime']['amp'] is False and config['runtime']['allow_tf32'] is False
    assert config['runtime']['deterministic'] and config['seed'] == 42
    hashes = {str(checkpoint): sha256(checkpoint), str(lock_path): sha256(lock_path)}
    for name in ('config.yaml', 'REPORT.md', 'execution_sources.json', 'final_checks.json', 'data_check.json'):
        path = original_run / name
        hashes[str(path)] = sha256(path)
    sources = json.loads((original_run / 'execution_sources.json').read_text())
    for name, expected in sources.items():
        assert sha256(ROOT / name) == sha256(dogs / name) == expected, 'Dogs execution source changed: ' + name
    data_check = json.loads((original_run / 'data_check.json').read_text())
    for reference in data_check['references'].values():
        path = Path(reference['path'])
        assert sha256(path) == reference['sha256'], 'Data manifest changed: ' + str(path)
        hashes[str(path)] = reference['sha256']
    prep = dogs / 'experiments/sceg_dogs_author/prep'
    split = json.loads((prep / 'split.json').read_text())
    sets = [set(split['classes'][s]) for s in ('train', 'val', 'test')]
    assert all(not sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    dataset = make_dataset(config, 'test')
    paths = [str(Path(p).relative_to(dataset.root)) for p, _ in dataset.samples]
    assert len(paths) == 5378 and len(dataset.classes) == 30
    assert dataset.class_to_idx == split['class_to_idx']['test']
    sample_hashes = []
    with (prep / 'samples.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['split'] != 'test':
                continue
            for kind in ('raw', 'pre'):
                path = Path(config['data']['root']) / row[kind + '_path']
                assert sha256(path) == row[kind + '_sha256'], 'Test image changed: ' + str(path)
            sample_hashes.append((row['pre_path'], row['pre_sha256']))
    assert sorted(p for p, _ in sample_hashes) == sorted('test_pre/' + p for p in paths)
    records = {}
    final = json.loads((original_run / 'final_checks.json').read_text())
    for way, shot in settings['settings']:
        key = f'{way}way_{shot}shot'
        source = original_run / f'test_{key}.json'
        hashes[str(source)] = sha256(source)
        assert hashes[str(source)] == final['raw_result_sha256'][source.name]
        original = json.loads(source.read_text())
        assert original['sample_index_paths'] == paths
        assert original['checkpoint_sha256'] == settings['checkpoint_sha256']
        assert original['split_sha256'] == config['data_protocol']['split_manifest_sha256']
        # Dogs records hash the parsed configuration, independently of YAML formatting.
        assert original['config_sha256'] == digest(config)
        assert (original['way'], original['shot'], original['query'], original['episodes']) == (way, shot, 16, 2000)
        indices = np.asarray(original['episode_indices'], dtype=np.int64)
        assert indices.shape == (2000, way * (shot + 16))
        assert indices.min() >= 0 and indices.max() < len(dataset)
        sample_labels = np.asarray(dataset.targets)[indices]
        support = sample_labels[:, :way*shot].reshape(2000, way, shot)
        query = sample_labels[:, way*shot:].reshape(2000, way, 16)
        assert np.all(support == support[:, :, :1]) and np.all(query == support[:, :, :1])
        for n, episode in enumerate(indices):
            assert len(np.unique(episode)) == len(episode), 'Support/query overlap'
            classes = [dataset.classes[c] for c in support[n, :, 0]]
            assert len(set(classes)) == way and classes == original['episode_classes'][n]
        values = np.asarray(original['episode_accuracies'], dtype=np.float64)
        assert len(values) == 2000
        summary = statistics(values)
        assert abs(summary['accuracy_pct'] - original['accuracy']) < 1e-10
        assert abs(summary['ci95_halfwidth_pp'] - original['confidence_interval']) < 1e-10
        records[key] = {'way': way, 'shot': shot, 'indices': indices, 'historical': values,
                        'source': str(source), 'source_sha256': hashes[str(source)]}
    identity = {'input_files_sha256': hashes, 'test_png_inventory_sha256': digest(sorted(sample_hashes)),
                'sample_mapping_sha256': digest(paths), 'dogs_execution_sources': sources,
                'experiment_sources': {p.name: sha256(p) for p in FAMILY.iterdir()
                                       if p.suffix in ('.py', '.yaml')}, 'settings': settings}
    # Canonical JSON keys make identity equality stable after restart (YAML shot keys are integers).
    return config, dataset, records, json.loads(json.dumps(identity))


def original_scores(features, classifier, way, shot):
    scores = [classifier(features[:, stage], way, shot) for stage in range(3)]
    return scores[1] + scores[2] + scores[0]


def accuracy(scores, targets):
    return 100 * (scores.argmax(1) == targets).sum().item() / len(targets)


@torch.inference_mode()
def cache_features(model, dataset, settings, run, metadata, logger):
    path = run / 'feature_cache.npy'
    if path.exists():
        assert sha256(path) == metadata['cache']['file_sha256'], 'Cache content changed'
        features = torch.from_numpy(np.load(path))
        logger.info('reusing one CPU feature cache, %.2f MiB', features.numel()*4/1024**2)
        return features
    features = torch.empty((len(dataset), 3, 640), dtype=torch.float32)
    loader = DataLoader(dataset, batch_size=settings['encoding_batch_size'], shuffle=False,
                        num_workers=settings['workers'], pin_memory=False)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    offset = 0
    for number, (images, _) in enumerate(loader):
        batch = torch.stack(model.encode(images.cuda()), dim=1).cpu()
        features[offset:offset+len(batch)] = batch
        offset += len(batch)
        if number % 20 == 0:
            logger.info('encoded %d/%d test images', offset, len(dataset))
    assert offset == len(dataset) and torch.isfinite(features).all()
    temporary = path.with_suffix('.npy.part')
    with temporary.open('wb') as stream:
        np.save(stream, features.numpy())
    temporary.replace(path)
    metadata['cache'] = {'file': str(path), 'file_sha256': sha256(path),
                         'shape': list(features.shape), 'dtype': 'float32', 'cpu_bytes': features.numel()*4,
                         'encoding_seconds': time.perf_counter()-start,
                         'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                         'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    write_json(run / 'metadata.json', metadata)
    return features


@torch.inference_mode()
def online_check(model, dataset, features, record, number, config, settings):
    way, shot = record['way'], record['shot']
    images = torch.stack([dataset[int(i)][0] for i in record['indices'][number]])
    online = torch.stack(model.encode(images.cuda()), dim=1)
    cached = features[record['indices'][number]].cuda()
    torch.testing.assert_close(cached, online, atol=1e-5, rtol=1e-5)
    direct = model.classify(tuple(online[:, s] for s in range(3)), way, shot).scores
    cached_scores = original_scores(cached, model.classifier, way, shot)
    torch.testing.assert_close(cached_scores/3, direct/3, atol=settings['float32_score_atol'], rtol=1e-5)
    labels = torch.arange(way, device='cuda').repeat_interleave(shot)
    targets = torch.arange(way, device='cuda').repeat_interleave(16)
    head = VisualEpisodeHead(config['model']).fit(cached[:way*shot], labels, way)
    f0 = head.baseline(cached[way*shot:])
    torch.testing.assert_close(f0, cached_scores/3, atol=1e-6, rtol=1e-6)
    assert torch.equal(f0.argmax(1), cached_scores.argmax(1)), 'Head/original cached predictions differ'
    zero = head.proker(cached[way*shot:], **(settings['parameters'][shot]['proker'] | {'residual_weight': 0.0}))
    assert torch.equal(zero, f0)
    assert accuracy(direct, targets) == record['historical'][number], 'Original online/history mismatch'
    flips = torch.where(cached_scores.argmax(1) != direct.argmax(1))[0]
    details = []
    for q in flips.tolist():
        top = cached_scores[q].topk(2)
        margin = float(top.values[0] - top.values[1]) / 3
        assert margin <= settings['float32_score_atol'], 'Prediction difference is not a float32 boundary'
        details.append({'query_position': q, 'cached_class': int(cached_scores[q].argmax()),
                        'online_class': int(direct[q].argmax()), 'cached_top2_margin': margin})
    return {'episode': number, 'max_feature_error': float((cached-online).abs().max()),
            'max_score_error': float(((cached_scores-direct)/3).abs().max()),
            'cached_accuracy_pct': accuracy(cached_scores, targets),
            'online_accuracy_pct': accuracy(direct, targets), 'prediction_flips': details,
            'zero_residual_exact': True, 'head_original_predictions_equal': True}


@torch.inference_mode()
def replay(model, dataset, features, records, config, settings, run, metadata, logger):
    checked = {}
    for key, record in records.items():
        checked[key] = [online_check(model, dataset, features, record, i, config, settings)
                        for i in range(settings['online_check_episodes_per_setting'])]
    baseline = {}
    total_mismatches = 0
    for key, record in records.items():
        way, shot = record['way'], record['shot']
        targets = torch.arange(way, device='cuda').repeat_interleave(16)
        labels = torch.arange(way, device='cuda').repeat_interleave(shot)
        values = []
        for indices in record['indices']:
            episode = features[indices].cuda()
            scores = original_scores(episode, model.classifier, way, shot)
            head = VisualEpisodeHead(config['model']).fit(episode[:way*shot], labels, way)
            assert torch.equal(head.baseline(episode[way*shot:]).argmax(1), scores.argmax(1))
            values.append(accuracy(scores, targets))
        values = np.asarray(values)
        mismatches = np.flatnonzero(values != record['historical'])
        total_mismatches += len(mismatches)
        assert total_mismatches <= settings['max_boundary_episodes'], 'Too many historical mismatches'
        diagnostics = [online_check(model, dataset, features, record, int(i), config, settings) for i in mismatches]
        baseline[key] = values
        checked[key] = {'real_online_regressions': checked[key], 'episodes_compared': len(values),
                        'matching_episodes': len(values)-len(mismatches),
                        'mismatch_diagnostics': diagnostics,
                        'mean_delta_from_history_pp': float((values-record['historical']).mean()),
                        'all_cached_head_predictions_match_original': True}
        logger.info('historical replay %s: %d/2000 episode accuracies match', key, 2000-len(mismatches))
    metadata['validation'] = {'status': 'passed', 'support_query_overlap': False,
                              'class_mapping_verified': True, 'query_labels_used_for_fitting': False,
                              'historical_predictions_available': False, 'settings': checked}
    # Four columns only: original episode indices continue to live in the Dogs run.
    with (run / 'baseline_replay.csv.part').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['setting', 'episode', 'accuracy_pct', 'historical_accuracy_pct'])
        for key, values in baseline.items():
            for i, value in enumerate(values):
                writer.writerow([key, i, value, records[key]['historical'][i]])
    (run / 'baseline_replay.csv.part').replace(run / 'baseline_replay.csv')
    write_json(run / 'metadata.json', metadata)
    return baseline


@torch.inference_mode()
def formal(features, classifier, records, baseline, config, settings, run, metadata, logger):
    results = {}
    for key, record in records.items():
        output = run / f'episode_metrics_{key}.csv'
        if output.exists():
            rows = read_rows(output)
            assert len(rows) == 2000 and [int(r['episode']) for r in rows] == list(range(2000))
            assert np.array_equal([float(r['baseline_accuracy_pct']) for r in rows], baseline[key])
            logger.info('reusing completed setting %s', key)
        else:
            way, shot = record['way'], record['shot']
            labels = torch.arange(way, device='cuda').repeat_interleave(shot)
            targets = torch.arange(way, device='cuda').repeat_interleave(16)
            params = settings['parameters'][shot]
            # One real task warmup per head, excluded from timing and statistics.
            warm = features[record['indices'][0]].cuda()
            original_scores(warm, classifier, way, shot)
            head = VisualEpisodeHead(config['model']).fit(warm[:way*shot], labels, way)
            head.ridge(warm[way*shot:], **params['ridge'])
            head.proker(warm[way*shot:], **params['proker'])
            del head, warm
            rows = []
            for i, indices in enumerate(record['indices']):
                episode = features[indices].cuda()
                row = {'episode': i}
                # Rotate measurement order to avoid a fixed first/last-method bias.
                for method in METHODS[i % 3:] + METHODS[:i % 3]:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    if method == 'baseline':
                        scores = original_scores(episode, classifier, way, shot)
                    else:
                        head = VisualEpisodeHead(config['model']).fit(episode[:way*shot], labels, way)
                        scores = getattr(head, method)(episode[way*shot:], **params[method])
                    predictions = scores.argmax(1)
                    torch.cuda.synchronize()
                    row[method + '_head_ms'] = (time.perf_counter()-start)*1000
                    row[method + '_peak_bytes'] = torch.cuda.max_memory_allocated()
                    row[method + '_accuracy_pct'] = 100 * (predictions == targets).sum().item()/len(targets)
                    assert torch.isfinite(scores).all()
                    del scores, predictions
                    if method != 'baseline':
                        del head
                assert row['baseline_accuracy_pct'] == baseline[key][i], 'Formal/replay baseline changed'
                rows.append(row)
                if (i+1) % 500 == 0:
                    logger.info('formal %s: %d/2000 episodes', key, i+1)
            temporary = output.with_suffix('.csv.part')
            with temporary.open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(output)
        results[key] = {'way': record['way'], 'shot': record['shot'], 'query': 16, 'episodes': len(rows),
                        'metrics_file': output.name, 'metrics_sha256': sha256(output),
                        'historical_source': record['source'], 'historical_sha256': record['source_sha256'],
                        'methods': summarize(rows)}
        write_json(run / 'results.json', {'status': 'running', 'settings': results})
    return results


def audit(run, records, results):
    replay_rows = read_rows(run / 'baseline_replay.csv')
    assert len(replay_rows) == 8000
    for key, result in results.items():
        path = run / result['metrics_file']
        assert sha256(path) == result['metrics_sha256']
        rows = read_rows(path)
        assert len(rows) == 2000 and [int(r['episode']) for r in rows] == list(range(2000))
        assert summarize(rows) == result['methods']
        replay_values = [float(r['accuracy_pct']) for r in replay_rows if r['setting'] == key]
        assert np.array_equal(replay_values, [float(r['baseline_accuracy_pct']) for r in rows])
        history = [float(r['historical_accuracy_pct']) for r in replay_rows if r['setting'] == key]
        assert np.array_equal(history, records[key]['historical'])
    return {'status': 'passed', 'episodes': 8000, 'accuracy_ci_paired_ci_recomputed': True,
            'formal_replay_baseline_equal': True, 'method_metrics_file_hashes_verified': True}


def report(run, results, metadata):
    lines = ['# SCEG + ProKeR Stanford Dogs 跨数据集验证', '', '状态：**completed**。固定 seed42 / epoch520 Dogs 模型，CUB 验证锁定参数直接迁移。', '',
             '## 目的与实现', '',
             '验证纯视觉 ProKeR 的跨数据集收益，不训练、不搜索 Dogs 参数。独立 worktree 基于 Dogs author 分支。仅原样复制 CUB 的 72 行 head.py；其余使用原 SCEG encode、CosineClassifier、ImageFolder 和预处理。', '',
             '核输入 u=concat(normalize(z2),normalize(z3),normalize(z4))/sqrt(3)，K=exp[-beta(1-u vᵀ)]；A=(I+Kss/lambda)^(-1)(Y-f0(S))，f(Q)=f0(Q)+w Kqs A。f0 为原三路 cosine 的平均值，原型先对未归一化 support 求均值。Baseline 使用原 stage3+stage4+stage2 分数和。Ridge 为同一 u 上无截距的线性岭回归。', '',
             '1-shot ProKeR=(beta10,lambda1,w1)，Ridge lambda100；5-shot ProKeR=(beta1,lambda0.1,w1.5)，Ridge lambda0.1。两种 way 共享相同 shot 参数，来自同一 CUB lock。', '',
             '## 正式结果', '', '| 设置 | Baseline % ± CI pp | Ridge % ± CI pp | Ridge Δ ± paired CI pp | ProKeR % ± CI pp | ProKeR Δ ± paired CI pp |',
             '|---|---:|---:|---:|---:|---:|']
    for key, result in results.items():
        b, r, p = (result['methods'][m] for m in METHODS)
        lines.append(f"| {key} | {b['accuracy_pct']:.4f} ± {b['ci95_halfwidth_pp']:.4f} | {r['accuracy_pct']:.4f} ± {r['ci95_halfwidth_pp']:.4f} | {r['gain_pp']:+.4f} ± {r['paired_ci95_halfwidth_pp']:.4f} | {p['accuracy_pct']:.4f} ± {p['ci95_halfwidth_pp']:.4f} | {p['gain_pp']:+.4f} ± {p['paired_ci95_halfwidth_pp']:.4f} |")
    lines += ['', '每组2000 episodes，每类16 query。CI 半宽=1.96×std(ddof=0)/sqrt(n)；paired CI 对逐 episode 方法准确率减 Baseline 的差值计算。准确率记录为百分数，增益和半宽为百分点。', '',
              '## 耗时与资源', '', '| 设置 | Baseline ms/episode | Ridge ms/episode | ProKeR ms/episode | 各头峰值 allocated MiB (B/R/P) |', '|---|---:|---:|---:|---:|']
    for key, result in results.items():
        values = [result['methods'][m] for m in METHODS]
        lines.append('| '+key+' | '+' | '.join(f"{v['head_mean_ms']:.3f}" for v in values)+' | '+
                     '/'.join(f"{v['peak_allocated_bytes']/1024**2:.2f}" for v in values)+' |')
    cache = metadata['cache']
    lines += ['', 'CUDA 同步墙钟计时，包括每头 support 拟合和 query argmax，排除 CPU→GPU 传输、准确率统计及 Backbone。方法顺序逐 episode 轮转，每头一集 warmup。Ridge/ProKeR 使用原复用 head.fit（含原型、核输入及 one-hot 建立），Baseline 直接复用原 cosine 分类器；这些是实际实现的端到端分类头开销，不是核求解器单独耗时。', '',
              f"单份 CPU/disk FP32 三路缓存 {cache['cpu_bytes']/1024**2:.2f} MiB，编码 batch48 / workers2 / CPU threads2。编码 {cache['encoding_seconds']:.2f} s，峰值 allocated {cache['peak_allocated_bytes']/1024**2:.2f} MiB / reserved {cache['peak_reserved_bytes']/1024**2:.2f} MiB。特征按 episode 送 GPU，不保存 episode 特征副本、核矩阵或 logits。在线一致性检查的少量重编码单独进行。", '',
              '## 历史重放与验证', '']
    for key, check in metadata['validation']['settings'].items():
        lines.append(f"- {key}：{check['matching_episodes']}/2000 历史 episode 准确率一致，均值差 {check['mean_delta_from_history_pp']:+.8f} pp；差异定位记录见 metadata.json 的 validation.settings。")
    lines += ['', '全部8000集验证复用 head 与原分类器在同缓存上预测一致；每组前2个真实episode用原整集 batch 在线 encode 对照缓存特征和分数，验证 w=0 精确退化。历史未保存逐 query 预测，因此历史核对基于逐 episode 准确率，真实在线回归和异常诊断才比较逐 query 预测。所有原始索引/路径/类别映射均通过原路径和文件SHA引用，不复制清单。', '',
              '验证：checkpoint/best epoch、32个 Dogs 运行源码、split类别不重叠、测试原图及PNG逐文件哈希、完整ImageFolder映射、每集支持/查询位置和无重复、CUB锁参数均通过。eval + inference_mode + requires_grad=False；BN buffer保持不变，无优化器/反向更新。统计已由CSV重算；历史实验目录内容指纹前后相同。', '',
              '## 身份、命令与环境', '',
              f"Dogs base `{metadata['identity']['settings']['dogs_commit']}`；CUB head source `{metadata['identity']['settings']['proker_commit']}`。checkpoint SHA256 `{metadata['identity']['settings']['checkpoint_sha256']}`，best epoch520。", '',
              '实际生效配置与所有输入SHA见 config.yaml / metadata.json；环境、命令、数据内容汇总哈希、源码SHA、输入来源和统计核验集中记录在 metadata.json。缓存只包含三路未归一化 post-FC 特征，权重和数据只读引用。', '',
              f"cwd: `{ROOT}`", '', '```bash', metadata['command'], '```', '',
              f"环境：Python {metadata['environment']['python']}，Torch {metadata['environment']['torch']}，CUDA {metadata['environment']['cuda']}，{metadata['environment']['gpu']}。FP32、AMP/TF32=false，deterministic warn_only。", '',
              '未完成同身份恢复加 --resume；完整设置CSV只读复用。completed入口直接退出，不改写产物。只读重算使用 --audit。', '',
              '## 分析、局限与工程判断', '']
    for key, result in results.items():
        p = result['methods']['proker']
        low, high = p['gain_pp']-p['paired_ci95_halfwidth_pp'], p['gain_pp']+p['paired_ci95_halfwidth_pp']
        evidence = '正向区间' if low > 0 else '负向区间' if high < 0 else '区间包含0'
        lines.append(f"- {key}：ProKeR paired区间 [{low:+.4f}, {high:+.4f}] pp（{evidence}）。收益量级 {p['gain_pp']:+.4f} pp，需同时考虑实际分类收益和额外拟合开销。")
    lines += ['', '单个训练 seed 的 episode CI 不能证明跨训练 seed 稳定性；paired normal CI 未作多重比较校正。固定2000个历史任务存在样本/类别复用；结论限于该checkpoint、作者划分及已锁定公式，不外推所有模型或数据集。测试标签仅用于准确率统计，没有用于拟合、调参或决策修改。', '',
              '工程建议：若30-way的收益具实用量级且5-way不退化，可继续做预先规定的多训练seed和更多数据集验证；若高way收益很小或低way退化，优先保留简洁原版分类头。任何下一轮应另设实验协议，当前Dogs测试结果不用于重新选参。', '']
    (run / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='dogs_seed42_proker_transfer_v1')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id
    settings = yaml.safe_load((FAMILY / 'config.yaml').read_text())
    settings['run_id'] = args.run_id
    run = FAMILY / 'runs' / args.run_id
    metadata_path = run / 'metadata.json'
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if saved['status'] == 'completed' and not args.audit:
            print(f'completed run protected; unchanged: {run}')
            return
        if not args.resume and not args.audit:
            raise FileExistsError('Existing incomplete run: use --resume after verifying input identity')
    elif args.audit or run.exists():
        raise FileNotFoundError('Missing valid run metadata; existing directory will not be overwritten')
    config, dataset, records, identity = inputs(settings)
    preservation_paths = [Path(settings['dogs_root']) / 'experiments/sceg_dogs_author',
                          Path(settings['proker_root']) / 'experiments/sceg_proker',
                          Path('/root/projects/aris/sceg_fsfg/experiments/sceg_original')]
    preservation = {str(p): tree_identity(p) for p in preservation_paths}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        assert metadata['identity'] == identity, 'Resume/audit input or source identity changed'
        assert metadata['preservation_before'] == preservation, 'Historical experiment changed'
    else:
        resource = resource_check()
        run.mkdir(parents=True)
        metadata = {'status': 'running', 'identity': identity, 'preservation_before': preservation,
                    'resource': resource, 'command': 'rtk proxy env CUDA_VISIBLE_DEVICES=0 ' + sys.executable +
                    ' -m experiments.sceg_proker_dogs.run --run-id ' + args.run_id,
                    'started_unix': time.time()}
        write_json(metadata_path, metadata)
        (run / 'config.yaml').write_text(yaml.safe_dump({'transfer': settings, 'frozen_sceg': config,
                                                       'input_identity': identity}, sort_keys=False))
    if args.audit:
        results = json.loads((run / 'results.json').read_text())
        assert results['status'] == 'completed' and metadata['status'] == 'completed'
        print(json.dumps(audit(run, records, results['settings'])))
        return
    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger = logging.getLogger('dogs_transfer')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        for handler in (logging.FileHandler(run / 'run.log'), logging.StreamHandler()):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        try:
            logger.info('input and all 8000 episode mappings verified; no Dogs parameter selection')
            seed_everything(42, deterministic=True, allow_tf32=False, warn_only=True)
            torch.set_num_threads(settings['cpu_threads'])
            payload = torch.load(Path(settings['dogs_root']) / settings['dogs_run'] / 'checkpoint.pth',
                                 map_location='cpu', weights_only=False)
            assert payload['completed_epochs'] == payload['best_epoch'] == 520
            assert payload['config']['model'] == config['model']
            model = SCEG(ModelConfig.from_dict(config['model'])).cuda()
            model.load_state_dict(payload['model_state_dict'], strict=True)
            model.eval().requires_grad_(False)
            del payload
            bn_before = {n: b.clone() for n, b in model.named_buffers()
                         if 'running_' in n or n.endswith('num_batches_tracked')}
            metadata['environment'] = {'python': platform.python_version(), 'executable': sys.executable,
                'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                'platform': platform.platform(), 'packages': {name: importlib.metadata.version(name)
                    for name in ('torchvision', 'numpy', 'Pillow', 'PyYAML')}}
            features = cache_features(model, dataset, settings, run, metadata, logger)
            baseline = replay(model, dataset, features, records, config, settings, run, metadata, logger)
            assert all(torch.equal(buffer, bn_before[name]) for name, buffer in model.named_buffers() if name in bn_before)
            assert not model.training and not any(p.requires_grad for p in model.parameters())
            classifier = model.classifier
            del model, bn_before
            torch.cuda.empty_cache()
            results = formal(features, classifier, records, baseline, config, settings, run, metadata, logger)
            metadata['statistics_audit'] = audit(run, records, results)
            metadata['preservation_after'] = {str(p): tree_identity(p) for p in preservation_paths}
            assert preservation == metadata['preservation_after'], 'Historical files changed during evaluation'
            # Recheck on-disk input/source identities after all GPU work.
            _, _, _, identity_after = inputs(settings)
            assert identity_after == identity, 'Inputs changed during evaluation'
            metadata['bn_unchanged'] = True
            metadata['completed_unix'] = time.time()
            metadata['status'] = 'completed'
            report(run, results, metadata)
            write_json(run / 'results.json', {'status': 'completed', 'settings': results})
            write_json(metadata_path, metadata)
            logger.info('completed: four settings / 8000 episodes, statistics and historical preservation passed')
        except Exception:
            metadata['status'] = 'failed'
            write_json(metadata_path, metadata)
            logger.exception('stopped; historical inputs untouched')
            raise


if __name__ == '__main__':
    main()
