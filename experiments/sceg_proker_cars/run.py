"""Fixed Cars transfer. Run as a module from this worktree; no training or sampling."""
import argparse
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets import make_dataset
from evaluators.fewshot_eval import encode_in_chunks
from models.sceg import SCEG
from utils.config import ModelConfig
from utils.reproducibility import seed_everything, seed_worker
from .head import prepare, original_scores, linear_ridge, proker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ('sceg', 'ridge', 'proker')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256_file_bytes': sha(path)}


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {'accuracy': float(v.mean()), 'confidence_interval': float(1.96 * v.std(ddof=0) / math.sqrt(len(v)))}


def summaries(values):
    result = {name: stats(v) for name, v in values.items()}
    for name in ('ridge', 'proker'):
        delta = np.asarray(values[name]) - np.asarray(values['sceg'])
        s = stats(delta)
        result[name].update(gain=s['accuracy'], paired_ci=s['confidence_interval'])
    return result


def inputs(cfg):
    base = Path(cfg['baseline_root'])
    run = base / cfg['baseline_run']
    model_cfg = yaml.safe_load((run / 'config.yaml').read_text())
    checkpoint = run / 'checkpoint.pth'
    assert sha(checkpoint) == cfg['checkpoint_sha256']
    assert model_cfg['data_protocol']['protocol_id'] == 'sceg_author_cars_v1'
    assert cfg['settings'] == [[5, 1], [5, 5], [49, 1], [49, 5]]
    assert cfg['episodes'] == 2000 and cfg['query'] == 16
    assert cfg['parameters'] == {1: dict(beta=10., ridge_lambda=1., residual_weight=1., linear_ridge_lambda=100.),
                                 5: dict(beta=1., ridge_lambda=.1, residual_weight=1.5, linear_ridge_lambda=.1)}
    assert cfg['runtime'] == dict(device='cuda:0', workers=4, cpu_threads=2, encoding_chunk_size=128, amp=False, allow_tf32=False)
    prep_path = base / model_cfg['data_protocol']['prepared']
    prep = read(prep_path)
    assert prep['class_counts'] == {'train': 130, 'val': 17, 'test': 49}
    assert prep['image_counts']['test'] == 4028 and prep['full_decode_passed']
    refs = {'checkpoint': ref(checkpoint), 'baseline_config': ref(run / 'config.yaml'), 'prepared': ref(prep_path)}
    for name in ('split_manifest', 'sample_manifest'):
        path = prep_path.parent / prep[name]
        assert sha(path) == prep[name + '_sha256'] == model_cfg['data_protocol'][name + '_sha256']
        refs[name] = ref(path)
    ds = make_dataset(model_cfg, 'test')
    paths = [str(Path(p).relative_to(ds.root)) for p, _ in ds.samples]
    assert len(paths) == 4028 and len(ds.classes) == 49
    assert all(Path(p).is_file() for p, _ in ds.samples)
    path_classes = np.array([Path(p).parts[0] for p in paths])
    tasks = {}
    for way, shot in cfg['settings']:
        key = f'{way}way_{shot}shot'
        source = run / f'test_{key}.json'
        d = read(source)
        assert d['checkpoint_sha256'] == cfg['checkpoint_sha256']
        assert d['split_sha256'] == model_cfg['data_protocol']['split_manifest_sha256']
        assert d['protocol_id'] == 'sceg_author_cars_v1' and d['sample_index_paths'] == paths
        assert (d['way'], d['shot'], d['query'], d['episodes']) == (way, shot, 16, 2000)
        indices = np.asarray(d['episode_indices'], dtype=np.int64)
        assert indices.shape == (2000, way * (shot + 16))
        assert indices.min() >= 0 and indices.max() < len(ds)
        assert np.all(np.diff(np.sort(indices, axis=1), axis=1) > 0), 'Repeated support/query sample'
        classes = np.asarray(d['episode_classes'])
        assert classes.shape == (2000, way)
        assert all(len(set(row)) == way for row in classes)
        assert np.array_equal(path_classes[indices[:, :way * shot]].reshape(2000, way, shot),
                              np.repeat(classes[:, :, None], shot, axis=2))
        assert np.array_equal(path_classes[indices[:, way * shot:]].reshape(2000, way, 16),
                              np.repeat(classes[:, :, None], 16, axis=2))
        historic = np.asarray(d['episode_accuracies'], dtype=np.float64)
        assert historic.shape == (2000,)
        assert abs(stats(historic)['accuracy'] - d['accuracy']) < 1e-10
        refs[key] = {**ref(source), 'relative_to_baseline_root': str(source.relative_to(base)),
                     'fields': ['episode_indices', 'episode_classes', 'sample_index_paths', 'episode_accuracies'],
                     'historical_parsed_config_sha256': d['config_sha256']}
        tasks[key] = (indices, historic)
    refs['reference_head'] = ref(Path(cfg['reference_root']) / 'experiments/sceg_proker_dogs/head.py')
    refs['reference_config'] = ref(Path(cfg['reference_root']) / 'experiments/sceg_proker_dogs/config.yaml')
    refs['reference_report'] = ref(Path(cfg['reference_root']) / 'experiments/sceg_proker_dogs/runs/dogs_seed42_proker_transfer_v1/REPORT.md')
    return model_cfg, checkpoint, ds, tasks, refs


def all_scores(features, way, shot, classifier, params):
    prepared = prepare(features, way, shot, classifier)
    return {'sceg': original_scores(prepared[1], prepared[2], classifier),
            'ridge': linear_ridge(prepared, params['linear_ridge_lambda']),
            'proker': proker(prepared, classifier, params['beta'], params['ridge_lambda'], params['residual_weight'])}


def compare(a, b, atol, rtol):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return float((a - b).abs().max())


def online_features(model, ds, indices):
    # Only a single small episode is loaded; never all test images at once.
    images = torch.stack([ds[int(i)][0] for i in indices])
    return torch.stack(encode_in_chunks(model, images, 128), dim=1)


def short_check(model, ds, tasks, features, cfg):
    checks = []
    tol = cfg['checks']
    for way, shot in cfg['settings']:
        key = f'{way}way_{shot}shot'
        indices = tasks[key][0][0]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        online = online_features(model, ds, indices)
        cached = features[indices].to('cuda:0')
        err = compare(online, cached, tol['feature_atol'], tol['feature_rtol'])
        reference = model.classify(tuple(online[:, s] for s in range(3)), way, shot).scores
        scores_online = all_scores(online, way, shot, model.classifier, cfg['parameters'][shot])
        scores_cache = all_scores(cached, way, shot, model.classifier, cfg['parameters'][shot])
        baseline_error = compare(scores_online['sceg'], reference, tol['baseline_atol'], tol['baseline_rtol'])
        prepared = prepare(cached, way, shot, model.classifier)
        w0 = proker(prepared, model.classifier, cfg['parameters'][shot]['beta'], cfg['parameters'][shot]['ridge_lambda'], 0.)
        assert torch.equal(w0, scores_cache['sceg'] / 3)
        assert torch.equal(w0.argmax(1), scores_cache['sceg'].argmax(1))
        assert torch.equal(reference.argmax(1), scores_online['sceg'].argmax(1))
        errors = {}
        for method in METHODS:
            errors[method] = compare(scores_online[method], scores_cache[method], tol['score_atol'], tol['score_rtol'])
            assert torch.equal(scores_online[method].argmax(1), scores_cache[method].argmax(1)), f'{key} {method} prediction mismatch'
        target = torch.arange(way, device='cuda:0').repeat_interleave(16)
        online_acc = 100 * (reference.argmax(1) == target).sum().item() / len(target)
        assert abs(online_acc - tasks[key][1][0]) < 1e-10
        torch.cuda.synchronize()
        checks.append(dict(setting=key, episode=0, feature_max_abs=err, score_max_abs=errors,
                           original_score_max_abs=baseline_error, prediction_differences=0, w0_exact=True,
                           seconds=time.perf_counter() - started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                           historical_accuracy_match=True))
        logging.info('short check passed %s feature_error=%.3g score_errors=%s', key, err, errors)
    # Same image batch at changed eval counters must produce identical features.
    images = torch.stack([ds[i][0] for i in range(2)]).to('cuda:0')
    first = model.encode(images)
    for module in model.modules():
        if hasattr(module, 'ramp_batches'):
            module.num_batches_tracked += 100000
    second = model.encode(images)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    return {'episodes': checks, 'dropblock_counter_independence_exact': True, 'status': 'passed'}


def diagnose(model, ds, features, indices, historic, cached_acc, way, shot, cfg):
    mismatches = np.flatnonzero(np.abs(np.asarray(cached_acc) - historic) > 1e-10)
    if len(mismatches) > 10:
        raise RuntimeError(f'Systematic historical mismatch: {len(mismatches)} episodes; no full replay')
    records = []
    for i in mismatches:
        online = online_features(model, ds, indices[i])
        cached = features[indices[i]].to('cuda:0')
        a = model.classify(tuple(online[:, s] for s in range(3)), way, shot).scores
        b = model.classify(tuple(cached[:, s] for s in range(3)), way, shot).scores
        tol = cfg['checks']
        feature_error = compare(online, cached, tol['feature_atol'], tol['feature_rtol'])
        score_error = compare(a, b, tol['score_atol'], tol['score_rtol'])
        target = torch.arange(way, device='cuda:0').repeat_interleave(16)
        accuracy = 100 * (a.argmax(1) == target).sum().item() / len(target)
        assert abs(accuracy - historic[i]) < 1e-10, 'Safe replay does not explain historical accuracy'
        assert abs(100 * (b.argmax(1) == target).sum().item() / len(target) - cached_acc[i]) < 1e-10
        changed = (a.argmax(1) != b.argmax(1)).nonzero().flatten()
        details = []
        for q in changed.tolist():
            pa, pb = int(a[q].argmax()), int(b[q].argmax())
            assert abs(float(a[q, pa] - a[q, pb])) <= 2 * score_error + 1e-7
            details.append(dict(query=q, online_prediction=pa, cache_prediction=pb, target=int(target[q]),
                                online_top2_gap=float(a[q].topk(2).values.diff().abs()),
                                cache_top2_gap=float(b[q].topk(2).values.diff().abs())))
        records.append(dict(episode=int(i), historical=float(historic[i]), cached=float(cached_acc[i]),
                            safe_replay=accuracy, feature_max_abs=feature_error, score_max_abs=score_error,
                            changed_predictions=details, explanation='Safe replay matches history; bounded FP32 score perturbation changes near-tied classes'))
    return records


def evaluate(model, ds, features, task, way, shot, cfg, identity):
    indices, historic = task
    target = torch.arange(way, device='cuda:0').repeat_interleave(16)
    values = {name: [] for name in METHODS}
    times = np.zeros(5)  # transfer, shared preparation, SCEG, Ridge, ProKeR
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    params = cfg['parameters'][shot]
    for offset in range(0, 2000, 100):
        pending = []
        for row in indices[offset:offset + 100]:
            # CPU gather is part of wall time, outside CUDA event intervals.
            cpu = features[row]
            events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            events[0].record()
            f = cpu.to('cuda:0')
            events[1].record()
            prepared = prepare(f, way, shot, model.classifier)
            events[2].record()
            p0 = original_scores(prepared[1], prepared[2], model.classifier).argmax(1)
            events[3].record()
            p1 = linear_ridge(prepared, params['linear_ridge_lambda']).argmax(1)
            events[4].record()
            p2 = proker(prepared, model.classifier, params['beta'], params['ridge_lambda'], params['residual_weight']).argmax(1)
            events[5].record()
            counts = torch.stack([(p == target).sum() for p in (p0, p1, p2)])
            pending.append((events, counts))
        torch.cuda.synchronize()
        counts = torch.stack([item[1] for item in pending]).cpu().numpy()
        for events, _ in pending:
            times += [events[j].elapsed_time(events[j + 1]) for j in range(5)]
        for j, name in enumerate(METHODS):
            values[name].extend((100 * counts[:, j] / len(target)).tolist())
        logging.info('%dway_%dshot episodes=%d/2000 elapsed=%.2fs', way, shot, offset + 100, time.perf_counter() - started)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    differences = diagnose(model, ds, features, indices, historic, values['sceg'], way, shot, cfg)
    return dict(status='completed', identity=identity, way=way, shot=shot, query=16, episodes=2000,
                episode_accuracies=values, statistics=summaries(values), historical_differences=differences,
                historical_mean_difference_pp=float(np.mean(np.asarray(values['sceg']) - historic)),
                timing=dict(evaluation_wall_seconds=elapsed, historical_diagnosis_seconds=time.perf_counter() - started - elapsed,
                            mean_cuda_event_ms=dict(zip(('transfer', 'shared_preparation', 'sceg', 'ridge', 'proker'), (times / 2000).tolist()))),
                peak_cuda_allocated_bytes=peak, peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())


def cpu_summary(run, info, save_report=False):
    rows = []
    for way, shot in info['config']['settings']:
        key = f'{way}way_{shot}shot'
        result = read(run / f'test_{key}.json')
        assert result['status'] == 'completed' and result['episodes'] == 2000
        assert result['identity'] == {'run_identity': info['run_identity'], 'tasks': info['inputs'][key]}
        assert sha(info['inputs'][key]['path']) == info['inputs'][key]['sha256_file_bytes']
        values = result['episode_accuracies']
        assert set(values) == set(METHODS)
        assert all(len(v) == 2000 and np.isfinite(v).all() and min(v) >= 0 and max(v) <= 100 for v in values.values())
        recomputed = summaries(values)
        assert recomputed == result['statistics']
        rows.append((key, result))
    if not save_report:
        print(json.dumps({k: r['statistics'] for k, r in rows}, indent=2))
        return
    lines = ['# Fixed CUB-parameter Cars transfer', '',
             'Status: completed. Fixed seed42 Cars model, best epoch 660; no training, Cars validation selection, parameter search, or extra seeds.',
             'This experiment does not resolve the Cars paper reproduction gap and does not establish stability across training seeds.', '',
             '| Setting | SCEG accuracy ± CI | Ridge accuracy ± CI | ProKeR accuracy ± CI | Ridge gain ± paired CI | ProKeR gain ± paired CI |',
             '|---|---:|---:|---:|---:|---:|']
    for key, r in rows:
        s = r['statistics']
        cells = [f"{s[n]['accuracy']:.6f} ± {s[n]['confidence_interval']:.6f}" for n in METHODS]
        cells += [f"{s[n]['gain']:+.6f} ± {s[n]['paired_ci']:.6f}" for n in ('ridge', 'proker')]
        lines.append('| ' + ' | '.join([key] + cells) + ' |')
    lines += ['', 'Accuracy/CI are percentages; paired gain/CI are percentage points. Each CI is 1.96 × population std (ddof=0) / sqrt(2000). Paired intervals use per-episode method minus same-feature SCEG.', '',
              '## Timing and resources', '',
              'CUDA events are collected and synchronized once per 100 tasks. Transfer includes pageable CPU→GPU copy; CPU gather, label comparison/accuracy reduction, event bookkeeping and reporting are included only in group wall time. Shared preparation includes raw-support means, original prototype normalization, support/query kernel normalization and support one-hot labels. Each method interval includes support fitting (where applicable), query scoring and argmax; ProKeR computes its own f0, preserving its average-score scale. Add shared preparation to a method interval for its head latency. Backbone encoding and transfers are excluded from head latency. Event intervals may include CPU launch gaps; these are observed inference timings, not isolated solver benchmarks.', '',
              '| Setting | Wall seconds | Transfer ms/task | Shared ms/task | SCEG ms/task | Ridge ms/task | ProKeR ms/task | Peak allocated GiB |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, r in rows:
        t = r['timing']['mean_cuda_event_ms']
        lines.append('| ' + ' | '.join([key, f"{r['timing']['evaluation_wall_seconds']:.3f}"] + [f'{v:.4f}' for v in t.values()] + [f"{r['peak_cuda_allocated_bytes']/2**30:.3f}"]) + ' |')
    lines += ['', f"Encoding: {info['encoding']['seconds']:.3f} seconds; peak allocated {info['encoding']['peak_cuda_allocated_bytes']/2**30:.3f} GiB; CPU cache {info['encoding']['cache_bytes']} bytes, [4028,3,640] FP32. Batch/chunk 128, workers 4, CPU threads 2; AMP/TF32 off.",
              f"All-stage elapsed before CPU summary: {info['inference_wall_seconds']:.3f} seconds. Short-check details and environment are in resolved.json.", '', '## Necessary checks and history', '',
              'Static loading validates every saved task index, support/query ordering, original class sequence and no repeated sample. Four episode-0 checks cover both shots and maximum shape; safe 128-chunk features and all method scores meet fixed FP32 tolerances with identical predictions. Original model.classify agrees with the migrated sum head; w=0 equals original score/3 exactly with identical predictions. BN buffers remain fixed; changing eval DropBlock counters leaves features exactly unchanged. No query labels enter fitting or normalization.', '']
    for key, r in rows:
        lines.append(f"- {key}: {len(r['historical_differences'])} historical episode accuracy differences; mean impact {r['historical_mean_difference_pp']:+.9f} pp. Details and selective safe-replay evidence are retained in test_{key}.json.")
    lines += ['', '## Provenance and reproduction', '',
              f"Base: `{info['config']['base_commit']}`. ProKeR reference: `{info['config']['reference_commit']}`. Checkpoint SHA-256: `{info['config']['checkpoint_sha256']}`.",
              'head.py carries the minimal reference algorithm. Differences are support-first reshape means, shared preparation, and validation at the loading boundary. Prototypes average raw supports before original normalization; f0 is the original stage3+stage4+stage2 sum divided by 3. A=solve(I+Kss/lambda,Y-f0(S)); f=f0+w*Kqs@A. Ridge uses solve(UsUs.T+lambda*I,Y), without intercept.',
              'resolved.json contains the actual resolved Cars configuration, fixed parameters, commands, environment, input byte hashes, canonical parsed-config hash, base commit and executed source byte hashes. Historical indices/path maps are referenced, not duplicated. Existing data-preparation evidence is reused; images are not rehashed. Cache remains local. Source files are the actual added execution version and are not edited after launch.', '',
              f"Working directory: `{info['cwd']}`", '```bash', info['command'],
              info['command'] + ' --summary', '```',
              'A completed run exits without rewriting. An incomplete run requires explicit continuation instructions; this entry does not auto-restart it.', '',
              'Only experiments/sceg_proker_cars/ is added on experiment/sceg_proker_cars. Public modules, historical inputs and other worktrees are read-only; no commit/push/merge or environment changes. CPU-only reaggregation verified all 8000 records and saved statistics. No unresolved evaluation issues.', '']
    tmp = run / 'REPORT.md.tmp'
    tmp.write_text('\n'.join(lines))
    tmp.replace(run / 'REPORT.md')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', action='store_true', help='CPU-only verification of saved results')
    args = parser.parse_args()
    cfg = yaml.safe_load((HERE / 'config.yaml').read_text())
    run = HERE / 'runs' / cfg['run_id']
    if run.exists():
        if (run / 'resolved.json').exists() and read(run / 'resolved.json').get('status') == 'completed':
            if args.summary:
                cpu_summary(run, read(run / 'resolved.json'))
            else:
                print(f'Already completed: {run / "REPORT.md"}')
            return
        raise RuntimeError('Existing incomplete/unknown run: stop; explicit continuation is required')
    assert not args.summary, 'No completed results to summarize'
    assert Path.cwd().resolve() == ROOT
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == cfg['base_commit']
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'experiment/sceg_proker_cars'
    assert not subprocess.check_output(['git', 'diff', '--name-only'], text=True).strip()
    source_files = ('config.yaml', 'head.py', 'run.py')
    sources = {n: ref(HERE / n) for n in source_files}
    start = time.perf_counter()
    model_cfg, checkpoint, ds, tasks, refs = inputs(cfg)
    torch.set_num_threads(2)
    seed_everything(model_cfg['seed'], deterministic=True, allow_tf32=False, warn_only=True)
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_properties(0).total_memory >= 15 * 2**30
    run.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(run / 'run.log'), logging.StreamHandler(sys.stdout)])
    command = '/root/projects/aris/sceg_fsfg/.venv/bin/python -B -m experiments.sceg_proker_cars.run'
    info = dict(status='running', config=cfg, cars_resolved_config=model_cfg, inputs=refs, sources=sources,
                config_sha256_canonical_json=canonical(cfg), cwd=str(ROOT), command=command,
                environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                 torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                                 gpu=torch.cuda.get_device_name(0), cpu_threads=torch.get_num_threads()),
                startup_resources=dict(gpu_free_total_bytes=torch.cuda.mem_get_info(),
                                       host_mem_available_bytes=os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'),
                                       disk_available_bytes=os.statvfs(run).f_bavail*os.statvfs(run).f_frsize),
                startup_observation='Original worktree has pre-existing untracked docs/paper/; left untouched. No compute process shown by nvidia-smi. 14812 MiB GPU free, 6.2 GiB host available, 855 GiB disk, 3.9 GiB shm.')
    info['run_identity'] = canonical({'config': cfg, 'inputs': refs, 'sources': sources})
    write(run / 'resolved.json', info)
    try:
        logging.info('Inputs verified; loading frozen Cars epoch660 checkpoint on CPU')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert payload['completed_epochs'] == 660 and payload['best_epoch'] == 660
        state = payload['model_state_dict']
        del payload
        model = SCEG(ModelConfig.from_dict(model_cfg['model']))
        model.load_state_dict(state, strict=True)
        del state
        gc.collect()
        model.requires_grad_(False).to('cuda:0').eval()
        buffers = {n: b.clone() for n, b in model.named_buffers()}
        assert all(not m.training for m in model.modules())
        assert all(m.track_running_stats for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
        with torch.inference_mode():
            logging.info('Encoding test-only cache: 4028 images, three 640D branches, FP32 batch128')
            loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=False, worker_init_fn=seed_worker)
            features = torch.empty((len(ds), 3, 640), dtype=torch.float32, device='cpu')
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            encoding_start = time.perf_counter()
            cursor = 0
            for images, _ in loader:
                encoded = torch.stack(model.encode(images.to('cuda:0')), dim=1).cpu()
                features[cursor:cursor + len(images)] = encoded
                cursor += len(images)
            torch.cuda.synchronize()
            assert cursor == 4028 and torch.isfinite(features).all()
            info['encoding'] = dict(seconds=time.perf_counter() - encoding_start,
                                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                                    cache_bytes=features.numel()*features.element_size())
            del loader, images, encoded
            assert all(torch.equal(buffers[n], b) for n, b in model.named_buffers())
            torch.save({'features': features, 'mapping_reference': refs['5way_1shot'], 'run_identity': info['run_identity']}, run / 'features.pt')
            info['cache'] = ref(run / 'features.pt')
            info['short_check'] = short_check(model, ds, tasks, features, cfg)
            assert all(torch.equal(buffers[n], b) for n, b in model.named_buffers())
            info['short_check']['bn_buffers_unchanged'] = True
            write(run / 'resolved.json', info)
            logging.info('Short checks passed; starting the four fixed formal groups sequentially')
            for way, shot in cfg['settings']:
                key = f'{way}way_{shot}shot'
                identity = {'run_identity': info['run_identity'], 'tasks': refs[key]}
                result = evaluate(model, ds, features, tasks[key], way, shot, cfg, identity)
                write(run / f'test_{key}.json', result)
                logging.info('Completed %s statistics=%s historical_differences=%d', key, result['statistics'], len(result['historical_differences']))
        assert all(sha(HERE / n) == sources[n]['sha256_file_bytes'] for n in source_files)
        del model, features, buffers
        gc.collect()
        info['inference_wall_seconds'] = time.perf_counter() - start
        cpu_summary(run, info, save_report=True)
        info['status'] = 'completed'
        write(run / 'resolved.json', info)
        logging.info('Completed: CPU reaggregation verified all 8000 episodes; report=%s', run / 'REPORT.md')
    except BaseException as exc:
        info.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        write(run / 'resolved.json', info)
        logging.exception('Stopped without automatic retry')
        raise


if __name__ == '__main__':
    main()
