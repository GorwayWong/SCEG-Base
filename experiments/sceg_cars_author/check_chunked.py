"""One small validation check and an auditable inference-only source update."""
import difflib
import hashlib
import subprocess

import torch

from datasets import make_loader
from evaluators.fewshot_eval import encode_in_chunks
from models import SCEG
from trainers.trainer import dropblock_counters
from utils.config import ModelConfig, load_config
from utils.experiment import write_json
from experiments.sceg_cars_author import run


@torch.no_grad()
def main():
    path = run.HERE / 'runs/cars800_seed42_author_v1'
    config = load_config(path / 'config.yaml')
    original_sources = run.read_json(path / 'execution_sources.json')
    sources = run.source_identity()
    changed = {name for name, sha in sources.items() if original_sources.get(name) != sha}
    assert changed == {'evaluators/fewshot_eval.py', 'experiments/sceg_cars_author/run.py',
                       'experiments/sceg_cars_author/check_chunked.py'}
    preserved = run.read_json(path / 'pause.json')['preserved_sha256_before_equals_after']
    for name, sha in preserved.items():
        assert run.digest(path / name) == sha, name
    run.configure(config)
    model = SCEG(ModelConfig.from_dict(config['model'])).cuda().eval()
    best = torch.load(path / 'checkpoint.pth', map_location='cpu', weights_only=False)
    run.checkpoint_identity(best, config)
    model.load_state_dict(best['model_state_dict'])
    del best
    boundary = run.read_json(path / 'evaluation_boundaries.json')['49way_1shot']
    run.restore_counters(model, boundary['counters'])
    tolerance_config = load_config(run.REPO / 'configs/sceg_original.yaml')['verification']
    tolerance = {'atol': tolerance_config['parity_atol'], 'rtol': tolerance_config['parity_rtol']}
    tasks = []
    for shot in (1, 5):
        loader = make_loader(config, 'val', way=5, shot=shot, episodes=1)
        images, labels = next(iter(loader))
        support = labels[:5*shot].reshape(5, shot)
        query = labels[5*shot:].reshape(5, 16)
        assert (support == support[:, :1]).all() and (query == support[:, :1]).all()
        assert support[:, 0].unique().numel() == 5
        indices = loader.batch_sampler.episode_indices[0]
        assert len(indices) == len(set(indices)) == len(images)
        tasks.append((shot, images))
    # Two original 5-way tasks; concatenate only for a >128-image encoder check,
    # then classify each original task separately with its untouched ordering.
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    original = [model.encode(images.cuda()) for _, images in tasks]
    before = dropblock_counters(model)
    joined = encode_in_chunks(model, torch.cat([images for _, images in tasks]), run.EVAL_CHUNK_SIZE)
    assert dropblock_counters(model) == {name: value+1 for name, value in before.items()}
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
    checks, offset = [], 0
    for (shot, images), expected in zip(tasks, original):
        actual = tuple(branch[offset:offset+len(images)] for branch in joined)
        offset += len(images)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, **tolerance)
            torch.testing.assert_close(left[:5*shot].reshape(5, shot, -1).mean(1),
                                       right[:5*shot].reshape(5, shot, -1).mean(1), **tolerance)
        a, b = model.classify(expected, 5, shot), model.classify(actual, 5, shot)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, **tolerance)
        torch.testing.assert_close(a.scores, b.scores, **tolerance)
        torch.testing.assert_close(b.scores, b[0]+b[1]+b[2], atol=0, rtol=0)
        assert torch.equal(a.scores.argmax(1), b.scores.argmax(1))
        checks.append({'way': 5, 'shot': shot, 'query': 16,
                       'embedding_max_abs_errors': [float((x-y).abs().max()) for x,y in zip(expected,actual)],
                       'scores_max_abs_error': float((a.scores-b.scores).abs().max()),
                       'predictions_equal': True})
    # Preserve the original source record. A small delta records exactly the three
    # reviewed inference/check files, including the untracked Cars entry.
    patch = ''
    for name in sorted(changed):
        if name == 'evaluators/fewshot_eval.py':
            old = subprocess.check_output(['rtk', 'proxy', 'git', 'show', 'HEAD:'+name], cwd=run.REPO).decode()
        elif name.endswith('/run.py'):
            old = (path / 'source_snapshot/run.py').read_text()
        else:
            old = ''
        if name in original_sources:
            assert hashlib.sha256(old.encode()).hexdigest() == original_sources[name]
        patch += ''.join(difflib.unified_diff(old.splitlines(True), (run.REPO/name).read_text().splitlines(True),
                                            fromfile='a/'+name, tofile='b/'+name))
    (path / 'evaluation_source.patch').write_text(patch)
    record = {'status': 'passed', 'at': run.utc(), 'encoding_chunk_size': run.EVAL_CHUNK_SIZE,
              'checkpoint_sha256': preserved['checkpoint.pth'], 'preserved_files': preserved,
              'config_sha256': run.config_hash(config), 'tolerance': tolerance, 'validation_tasks': checks,
              'bn_buffers_unchanged': True, 'encoding_rng_unchanged': True,
              'dropblock_counter_policy': 'one logical increment per episode; eval DropBlock inactive',
              'original_source_manifest_sha256': run.digest(path / 'execution_sources.json'),
              'source_identity': sources, 'changed_files': sorted(changed),
              'patch_sha256': run.digest(path / 'evaluation_source.patch'),
              'scope': 'inference-only memory adaptation; no config, model, training or sampler changes',
              'prior_slowdown': 'shared GPU memory observed; extra tensor retention contributes but is not proven to fully explain slowdown, particularly given historical CUB throughput'}
    write_json(path / 'evaluation_execution.json', record)
    print({'status': 'passed', 'chunk': run.EVAL_CHUNK_SIZE, 'tolerance': tolerance, 'validation_tasks': checks})


if __name__ == '__main__':
    main()
