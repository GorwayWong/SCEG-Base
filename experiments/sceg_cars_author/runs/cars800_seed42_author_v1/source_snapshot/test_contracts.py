"""Focused Cars mapping, sampler, identity/statistics and optional timing tests."""
from copy import deepcopy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.sceg_cars_author import prepare, run
from datasets.folder import make_dataset
from datasets.samplers import TrainingEpisodeSampler
from models import SCEGOutput
from utils.reproducibility import seed_everything
import trainers.trainer as trainer


def cfg():
    return {**run.configuration(), 'run_id': 'contract_only'}


def test_original_annotation_author_membership_and_collision_safe_imagefolder():
    groups = prepare.parse_split()
    records, names = prepare.annotation_records()
    manifest = [json.loads(x) for x in (run.HERE / 'prep/samples.jsonl').read_text().splitlines()]
    assert len(manifest) == len(records) == 16185
    assert len({r['relative_path'] for r in manifest}) == 16185
    by_identity = {r['identity']: r for r in records}
    source_filenames = [Path(r['identity']).name for r in records]
    assert len(set(source_filenames)) < len(source_filenames)  # collision is real
    config = cfg()
    for split in ('train', 'val', 'test'):
        ds = make_dataset(config, split)
        subset = [r for r in manifest if r['split'] == split]
        assert len(ds) == len(subset)
        assert len(ds.classes) == len(groups[split])
        for record in subset:
            original = by_identity[record['identity']]
            cid = record['original_class_id']
            assert cid == original['original_class_id'] and cid in groups[split]
            assert record['class_name'] == names[cid-1]
            assert record['imagefolder_label'] == ds.class_to_idx[f'{cid:03d}']
            assert (prepare.ROOT / record['relative_path']).resolve() == Path(original['source_path']).resolve()
    assert set(r['train_global_label'] for r in manifest if r['split'] == 'train') == set(range(130))


def test_support_first_depletion_and_train_labels():
    seed_everything(42, warn_only=True)
    ds = make_dataset(cfg(), 'train')
    seen = set()
    for indices in TrainingEpisodeSampler(ds, 10, (5, 15)):
        assert len(indices) == len(set(indices)) == 200 and not seen.intersection(indices)
        seen.update(indices)
        labels = np.array([ds.samples[i][1] for i in indices])
        supports, queries = labels[:50].reshape(10,5), labels[50:].reshape(10,15)
        assert (supports == supports[:, :1]).all() and (queries == supports[:, :1]).all()
        assert len(set(supports[:,0])) == 10 and labels.min() >= 0 and labels.max() < 130


def result_fixture():
    config = cfg()
    paths = [f'{c:03d}/cars_test__{i}.png' for c in range(5) for i in range(17)]
    indices = [c*17 for c in range(5)] + [c*17+i for c in range(5) for i in range(1,17)]
    values = [25.0, 75.0]*1000
    result = dict(protocol_id=run.PROTOCOL, run_id=config['run_id'], seed=42,
                  dataset='stanford_car', split='test', way=5, shot=1, query=16,
                  episodes=2000, checkpoint_sha256='best',
                  split_sha256=config['data_protocol']['split_manifest_sha256'],
                  config_sha256=run.config_hash(config), episode_accuracies=values,
                  sample_index_paths=paths, episode_indices=[indices]*2000,
                  episode_classes=[[f'{c:03d}' for c in range(5)]]*2000,
                  accuracy=50.0, confidence_interval=float(1.96*np.sqrt(np.var(values)/2000)),
                  seconds=1.0, peak_cuda_memory_bytes=0, rng_after=run.json_rng(), dropblock_after={})
    return config, result, paths


def test_result_identity_ci_population_variance_and_disjointness():
    config, result, paths = result_fixture()
    assert run.validate_result(result,config,5,1,'best',paths)['statistics_recomputed']
    for key,value in [('run_id','other'), ('checkpoint_sha256','other'), ('split_sha256','old'),
                      ('query',15), ('accuracy',99), ('episode_accuracies',[50]*1999)]:
        with pytest.raises(ValueError):
            run.validate_result({**result,key:value},config,5,1,'best',paths)
    broken = deepcopy(result)
    broken['episode_indices'][0] = [0]*85
    with pytest.raises(ValueError, match='support/query'):
        run.validate_result(broken,config,5,1,'best',paths)


def test_rng_checkpoint_and_completed_report_constraints(tmp_path):
    seed_everything(42, warn_only=True)
    state = json.loads(json.dumps(run.json_rng()))
    expected = np.random.permutation(49), torch.rand(4)
    run.restore_json_rng(state)
    np.testing.assert_array_equal(np.random.permutation(49), expected[0])
    torch.testing.assert_close(torch.rand(4),expected[1],rtol=0,atol=0)
    config = cfg()
    payload = dict(config=config, optimizer={}, scheduler={}, rng={}, dropblock_counters={}, completed_epochs=800)
    run.checkpoint_identity(payload,config)
    with pytest.raises(ValueError):
        run.checkpoint_identity(payload,{**config,'run_id':'other'})
    del payload['rng']
    with pytest.raises(ValueError,match='Incomplete'):
        run.checkpoint_identity(payload,config)
    with pytest.raises(ValueError,match='four verified'):
        run.write_report(SimpleNamespace(path=tmp_path,metrics={'status':'completed','best_sha256':'best'}),config)


def test_cli_evaluate_guard_is_nondestructive(tmp_path, monkeypatch):
    config = cfg()
    path = tmp_path / 'runs/contract_only'
    path.mkdir(parents=True)
    monkeypatch.setattr(run,'HERE',tmp_path)
    monkeypatch.setattr(run,'configuration',lambda: {k:v for k,v in config.items() if k != 'run_id'})
    monkeypatch.setattr(run,'configure',lambda config: None)
    monkeypatch.setattr(run,'load_config',lambda p:config)
    monkeypatch.setattr(run,'source_identity',lambda:{})
    monkeypatch.chdir(run.REPO)
    monkeypatch.setattr(run.sys,'argv',['run','--mode','evaluate','--run-id','contract_only'])
    (path/'metrics.json').write_text('{"status":"training"}')
    (path/'execution_sources.json').write_text('{}')
    payload = dict(config=config,optimizer={},scheduler={},rng={},dropblock_counters={},completed_epochs=20)
    monkeypatch.setattr(torch,'load',lambda *a,**kw:payload)
    before = (path/'metrics.json').read_bytes()
    with pytest.raises(ValueError,match='epoch800'):
        run.main()
    assert (path/'metrics.json').read_bytes() == before
    (path/'metrics.json').write_text('{"status":"completed"}')
    monkeypatch.setattr(run.sys,'argv',['run','--mode','pipeline','--resume','--run-id','contract_only'])
    run.main()
    assert json.loads((path/'metrics.json').read_text())['status'] == 'completed'


def test_public_timing_addition_keeps_default_update(tmp_path, monkeypatch):
    config = cfg()
    config['model'].update(train_way=3,train_shot=1,train_query=1,base_classes=3)
    config['train'].update(epochs=1,validation_every=1)
    config['eval'].update(way=3,validation_shot=1)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.eye(3)*.1)
        def forward(self,images):
            x = self.weight.tanh(); return SCEGOutput(x,x,x,x.repeat(2,1))
        def expected_inactive_parameters(self): return set()
    class Record:
        def __init__(self,path):
            self.path=path; path.mkdir(); self.metrics={}; self.logger=logging.getLogger(str(path))
        def update_metrics(self,**values): self.metrics.update(values)
        def checkpoint(self,payload): torch.save(payload,self.path/'checkpoint.pth')
    class Loader(list): dataset=SimpleNamespace(classes=[0,1,2])
    monkeypatch.setattr(trainer,'make_loader',lambda *a,**k:Loader([(torch.zeros(6,1),torch.arange(3).repeat(2))]))
    monkeypatch.setattr(trainer,'evaluate_episodes',lambda *a,**k:{'accuracy':10.,'confidence_interval':0.})
    model1,model2=Model(),Model()
    record1,record2=Record(tmp_path/'timed'),Record(tmp_path/'untimed')
    trainer.train(model1,config,record1)
    config['runtime']['record_epoch_timing']=False
    trainer.train(model2,config,record2)
    torch.testing.assert_close(model1.weight,model2.weight,atol=0,rtol=0)
    assert record1.metrics['train_seconds'] > 0 and record1.metrics['validation_seconds'] >= 0
    assert 'train_seconds' not in record2.metrics
    assert not list(record1.path.glob('activations*'))
    assert set(p.name for p in record1.path.glob('*.pth')) == {'latest.pth','checkpoint.pth'}
