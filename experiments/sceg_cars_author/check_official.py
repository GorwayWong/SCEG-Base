"""Small Cars-only CUDA oracle against the fixed upstream checkout; no saved weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from tools.verify_official import load_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference.resolve()
    commit = subprocess.check_output(['rtk', 'proxy', 'git', 'rev-parse', 'HEAD'], cwd=reference, text=True).strip()
    assert commit == 'f7f130aeb5b1b49246150574c53a874d4ce8de80'
    root = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != root]
    sys.path.insert(0, str(reference))
    import torch
    from models.SCEG import SCEG as Official
    config_module = load_module('cars_config', root / 'utils/config.py')
    cfg = config_module.load_config(here / 'configs/cars.yaml')
    replica_module = load_module('cars_replica', root / 'models/__init__.py', package=True)
    seed_module = load_module('cars_seed', root / 'utils/reproducibility.py')
    original_loss = load_module('cars_original_loss', reference / 'loss.py').Intra_Inter_Loss
    replica_loss = load_module('cars_replica_loss', root / 'losses/intra_inter_loss.py').IntraInterLoss
    torch.set_num_threads(2)
    seed_module.seed_everything(42, warn_only=True, allow_tf32=False)
    values = dict(cfg['model'], train_way=3, train_shot=1, train_query=1)
    official = Official(SimpleNamespace(dataset='stanford_car', train_way=3, train_shot=1,
                                        train_query_shot=1, base_class=130)).cuda()
    replica = replica_module.SCEG(config_module.ModelConfig.from_dict(values)).cuda()
    assert set(replica.state_dict()) == set(official.state_dict())
    replica.load_state_dict(official.state_dict(), strict=True)
    images = torch.randn(6, 3, 84, 84, device='cuda')
    official.eval(); replica.eval()
    with torch.no_grad():
        expected, actual = official(images), replica(images)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    eval_errors = [float((a-b).abs().max()) for a,b in zip(expected, actual)]
    outputs, losses, gradients, missing, bn_updates = [], [], [], [], []
    for model, loss_type in [(official, original_loss), (replica, replica_loss)]:
        seed_module.seed_everything(42, warn_only=True, allow_tf32=False)
        model.train()
        before = model.SC.g_st[1].running_mean.clone()
        output = model(images)
        criterion = loss_type(3, **cfg['loss']).cuda()
        global_criterion = loss_type(130, **cfg['loss']).cuda()
        target = torch.arange(3, device='cuda')
        loss = sum(criterion(x, target) for x in output[:3]) + global_criterion(output[3], target.repeat(2))
        loss.backward()
        outputs.append([x.detach() for x in output]); losses.append(float(loss))
        gradients.append({n: p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None})
        missing.append({n for n,p in model.named_parameters() if p.grad is None})
        bn_updates.append(not torch.equal(before, model.SC.g_st[1].running_mean))
    assert missing[0] == missing[1] == replica.expected_inactive_parameters()
    assert all(bn_updates) and any(n.startswith('SC.g_st.') for n in missing[1])
    for a,b in zip(*outputs):
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(torch.tensor(losses[0]), torch.tensor(losses[1]), atol=1e-4, rtol=1e-5)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=2e-3, rtol=2e-3)
    sources = {str(p.relative_to(reference)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ('models', 'utils') for p in (reference / folder).rglob('*.py')}
    sources['loss.py'] = hashlib.sha256((reference / 'loss.py').read_bytes()).hexdigest()
    result = {'status': 'passed', 'official_commit': commit, 'official_sources': sources,
              'replica_sources': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for folder in ('models', 'losses') for p in (root / folder).rglob('*.py')},
              'dataset': 'stanford_car', 'base_classes': 130, 'loss_config': cfg['loss'],
              'eval_max_abs_errors': eval_errors, 'train_losses': losses,
              'gradient_max_abs_error': max(float((gradients[0][n]-gradients[1][n]).abs().max()) for n in gradients[0]),
              'gradient_tolerance': {'atol': 0.002, 'rtol': 0.002},
              'inactive_parameters': sorted(missing[1]), 'g_st_bn_updated': bn_updates,
              'weights_saved': False}
    (here / 'checks').mkdir(exist_ok=True)
    (here / 'checks/official_parity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'official_sources'}, indent=2))


if __name__ == '__main__':
    main()
