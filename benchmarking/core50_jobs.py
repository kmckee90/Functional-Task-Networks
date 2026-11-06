"""First complete CORe50 comparison, with domain support and active-budget DER++."""
import hashlib
import json

from ftn_models import VARIANTS
from storage import WORKSPACE, bulk, write_json


def jobs():
    result = []
    for variant in VARIANTS:
        result.append({'id': f'ftn/core50/features/{variant}/seed0', 'dataset': 'core50',
                       'gate': 'core50_full', 'needs_features': True,
                       'command': ['run_ftn.py', '--variant', variant, '--mode', 'features',
                                   '--dataset', 'core50', '--seed', '0']})
    for matched in [False, True]:
        label = 'DER++-matched-features-h256' if matched else 'DER++-500-features-h256'
        result.append({'id': f'mammoth/core50/{label}/seed0', 'dataset': 'core50',
                       'gate': 'core50_full', 'needs_features': True,
                       'command': ['run_mammoth.py', '--method', 'derpp', '--features',
                                   '--head-atoms', '256', '--dataset', 'core50', '--seed', '0']
                                  + (['--matched-memory'] if matched else [])})
    result.append({'id': 'mammoth/core50/l2p/seed0', 'dataset': 'core50', 'gate': 'core50_full',
                   'command': ['run_mammoth.py', '--method', 'l2p', '--dataset', 'core50', '--seed', '0']})
    # Start the principal comparisons before the additional FTN variants.
    return [result[0], *result[6:], *result[1:6]]


def release():
    evidence = []
    def read(*parts):
        path = bulk(*parts)
        data = json.loads(path.read_text())
        evidence.append(str(path))
        return data
    assert bulk('data', 'core50', 'ready.json').exists()
    assert bulk('data', 'features', 'core50_vit_run0.pt').exists()
    for variant in VARIANTS:
        assert read('validation', 'isolation', variant + '.json')['status'] == 'pass'
    base = read('runs', 'ftn', 'core50', 'features', 'Base', 'seed0_timing', 'result.json')
    der = read('runs', 'mammoth', 'core50', 'DER++-500-features-h256', 'seed0_timing', 'result.json')
    l2p = read('runs', 'mammoth', 'core50', 'l2p', 'seed0_timing', 'result.json')
    for data in [base, der, l2p]:
        assert data['status'] == 'complete' and data['timing'] and data['smoke']
    assert base['support_scope'] == 'evaluated block only'
    assert base['optimizer_policy']['weight_decay'] == 0
    assert base['optimizer_policy']['reconfiguration_resets'] > 0
    assert base['trainable_parameters'] // 16 == der['trainable_parameters'] == 1719040
    assert len(der['epoch_timings']) == len(l2p['epoch_timings']) == 2
    for rel, entry in json.loads((WORKSPACE / 'sources/manifest.json').read_text()).items():
        assert hashlib.sha256((WORKSPACE / 'sources' / rel).read_bytes()).hexdigest() == entry['sha256']
    write_json(bulk('manifests', 'core50_first_comparison.json'), {'jobs': jobs(), 'seeds': [0],
               'ftn_support': 'per evaluated block, disjoint training holdout',
               'der_head': 'matches FTN Base active parameters, not its entire bank'})
    write_json(bulk('validation', 'core50_full.json'), {'status': 'pass', 'evidence': evidence,
               'harness_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in WORKSPACE.glob('*.py')}})
    print('Released 9 full CORe50 runs: six FTN variants, two active-budget DER++ runs, L2P.')


if __name__ == '__main__':
    release()
