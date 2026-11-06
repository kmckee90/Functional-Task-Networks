"""Maintain a small report from the declared full CORe50 runs only."""
import argparse
import datetime
import json
import time

from core50_jobs import jobs
from storage import WORKSPACE, bulk


def generate():
    rows = []
    complete = 0
    failures = []
    details = []
    for job in jobs():
        result = bulk('runs', job['id'], 'result.json')
        failure = bulk('runs', job['id'], 'failure.json')
        label = job['id'].replace('ftn/core50/features/', 'FTN ').replace('mammoth/core50/', '').replace('/seed0', '')
        label = label.replace('-features-h256', ' (256-atom head)')
        if result.exists():
            data = json.loads(result.read_text())
            assert data['status'] == 'complete' and not data['smoke']
            if job['id'].startswith('ftn/'):
                assert data['support_scope'] == 'evaluated block only'
                assert data['optimizer_policy']['weight_decay'] == 0
            metric = data['class_il_metrics']
            rows.append(f'| {label} | {100*metric["final_accuracy"]:.2f}% | {100*metric["final_micro_accuracy"]:.2f}% | Complete |')
            complete += 1
            details.append(f'| {label} | {data["wall_seconds"]:.1f} | {data["trainable_parameters"]:,} | {data["total_parameters"]:,} |')
        else:
            state = 'Needs inspection' if failure.exists() else 'Running / queued'
            rows.append(f'| {label} | — | — | {state} |')
        if failure.exists():
            failures.append(job['id'])
    report = ['# CORe50 NC: FTN and continual-learning baselines', '',
              '| Method | Final mean block accuracy | Final overall test accuracy | Status |',
              '|---|---:|---:|---|', *rows, '',
              f'Completed **{complete}/9** full single-seed runs (official run / seed 0). '
              f'Updated {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}. '
              'Smoke tests and shortened timing runs are excluded.', '',
              '## Protocol', '',
              'Official CORe50 NC: 50 classes, nine sequential blocks (10 classes, then eight blocks of 5). '
              'The original held-out sessions supply 44,972 test images. A 10% class-stratified holdout '
              'from training provides calibration support; 107,897 remaining images are eligible for weight training. '
              'All methods share that training/calibration split.', '',
              'FTN and DER++ operate on the same frozen Google ImageNet-21k ViT-B/16 features. '
              'The six FTN variants retain their native cores/configurations and training durations. '
              'L2P uses its native pretrained ViT prompt architecture. The LoRA FTN rows adapt native '
              'feature-processing networks; they do not insert adapters into ViT.', '',
              'After each FTN training block, each learned block is evaluated using its own 256 class-balanced '
              'labeled holdout support examples and separate test queries. Mask search uses the native '
              'evaluation settings (60 steps for root FTN variants; 30 for Bayesian variants). '
              'The final evaluation permits all 50 output classes. No query label or task ID is supplied to '
              'the model. The evaluator pairs support and queries from the same learned block.', '',
              'FTN uses zero weight decay and resets Adam moments at training mask reconfigurations. '
              'Gradient/moment updates are isolated to the union of masks used for current/replay inputs. '
              'The shared upstream encoder is frozen.', '',
              'DER++ uses 256 independent atoms and the same per-atom MLP/readout structure as FTN Base. '
              'Its 1,719,040 trainable head parameters match FTN Base’s K=256 active budget; '
              'FTN Base stores H=4,096 atoms overall. This match does not extend to the other FTN variants or L2P. '
              'DER++ trains for 50 epochs per block with buffer capacities 500 and 230,400 respectively. '
              'L2P trains for 5 epochs per block. Training budgets are not equalized.', '',
              'DER++ predicts directly; L2P selects prompts from the input. Neither receives labeled support '
              'at evaluation. The table therefore compares different inference information, rather than '
              'isolating the benefit of FTN’s mechanism from the benefit of support. This fixed-label benchmark '
              'does not test changing label meanings.', '',
              'Mean block accuracy equally weights the nine blocks. Overall test accuracy weights by query count; '
              'the blocks have unequal sizes. Both are measured after the final training block, not averaged '
              'over training phases.', '',
              '## Measured timing and estimates', '',
              'Precomputation completed in 2 min 30 s. Full-model timing probes used two blocks: '
              'FTN Base used 50 updates per block; baselines used one complete epoch per block. '
              'FTN Base took 1.878 s of training per 50-update block on average. '
              'L2P epochs took 66.57 s for 21,581 examples and 33.68 s for 10,793 examples. '
              'The active-budget DER++ epochs took 5.88 s and 2.82 s. '
              'The initial estimate is 35–50 min for the full first-seed comparison in parallel; '
              'this is an estimate, not a completion guarantee.', '',
              '## Completed-run resource measurements', '',
              '| Method | Wall seconds | Trainable parameters (entire model) | Total parameters |',
              '|---|---:|---:|---:|', *details, '',
              '## Artifacts', '',
              'Large files remain on `/mnt/obelisk-shared/benchmarking/kmckee90/neurips_ftn_2026/`, '
              'outside sync. Full-run results are under `runs/ftn/core50/features/` and '
              '`runs/mammoth/core50/`; feature caches under `data/features/`; '
              'FTN final checkpoints under `checkpoints/ftn/core50/features/`. '
              'The declared plan is `manifests/core50_first_comparison.json`, with validation evidence '
              'and harness hashes in `validation/core50_full.json`.', '',
              'Sources: [experiment manifest](core50_jobs.py), [FTN harness](run_ftn.py), '
              '[baseline harness](run_mammoth.py), [provenance](PROVENANCE.json).']
    if failures:
        report += ['', '## Runs requiring inspection', ''] + [f'- `{name}`' for name in failures]
    text = '\n'.join(report) + '\n'
    path = WORKSPACE / 'CORE50_REPORT.md'
    temporary = path.with_suffix('.md.tmp')
    temporary.write_text(text)
    temporary.replace(path)
    bulk('reports', 'CORE50_REPORT.md').write_text(text)
    print(f'CORe50: {complete}/9 complete; {len(failures)} failures', flush=True)
    return complete == 9 and not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        done = generate()
        if done or not args.watch:
            break
        time.sleep(30)
