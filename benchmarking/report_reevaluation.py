"""Summarize final-checkpoint domain-support reevaluations; no smoke results."""
import json

from ftn_models import VARIANTS
from storage import WORKSPACE, bulk


def main():
    lines = ['# FTN TinyImageNet: final-checkpoint domain-support evaluation', '',
             '| FTN variant | Domain-support accuracy | Earlier pooled-support accuracy |',
             '|---|---:|---:|']
    runs = []
    for variant in VARIANTS:
        run = f'tinyimagenet/features/{variant}/seed0'
        path = bulk('reevaluations', 'domain_support', run, 'result.json')
        if not path.exists():
            lines.append(f'| {variant} | Pending | — |')
            continue
        result = json.loads(path.read_text())
        assert result['status'] == 'complete' and result['weights_unchanged']
        assert len(result['blocks']) == 10
        original = json.loads(bulk('runs', 'ftn', run, 'result.json').read_text())
        lines.append(f'| {variant} | {100*result["final_accuracy"]:.2f}% | {100*original["class_il_metrics"]["final_accuracy"]:.2f}% |')
        runs.append(result)
    lines += ['', 'These are single-seed results using the saved final training weights and frozen ViT features. '
              'They are not results for adapters inserted into ViT itself.', '',
              'Each of the ten learned blocks receives its own class-balanced support set of 256 examples '
              'from the original 10% training holdout. Support was excluded from weight training. '
              'A fresh mask is searched using the native evaluation procedure (60 steps for the root FTN variants; '
              '30 for the Bayesian variants), then fixed for that block’s 1,000 separate test images. '
              'All 200 output classes remain available. No task ID or query label enters the model or mask search. '
              'The evaluator groups support and queries by learned block.', '',
              'Accuracy is the mean over all ten block test accuracies after the final training block. '
              'This does not reconstruct earlier training phases. Every model parameter was checked for exact '
              'equality with its checkpoint after evaluation. Support indexes and resulting masks are saved on the shared PVC.', '',
              'These FTN evaluations use labeled support at inference. Earlier L2P and DER++ results do not use '
              'that additional support, so the protocols supply different inference information. '
              'The DER++ feature heads also have 16 times FTN Base’s per-input active parameter count; '
              'the existing results are not matched comparisons of active parameter budgets.', '',
              '| Block | ' + ' | '.join(r['method'] for r in runs) + ' |',
              '|---|' + '---:|' * len(runs)]
    for block in range(10):
        lines.append(f'| {block+1} | ' + ' | '.join(f'{100*r["blocks"][block]["accuracy"]:.2f}%' for r in runs) + ' |')
    lines += ['', 'Remote artifacts: `/mnt/obelisk-shared/benchmarking/kmckee90/neurips_ftn_2026/reevaluations/domain_support/tinyimagenet/features/`.']
    report = '\n'.join(lines) + '\n'
    (WORKSPACE / 'FTN_DOMAIN_SUPPORT_RESULTS.md').write_text(report)
    bulk('reports', 'FTN_DOMAIN_SUPPORT_RESULTS.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
