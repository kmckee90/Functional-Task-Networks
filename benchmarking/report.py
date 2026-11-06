"""Regenerate small comparison tables from completed production runs only."""
import argparse
import datetime
import json
import statistics
import time

from jobs import jobs,SEEDS
from ftn_models import VARIANTS
from storage import WORKSPACE,bulk

DATASETS=['cifar10','cifar100','tinyimagenet','core50']
TITLES=['Split CIFAR-10','Split CIFAR-100','Tiny-ImageNet','CORe50 NC']


def load_results():
    rows=[]; failures=[]
    for job in jobs():
        path=bulk('runs',job['id'],'result.json')
        if path.exists():
            d=json.loads(path.read_text())
            if d.get('status')=='complete' and not d.get('smoke'):
                rows.append(d)
        fail=bulk('runs',job['id'],'failure.json')
        if fail.exists(): failures.append(job['id'])
    return rows,failures


def cell(data,method,dataset,metric,mode=None,field='final_accuracy'):
    vals=[]
    for d in data:
        if d['method']!=method or d['dataset']!=dataset: continue
        if mode is not None and d.get('mode')!=mode: continue
        actual_metric=metric
        if metric=='task_support_metrics' and d.get('mode') not in ('pixels','vit'):
            actual_metric='task_il_metrics'
        if actual_metric in d and field in d[actual_metric]: vals.append(d[actual_metric][field]*100.)
    if not vals: return 'Pending (0/5)'
    mean=statistics.mean(vals)
    sd=statistics.stdev(vals) if len(vals)>1 else 0.
    value=f'{mean:.2f} ± {sd:.2f}' if len(vals)>1 else f'{mean:.2f}'
    return value if len(vals)==len(SEEDS) else value+f' (n={len(vals)}/5; provisional)'


def table(data,entries,metric):
    text=['| Method | '+' | '.join(TITLES)+' |','|---|'+'---:|'*len(TITLES)]
    for method,mode in entries:
        label=method+(' (ViT)' if mode=='vit' else '')
        text.append('| '+label+' | '+' | '.join(cell(data,method,ds,metric,mode) for ds in DATASETS)+' |')
    return '\n'.join(text)+'\n'


def generate():
    data,failed=load_results()
    count=len(jobs())
    status='COMPLETE' if len(data)==count and not failed else 'IN PROGRESS — final comparison is not yet available'
    text=['# FTN reviewer benchmark comparison',
          '\n**Final test accuracy (%) — native FTN MLPs and scratch replay baseline**\n',
          table(data,[(v,'pixels') for v in VARIANTS]+[('DER++-500',None),('DER++-matched',None)],'class_il_metrics'),
          '\n**Final test accuracy (%) — shared pretrained ViT-B/16**\n',
          table(data,[(v,'vit') for v in ['LoRA','LoRA_Reactive_Low_NoReplay','BayesLoRA']]+[('l2p',None),('dualprompt',None)],'class_il_metrics'),
          '\n**Final test accuracy (%) — native task-incremental protocols**\n',
          table(data,[(v,'pixels') for v in VARIANTS]+[('SPG',None),('MNTDP-D',None)],'task_support_metrics'),
          f'\nStatus: **{status}**. Completed production runs: **{len(data)}/{count}**; failures requiring inspection: **{len(failed)}**. '
          f'Updated {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}.\n',
          '\nCells show mean ± sample standard deviation across seeds 0–4. Provisional cells identify their completed seed count. '
          'Smoke tests are excluded. TIL baselines use oracle task identity; the FTN rows in the TIL table use task-matched labeled support and the known class group. '
          'They receive different inference information; this is disclosed rather than treated as an identical task-agnostic protocol.\n',
          '\n## Metrics and resource comparisons\n',
          '| Dataset | Method / backbone | Seeds | Forgetting (pp) | BWT (pp) | Training s | Total wall s | Parameters | Peak allocated GPU GiB |\n'
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    groups={}
    for d in data:
        key=(d['dataset'],d['method'],d.get('mode','native'))
        groups.setdefault(key,[]).append(d)
    for (ds,method,mode),runs in sorted(groups.items()):
        met='class_il_metrics' if 'class_il_metrics' in runs[0] else 'task_il_metrics'
        def mean_field(name,default=0.): return statistics.mean(r.get(name,default) for r in runs)
        fm=statistics.mean(r[met]['forgetting']*100 for r in runs)
        bwt=statistics.mean(r[met]['bwt']*100 for r in runs)
        train=[]
        for r in runs:
            block=r.get('training_metrics',{}).get('avg_task_block_wall_s')
            if block is not None: train.append(block*r['n_tasks'])
        train_value=f'{statistics.mean(train):.1f}' if train else 'See native logs'
        text.append(f'| {ds} | {method} / {mode} | {len(runs)} | {fm:.2f} | {bwt:.2f} | {train_value} | {mean_field("wall_seconds"):.1f} | {mean_field("total_parameters"):.0f} | {mean_field("peak_gpu_memory_bytes")/2**30:.2f} |')
    if not groups: text.append('| Pending | Production runs have not completed | 0 | — | — | — | — | — | — |')
    text.append('\nFTN training seconds come from the native block timer, including mask search, data loading, backward, replay, and optimizer update. '
                'Total wall time also includes complete query evaluations, support inference diagnostics, and the final inference checkpoint. '
                'FTN evaluates additional routing diagnostics, so its total wall time is not a direct training-speed comparison with a baseline. '
                'Checkpoint/cached-data download time before training is excluded.\n')
    if failed:
        text+=['\n## Runs requiring attention\n']+['- `'+x+'`' for x in failed]
    methodology=(WORKSPACE/'METHODOLOGY.md').read_text()
    text.append('\n'+methodology)
    report='\n'.join(text)+'\n'
    # The only generated artifact copied into sync is this bounded small report.
    if len(report.encode())>2*1024*1024:
        raise RuntimeError('Refusing to sync an unexpectedly large report.')
    remote=bulk('reports','REPORT.md')
    remote.write_text(report)
    local=WORKSPACE/'REPORT.md'
    temp=local.with_suffix('.md.tmp')
    temp.write_text(report)
    temp.replace(local)
    return len(data)==count and not failed


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    while True:
        done=generate()
        if done or not a.watch: break
        time.sleep(30)
