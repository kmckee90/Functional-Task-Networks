"""Evaluate each lifecycle stage before atomically replacing its sole checkpoint.

The original data, loss, Bayesian inference, IC, repair, replay and generation
implementations are imported unchanged. Evaluation is isolated from training RNG.
Retries resume at the most recent completed training block; an interrupted block
is replayed from that checkpoint and its incomplete training log is truncated.
"""
import argparse
import copy
import json
import math
import time
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from ..model import InputPrior, transport_batch
from ..run import Replay
from ..identity_distinct.run import evaluate
from ..identity_distinct.metrics import IdentityMetric
from ..identity_distinct.representative.sampling import storage_steps
from .common import ROOT, ART, ASSETS, DATA_ROOT, atomic, model, data, names, generate


def write_checkpoint(folder, ck):
    # Fixed temporary filename, never timestamped or per-stage model snapshots.
    temp = folder/'latest.tmp'
    torch.save(ck, temp)
    temp.replace(folder/'latest.pt')


def quality(m, vae, metric, row, out, records, final, repaired=False):
    cache = {}
    done = {(r['stage'], r['domain'], r['mode']) for r in records}
    for n in names():
        modes = ['ic_direct4_d0', 'ic_refine4_d0'] if repaired else ['cold4_d0']
        if final and not repaired:
            modes += ['cold2_d0', 'cold16_d0', 'ic_direct4_d0', 'ic_refine4_d0']
            if 'stored' in row['domains'][n]:
                modes.append('stored')
        for mode in modes:
            key = (row['stage'], n, mode)
            sig = tuple(row['domains'][n][mode]['mask'])
            if key in done:
                cache[sig] = next(r for r in records if (r['stage'], r['domain'], r['mode']) == key)
                continue
            if sig in cache:
                r = dict(cache[sig], domain=n, mode=mode)
            else:
                images = generate(m, vae, torch.tensor(sig, device='cuda'), count=64 if final else 32)
                grid = f'{row["stage"]}_{n}_{mode}.png'
                save_image(images[:16], out/grid, nrow=8)
                r = dict(stage=row['stage'], domain=n, mode=mode, mask=list(sig),
                         grid=grid, metrics=metric(images), copy_grids={})
                sampledir = out/'samples'/f'{row["stage"]}_{n}_{mode}'
                sampledir.mkdir(parents=True, exist_ok=True)
                for i, im in enumerate(images):
                    save_image(im, sampledir/f'{i:04d}.png')
                for person in names():
                    nearest = r['metrics']['targets'][person]['nearest_training_images'][:8]
                    pairs = torch.stack([im for j, path in enumerate(nearest)
                                         for im in [images[j], to_tensor(Image.open(DATA_ROOT/path))]])
                    copygrid = f'{row["stage"]}_{n}_{mode}_nearest_{person}.png'
                    save_image(pairs, out/copygrid, nrow=8)
                    r['copy_grids'][person] = copygrid
                cache[sig] = r
            records.append(r)
            atomic(out/'quality.json', records)
            print('QUALITY', key, r['metrics']['targets'][n]['accepted_identity_rate'], flush=True)


def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def stage_eval(m, ds, ck, out, do_repair, final):
    stage = ck['stage']
    marker = out/f'{stage}_complete.json'
    if marker.exists():
        return
    state = m.get_reconfig_state()
    controller_mask = m.controller.mask
    versions = [p._version for p in m.parameters()]
    cpu, gpu = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        router = InputPrior()
        router.entries = copy.deepcopy(ck['router'])
        hist = load_json(out/'lifecycle.json', [])
        existing = next((r for r in hist if r['stage'] == stage), None)
        row = existing if existing is not None else evaluate(m, ds, router, ck['saved'], stage)
        if existing is None:
            hist.append(row)
            atomic(out/'lifecycle.json', hist)
        vae = AutoencoderKL.from_pretrained(ASSETS/'vae').cuda().eval().requires_grad_(False)
        metric = IdentityMetric()
        quality(m, vae, metric, row, out, load_json(out/'quality.json', []), final)
        if do_repair and ck['completed']:
            repair_out = out/'repair'
            repair_out.mkdir(exist_ok=True)
            # This small state contains projectors and logits, never LoRA weights.
            statefile = repair_out/'state.pt'
            saved_repair = torch.load(statefile, map_location='cpu', weights_only=False) if statefile.exists() else None
            repaired = InputPrior()
            if saved_repair and saved_repair['completed'] == ck['completed']:
                repaired.entries = saved_repair['entries']
                repaired_row = saved_repair['row']
            else:
                assert saved_repair is None or saved_repair['completed'] == ck['completed']-1
                repaired.entries = copy.deepcopy(saved_repair['entries']) if saved_repair else []
                repaired.entries.append(copy.deepcopy(ck['router'][-1]))
                repairs = repaired.repair_from_replay(m, ck['memory'],
                    live_projector=repaired.entries[-1]['projector'], fit_rows=32,
                    validation_rows=32, steps=100, restarts=3,
                    seed=ck['seed']+7401+ck['completed']-1)
                for r in repairs:
                    r['routed_task_counts_audit_only'] = {
                        n: sum(ck['memory_domains'][i] == n for i in r['routed_batches']) for n in names()}
                    if 'fit_hashes' in r:
                        assert not set(r['fit_hashes']) & set(r['validation_hashes'])
                repaired_row = evaluate(m, ds, repaired, ck['saved'], stage)
                repaired_row['repairs'] = repairs
                for n in names():
                    for mode in ['cold2_d0', 'cold4_d0', 'cold16_d0']:
                        assert repaired_row['domains'][n][mode]['mask'] == row['domains'][n][mode]['mask']
                temp = repair_out/'state.tmp'
                torch.save(dict(completed=ck['completed'], entries=repaired.entries, row=repaired_row), temp)
                temp.replace(statefile)
            rh = load_json(repair_out/'lifecycle.json', [hist[0]])
            if not any(r['stage'] == stage for r in rh):
                rh.append(repaired_row)
            atomic(repair_out/'lifecycle.json', rh)
            quality(m, vae, metric, repaired_row, repair_out,
                    load_json(repair_out/'quality.json', []), final, repaired=True)
        del vae, metric
    m.set_reconfig_state(state)
    m.controller.mask = controller_mask
    assert versions == [p._version for p in m.parameters()]
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(gpu, torch.cuda.get_rng_state())
    for key in ['m', 'mask_logits']:
        assert torch.equal(state[key], m.get_reconfig_state()[key])
    atomic(marker, dict(complete=True, weights_unchanged=True, rng_restored=True, posterior_restored=True))
    print('STAGE_COMPLETE', stage, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--replay', action='store_true')
    p.add_argument('--plain', action='store_true')
    p.add_argument('--only')
    a = p.parse_args()
    folder, out = ROOT/'runs'/a.name, ART/a.name
    folder.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock=(folder/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists():
        return
    cfg = dict(vars(a), H=1 if a.plain else 1024, k=1 if a.plain else 256,
               rank=256 if a.plain else 1, alpha=.25 if a.plain else 64., steps=1500)
    atomic(out/'config.json', cfg)
    m, ds = model(a.seed, a.plain), data()
    if a.plain:
        m.m.fill_(1)
    from .common import FILTERS
    for domain, split in ds.items():
        if domain not in FILTERS:
            assert all((v['y'] == 1000).all() for v in split.values())
        else:
            assert all(((v['y'] >= 0) & (v['y'] < 1000)).all() for v in split.values())
    order = [a.only] if a.only else names()
    replay, router, saved = Replay(a.seed+1000), InputPrior(), {}
    memory, memory_domains = [], []
    dg = torch.Generator(device='cuda').manual_seed(a.seed+20)
    tg = torch.Generator(device='cuda').manual_seed(a.seed+10)
    mu, var, since, start = None, 0., 0, 0
    ck = None
    if (folder/'latest.pt').exists():
        ck = torch.load(folder/'latest.pt', map_location='cpu', weights_only=False)
        assert ck['config'] == cfg
        m.load_adapter(ck['adapter'])
        router.entries, saved, start = ck['router'], {n:v.cuda() for n,v in ck['saved'].items()}, ck['completed']
        memory = [tuple(v.cuda() for v in entry) for entry in ck['memory']]
        memory_domains = ck['memory_domains']
        replay.entries = memory if a.replay else []
        replay.rng.bit_generator.state = ck['replay_rng']
        replay.generator.set_state(ck['replay_generator'])
        replay.hits, replay.attempts = ck['replay_hits'], ck['replay_attempts']
        dg.set_state(ck['dg']); tg.set_state(ck['tg'])
        torch.set_rng_state(ck['cpu_rng']); torch.cuda.set_rng_state(ck['cuda_rng'])
        mu, var, since = ck['detector']
        ck['memory'], ck['saved'] = memory, saved
    def checkpoint(completed, stage):
        obj = dict(config=cfg, seed=a.seed, stage=stage, completed=completed,
                   adapter=m.adapter_state(), router=router.entries, saved=saved,
                   memory=memory, memory_domains=memory_domains,
                   replay_rng=replay.rng.bit_generator.state,
                   replay_generator=replay.generator.get_state(), replay_hits=replay.hits,
                   replay_attempts=replay.attempts, dg=dg.get_state(), tg=tg.get_state(),
                   cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                   detector=(mu,var,since))
        write_checkpoint(folder, obj)
        return obj
    if ck is None:
        ck = checkpoint(0, 'initial')
    stage_eval(m, ds, ck, out, not a.plain and not a.only, start == len(order))
    del ck
    # Retries overwrite incomplete block logs, rather than appending duplicates.
    logpath = out/'train.jsonl'
    if logpath.exists():
        rows = [line for line in logpath.read_text().splitlines() if json.loads(line)['block'] < start]
        logpath.write_text(''.join(line+'\n' for line in rows))
    with logpath.open('a', buffering=1) as log:
        for block in range(start, len(order)):
            n = order[block]
            train = ds[n]['train']
            schedule = set(storage_steps(a.seed, block))
            opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-4)
            tic = time.time()
            for step in range(1500):
                ids = torch.randint(len(train['z']), (16,), device='cuda', generator=dg)
                z,y = train['z'][ids], train['y'][ids]
                # Preserve the filter protocol's classifier-free dropout.
                # Null-conditioned faces/objects retain the original zero rate.
                x,target = transport_batch(z,y,tg,dropout=.1 if n in FILTERS else 0.)
                with torch.no_grad():
                    observed = m.loss(m(x),target).item()
                threshold = None if mu is None else mu+3*math.sqrt(max(var,0))
                cp = not a.plain and (mu is None or (since >= math.ceil(1/max(1-.9,1e-8)) and observed > threshold))
                if not a.plain:
                    if cp:
                        m.reconfigure_changepoint(x,target); since=0
                    else:
                        m.reconfigure(x,target)
                opt.zero_grad(set_to_none=True)
                loss = m.loss(m(x),target); value=loss.item()
                if cp or mu is None:
                    mu,var=value,0.
                else:
                    diff=value-mu; mu,var=.9*mu+(1-.9)*value,.9*var+(1-.9)*diff*diff
                since += 1
                if step in schedule:
                    entry=(z[:4].clone(),y[:4].clone(),m.m.clone())
                    memory.append(entry); memory_domains.append(n)
                    if a.replay:
                        replay.entries=memory
                loss.backward()
                rep=replay.backward(m) if a.replay else None
                opt.step()
                assert math.isfinite(value)
                r=dict(block=block,domain=n,step=step,loss=value,detector_loss=observed,
                       threshold=threshold,changepoint=cp,replay_loss=rep,
                       mask=m.m.cpu().tolist(),replay_admitted=step in schedule,
                       source_image_indices=ids[:4].cpu().tolist() if step in schedule else None)
                log.write(json.dumps(r)+'\n')
                if step%100 == 0:
                    print('TRAIN',a.name,n,step,value,'CP',cp,'seconds',time.time()-tic,flush=True)
                    atomic(out/'progress.json',dict(phase='train',block=block,person=n,step=step,seconds=time.time()-tic))
            opt.zero_grad(set_to_none=True)
            del opt
            saved[n]=m.m.clone()
            router.archive(train['z'],m.mask_logits)
            stage=f'after_{block+1}_{n}'
            ck=checkpoint(block+1,stage)
            atomic(out/'progress.json',dict(phase='evaluate',stage=stage))
            stage_eval(m,ds,ck,out,not a.plain and not a.only,block==len(order)-1)
            del ck
    assert len(list(folder.glob('*.pt'))) == 1
    atomic(out/'complete.json',dict(complete=True,blocks=len(order),replay_hits=replay.hits,
           replay_attempts=replay.attempts,checkpoint_bytes=(folder/'latest.pt').stat().st_size,
           trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad)))
    print('RUN_COMPLETE',a.name,flush=True)


if __name__ == '__main__':
    from .evaluate import stage_eval
    main()
