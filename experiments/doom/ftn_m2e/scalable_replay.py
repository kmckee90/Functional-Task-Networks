"""Bounded-work trajectory replay and incremental, CPU-portable snapshots."""
from pathlib import Path
import os
import uuid
import torch

FIELDS = ('z', 'z_next', 'emb', 'action', 'env_r', 'online_nov', 'terminal', 'boundary', 'model_nov', 'model_step')


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


class ChunkStore:
    """Immutable dirty chunks; delete superseded files only after checkpoint commit.

    One chunk crosses PCIe at a time. Unused capacity is never serialized.
    The checkpoint owns the manifest; all paths stay outside the source tree.
    """
    chunk_size = 2048

    def __init__(self, root, buffers):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.buffers = buffers
        self.manifest = {name: {} for name in buffers}
        for buf in buffers.values():
            buf.dirty_chunks = set()
            original = buf.add
            def add(*args, _buf=buf, _original=original, **kwargs):
                chunk=_buf.ptr // self.chunk_size
                result=_original(*args, **kwargs)
                _buf.dirty_chunks.add(chunk)
                return result
            buf.add = add

    def save(self):
        result = {}
        for name, buf in self.buffers.items():
            files = self.manifest[name].copy()
            for chunk in sorted(buf.dirty_chunks):
                start = chunk * self.chunk_size
                end = min(start + self.chunk_size, buf.size)
                if end<=start:continue
                filename = f'{name}-{chunk}-{uuid.uuid4().hex}.pt'
                data = {k: getattr(buf, k)[start:end].cpu() for k in FIELDS}
                torch.save(data, self.root / filename)
                files[chunk] = filename
            self.manifest[name] = files
            buf.dirty_chunks.clear()
            result[name] = dict(size=buf.size, ptr=buf.ptr, capacity=buf.capacity, files=files)
        return result

    def load(self, saved):
        for name, meta in saved.items():
            buf = self.buffers[name]
            if meta['capacity'] != buf.capacity:
                raise ValueError('Resume with the same replay capacity as the checkpoint')
            for chunk, filename in meta['files'].items():
                data = torch.load(self.root / filename, map_location='cpu', weights_only=True)
                buf._lazy(data['z'][0])
                start = int(chunk) * self.chunk_size
                for k, v in data.items():
                    getattr(buf, k)[start:start + len(v)].copy_(v)
            buf.size, buf.ptr = meta['size'], meta['ptr']
            self.manifest[name] = meta['files'].copy()
            buf.dirty_chunks.clear()

    def committed(self):
        live = {f for files in self.manifest.values() for f in files.values()}
        for path in self.root.glob('*.pt'):
            if path.name not in live:
                path.unlink()


def trajectory(buf, endpoint, length):
    """Physical indices ending at endpoint; never cross a reset or ring's oldest row."""
    oldest = buf.ptr if buf.size == buf.capacity else 0
    age = (endpoint - oldest) % buf.capacity
    count = min(length, age + 1)
    idx = (torch.arange(count, device=buf.device) + endpoint - count + 1) % buf.capacity
    cuts = torch.nonzero(buf.boundary[idx[:-1]], as_tuple=False).flatten()
    if cuts.numel():
        idx = idx[int(cuts[-1]) + 1:]
    return idx


def install_scalable_sampling(agent):
    # Query chunks cap the distance matrix independently of replay/density capacity.
    old_novelty = agent._novelty
    def novelty(emb):
        return torch.cat([old_novelty(x) for x in emb.split(128)])
    agent._novelty = novelty
    @torch.no_grad()
    def nearest_cache():
        density = agent.density
        memory = density.memory[:density.size]
        for start in range(0, density.size, 128):
            end = min(start + 128, density.size)
            d2 = torch.cdist(memory[start:end], memory).square()
            d2[d2 == 0] = float('inf')
            d2[torch.arange(end-start, device=memory.device),
               torch.arange(start, end, device=memory.device)] = float('inf')
            density.d2_nearest[start:end] = d2.min(dim=1).values
    agent.density._update_d2_nearest = nearest_cache

    def frontier(buf):
        with torch.no_grad():
            scan = (torch.arange(buf.size, device=buf.device) if buf.size <= agent.offline_scan_max
                    else torch.randint(buf.size, (agent.offline_scan_max,), device=buf.device))
            nov = novelty(buf.emb[scan])
            pos = int(nov.argmax())
            return int(scan[pos]), float(nov[pos])

    def train_round(n_updates):
        main, goal = agent.memory, agent.goal
        if main.size < agent.minibatch_size:
            return None
        if hasattr(agent,'frontier_rotation') and os.environ.get('GLOBO_FRONTIER_ROTATION','1')!='0':
            return rotating_round(n_updates)
        endpoint, best = frontier(main)
        gp, gb = frontier(goal) if goal.size else (None, 0.)
        if best * agent.main_sparse_reward_scale >= gb * agent.goal_sparse_reward_scale:
            data = main.gather(trajectory(main, endpoint, agent.minibatch_size))
            # Distinct copied trajectories must not join, but this is not a terminal.
            data['boundary'][-1] = True
            goal.insert(data)
            gp, gb = frontier(goal)
        modes = ('goal', 'last', 'random') if agent.prioritized_replay else ('random',)
        # Display diagnostic only: the frontier used by this offline goal batch.
        goal_source=goal if gp is not None else main
        goal_index=gp if gp is not None else endpoint
        agent.display_frontier=goal_source.emb[goal_index].detach().cpu().clone()
        agent.display_frontier_step=int(goal_source.model_step[goal_index])
        losses = []
        for i in range(n_updates):
            mode = modes[i % len(modes)]
            buf = goal if mode == 'goal' and gp is not None else main
            end = (gp if buf is goal else (main.ptr - 1) % main.capacity if mode == 'last'
                   else int(torch.randint(main.size, (1,))))
            loss = agent._td_update(buf, trajectory(buf, end, agent.minibatch_size), buf is goal)
            if loss is not None:
                losses.append(loss)
        agent._last_metrics = dict(frontier_novelty=best, goal_buffer=float(goal.size))
        return sum(losses) / len(losses) if losses else None

    def rotating_round(n_updates):
        from .frontier_rotation import candidates
        main,goal=agent.memory,agent.goal;rotation=agent.frontier_rotation
        pool=[]
        for source,buf in [('memory',main),('goal',goal)]:
            if not buf.size:continue
            scan=(torch.arange(buf.size,device=buf.device) if buf.size<=agent.offline_scan_max else
                torch.randint(buf.size,(agent.offline_scan_max,),device=buf.device))
            pool.extend(candidates(agent,buf,source,scan))
        current=rotation.current
        valid=current is not None
        if valid:
            buf=getattr(agent,current['source']);idx=current['index']
            valid=idx<buf.size and int(buf.model_step[idx])==current['id']
        chosen=rotation.choose(pool,agent.cur_step,valid)
        if chosen is None:return None
        source=getattr(agent,chosen['source']);endpoint=chosen['index']
        # Copy a newly selected main-memory approach once; retain the endpoint
        # identity in the goal ring until rotation or ring eviction.
        if source is main:
            data=main.gather(trajectory(main,endpoint,agent.minibatch_size))
            data['boundary'][-1]=True;goal.insert(data)
            chosen['source']='goal';chosen['index']=(goal.ptr-1)%goal.capacity
        gp=chosen['index']
        agent.display_frontier=chosen['emb'].clone();agent.display_frontier_step=chosen['id']
        losses=[]
        modes=('goal','last','random') if agent.prioritized_replay else ('random',)
        for i in range(n_updates):
            mode=modes[i%len(modes)];buf=goal if mode=='goal' else main
            end=gp if mode=='goal' else (main.ptr-1)%main.capacity if mode=='last' else int(torch.randint(main.size,(1,)))
            loss=agent._td_update(buf,trajectory(buf,end,agent.minibatch_size),mode=='goal')
            if loss is not None:losses.append(loss)
        agent._last_metrics=dict(frontier_novelty=chosen['novelty'],goal_buffer=float(goal.size),
            **rotation.metrics(agent.cur_step))
        return sum(losses)/len(losses) if losses else None
    agent.train_round = train_round
