"""Pod-only full allocation + populated RECODE/training capacity probe."""
import argparse
import json
import time
import torch
from .model_free import ModelFreeBehavior


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main', type=int, default=4000000)
    p.add_argument('--goal', type=int, default=500000)
    p.add_argument('--atoms', type=int, default=100000)
    a = p.parse_args()
    b = ModelFreeBehavior(replay_capacity=a.main, goal_capacity=a.goal, recode_capacity=a.atoms)
    agent = b.agent
    image = torch.rand(3, 32, 64, device='cuda')
    z = agent.encode(image, torch.tensor(0., device='cuda'))
    for buf in (agent.memory, agent.goal):
        buf._lazy(z)
        buf.z[:1024].copy_(z)
        buf.z_next[:1024].copy_(z)
        buf.emb[:1024].uniform_()
        buf.size = min(1024, buf.capacity)
        buf.ptr = buf.size
    agent.density.memory.uniform_()
    agent.density.counts.fill_(1)
    agent.density.size = a.atoms
    torch.cuda.synchronize()
    start = time.monotonic()
    loss = agent.train_round(3)
    torch.cuda.synchronize()
    train_seconds = time.monotonic()-start
    timings = {}
    for atoms in (20000, a.atoms):
        agent.density.size = atoms
        queries = torch.rand(8192, 512, device='cuda')
        torch.cuda.synchronize()
        start = time.monotonic()
        agent._novelty(queries)
        torch.cuda.synchronize()
        timings[str(atoms)] = time.monotonic()-start
    result = dict(main=a.main, goal=a.goal, atoms=a.atoms, feature_shape=list(z.shape),
        allocated_GiB=torch.cuda.memory_allocated()/2**30,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        free_GiB=torch.cuda.mem_get_info()[0]/2**30,
        training_seconds=train_seconds, query8192_seconds=timings, loss=loss)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
