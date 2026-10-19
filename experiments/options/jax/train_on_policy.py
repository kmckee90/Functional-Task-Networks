"""On-policy bootstrap on top of train_options_2.py.

train_options_2.py trains every option's mask on ONE shared buffer of OFF-policy
random-walk transitions. This script asks: what if each option first gathers its
OWN on-policy data before training? Loading that checkpoint, then for each option
in turn (one epoch = one pass over all options):

  1. ACTIVATE the option's mask (the one learned by train_options_2.py).
  2. COLLECT fresh transitions under THAT mask's epsilon-greedy policy -- many
     parallel episodes that reset on reaching the option's centroid or timing out
     (collect.collect_on_policy). Reward is mapped to centroid membership exactly
     as in training; the collector just gathers (obs, act, next_obs, boundary).
  3. TRAIN the mask with DDPG on that on-policy buffer (agent.train_option, same
     masked-subspace + sibling-consolidation step as train_options_2.py).
  4. Repeat for the next option.

The behavioral graph and centroids are reused as-is (on-policy data is goal-
directed, not exploratory, so it is NOT used to rebuild the graph). The result is
written to a SEPARATE checkpoint so it can be compared against the off-policy one
(point demo.py at it with --checkpoint checkpoint_on_policy.pkl).

    uv run --project ../.. python train_on_policy.py
    uv run --project ../.. python train_on_policy.py --epochs 5 --epsilon 0.1
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import jax

import collect as collect_mod
from agent import OptionsAgent, adam_init
from color_world import ColorWorld
from config import OnPolicyConfig

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _fmt_dt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:d}m{s:02d}s" if m else f"{s:d}s"


def load_agent(ckpt, option_train_steps, seed):
    """Reconstruct an OptionsAgent from a train_options_2.py checkpoint, ready to
    keep training (fresh Adam state -- optimizer moments aren't checkpointed)."""
    cfg = ckpt["cfg"]
    agent = OptionsAgent(cfg, seed=seed, option_train_steps=option_train_steps)
    tm = jax.tree_util.tree_map
    import jax.numpy as jnp
    agent.actor = tm(jnp.asarray, ckpt["actor"])
    agent.critic = tm(jnp.asarray, ckpt["critic"])
    agent.actor_t = tm(jnp.asarray, ckpt["actor_t"])
    agent.critic_t = tm(jnp.asarray, ckpt["critic_t"])
    agent.actor_opt = adam_init(agent.actor)
    agent.critic_opt = adam_init(agent.critic)
    agent.options = {int(c): {"m_a": jnp.asarray(o["m_a"]), "m_c": jnp.asarray(o["m_c"])}
                     for c, o in ckpt["options"].items()}
    agent.registry.centroids = np.asarray(ckpt["centroids"], np.float32)
    agent.registry.pos = np.asarray(ckpt["centroid_pos"], np.float32)
    return agent


def main():
    d = OnPolicyConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--steps-per-option", type=int, default=d.steps_per_option)
    p.add_argument("--workers", type=int, default=d.workers)
    p.add_argument("--epsilon", type=float, default=d.epsilon)
    p.add_argument("--episode-max-steps", type=int, default=d.episode_max_steps)
    p.add_argument("--option-train-steps", type=int, default=d.option_train_steps)
    p.add_argument("--no-consolidate", dest="consolidate", action="store_false", default=d.consolidate)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--in-checkpoint", default=d.in_checkpoint)
    p.add_argument("--graph", default=d.graph)
    p.add_argument("--out-checkpoint", default=d.out_checkpoint)
    args = p.parse_args()

    with open(os.path.join(ART, args.in_checkpoint), "rb") as f:
        ckpt = pickle.load(f)
    with open(os.path.join(ART, args.graph), "rb") as f:
        gd = pickle.load(f)
    env_kwargs = gd.get("env_kwargs", {})

    agent = load_agent(ckpt, args.option_train_steps, args.seed)
    cids = sorted(agent.options.keys())
    if not cids:
        raise SystemExit("checkpoint has no options to train.")

    # Same world, but a short per-episode horizon so the policy actually resets
    # and re-explores rather than running one long rollout.
    env = ColorWorld(**{**env_kwargs, "max_steps": args.episode_max_steps})

    print(f"on-policy[jax]: backend={jax.default_backend()} options={len(cids)} epochs={args.epochs} "
          f"steps/option={args.steps_per_option} workers={args.workers} epsilon={args.epsilon} "
          f"episode_max_steps={args.episode_max_steps} option_train_steps={args.option_train_steps} "
          f"consolidate={args.consolidate} torus={env.torus}", flush=True)

    key = jax.random.PRNGKey(args.seed)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        reach_sum = 0.0
        for i, cid in enumerate(cids, 1):
            # 1+2) on-policy collection under this option's mask (epsilon-greedy).
            key, ck = jax.random.split(key)
            arrays = collect_mod.collect_on_policy(ck, agent, cid, env,
                                                   args.steps_per_option, args.workers, args.epsilon)
            jax.block_until_ready(arrays)
            next_obs_np = np.asarray(arrays[2])  # raw color (slot 2), for membership
            reach = float((agent.registry.membership_np(next_obs_np) == cid).mean())
            reach_sum += reach
            # 3) DDPG-train on the history-AUGMENTED obs (slots 6/7; aug==raw if n=0).
            train_buf = (arrays[6], arrays[1], arrays[7], arrays[5])  # aug_obs, act, aug_next, boundary
            cl, al = agent.train_option(cid, train_buf, consolidate=args.consolidate)
            if i % 5 == 0 or i == len(cids):
                dt = time.time() - t_ep
                print(f"    [ep{epoch} opt {i}/{len(cids)} C{cid}] reach={reach:.2f} "
                      f"critic_loss={cl:.4f} actor_loss={al:.4f}  elapsed {_fmt_dt(dt)}", flush=True)
        print(f"  [epoch {epoch}/{args.epochs}] trained {len(cids)} options on-policy in "
              f"{_fmt_dt(time.time() - t_ep)}  mean_reach={reach_sum / len(cids):.2f}", flush=True)

    ckpt_out = {
        "actor": jax.device_get(agent.actor), "critic": jax.device_get(agent.critic),
        "actor_t": jax.device_get(agent.actor_t), "critic_t": jax.device_get(agent.critic_t),
        "options": {c: {k: np.asarray(v) for k, v in o.items()} for c, o in agent.options.items()},
        "centroids": agent.registry.centroids, "centroid_pos": agent.registry.positions_np(),
        "cfg": agent.cfg,
    }
    out_path = os.path.join(ART, args.out_checkpoint)
    with open(out_path, "wb") as f:
        pickle.dump(ckpt_out, f)
    print(f"done in {_fmt_dt(time.time() - t0)}. wrote {out_path}\n"
          f"  demo it with: uv run --project ../.. python demo.py --checkpoint {args.out_checkpoint}",
          flush=True)


if __name__ == "__main__":
    main()
