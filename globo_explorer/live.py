"""Live actor-learner runtime for `play.py --train`.

Watch the explorer KEEP LEARNING in real time. For this agent training and behaviour
are inseparable -- RECODE spawns/merges centroids, the goal/frontier prioritized replay
reshapes the value heads, epsilon anneals -- so this runs the *whole* train.py machinery,
just split across two GPUs so the live feed never freezes:

  * Actor (inference GPU): steps the env, acts with the current behaviour weights, evolves
    the RECODE density LIVE (update + decay every step -- the thing that drives exploration),
    dumps centroids, and streams the FPV. Never blocks on gradient updates.
  * Learner (a second GPU): drains the actor's transitions into the replay buffer and runs
    the real prioritized train_round() asynchronously, pushing fresh q_net/q_env weights back
    to the actor after every round.

Ape-X / SEED-style semantics: the watched policy lags the learner by ~one training round,
and the learner trains on a per-round snapshot of the actor's live density. This is faithful
to "watch behaviour change as optimization happens" -- it is NOT bit-identical to synchronous
train.py (stale-by-one-round weights), which is inherent to async, non-blocking learning.

Nothing in train.py is modified; its build_agent()/save_checkpoint() are imported as-is. If
infer_device == learn_device this still works (single GPU): the learner thread overlaps with
the actor because env.step() and CUDA kernels both release the GIL, though heavy rounds may
then briefly slow the feed.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from copy import copy

import torch

from env import downsample_obs, make_globo_env, obs_to_chw, resolve_render_size
from train import CENTROIDS, HERE, build_agent, save_checkpoint


def _cpu_sd(sd: dict) -> dict:
    return {k: v.detach().cpu() for k, v in sd.items()}


def run_live(args, *, infer_device, learn_device, frame_sink, should_run, ckpt=None):
    """Run the actor loop in the calling thread, with a learner thread on learn_device.

    args         a config namespace (all default.yaml keys + .name/.config), as train.py uses.
    frame_sink   callable(obs) -> None; called every step with the rendered FPV to stream it.
    should_run   callable() -> bool; return False to stop (e.g. on server shutdown).
    ckpt         optional loaded checkpoint dict to warm-start from (continues its eps schedule).
    """
    factored = args.action_mode == "factored"
    max_timesteps = None if args.max_steps <= 0 else args.max_steps
    obs_w, obs_h = int(args.obs_width), int(args.obs_height)
    render_w, render_h = resolve_render_size(obs_w, obs_h, args.render_width, args.render_height)

    env = make_globo_env(
        obs_width=render_w, obs_height=render_h,   # engine renders high; agent input is downsampled below
        max_timesteps=max_timesteps, init_frames=args.init_frames, seed=args.seed,
        action_mode=args.action_mode,
        invincible=args.invincible, freeze_daylight=args.freeze_daylight, spawn=args.spawn,
        weather_seed=args.weather_seed, frameskip=args.frameskip,
        move_speed=args.move_speed, move_accel=args.move_accel, climb_speed=args.climb_speed,
        jump_speed=args.jump_speed, gravity=args.gravity, mouse_mov=args.mouse_mov,
        env_dir=os.path.join(HERE, "craftium_live_env"),
        run_dir_prefix=os.path.join(HERE, "runs", "live"),
    )

    # Two agents from the same config+seed: identical frozen reservoir/feedback and identical
    # initial q-weights. Only the learner's replay buffer/optimizer are exercised; the actor's
    # buffer stays lazily unallocated (it never calls remember()).
    a_actor = copy(args); a_actor.device = str(infer_device)
    a_learn = copy(args); a_learn.device = str(learn_device)
    actor = build_agent(a_actor, env)
    learner = build_agent(a_learn, env)
    if ckpt is not None:
        learner.load_state_dict(ckpt["agent"])
        actor.load_state_dict(ckpt["agent"], load_optimizer=False)
    print(f"[live] actor@{infer_device}  learner@{learn_device}  factored={factored} "
          f"obs={obs_w}x{obs_h} render={render_w}x{render_h} warm_start={ckpt is not None}", flush=True)

    q: "queue.Queue[dict]" = queue.Queue()
    density_lock = threading.Lock()   # guards the actor's live density (actor R/W vs learner snapshot)
    weights_lock = threading.Lock()   # guards the staged weight hand-off slot
    slot = {"sd": None, "v": 0}       # latest learner weights for the actor to pick up
    applied = {"v": 0}
    stop = threading.Event()
    online_coef = float(args.online_coef); env_reward_coef = float(args.env_reward_coef)
    name = args.name

    # ---------------- learner thread (learn_device) ----------------
    def learner_loop():
        ingested = last_train = last_ckpt = rounds = 0
        ckpt_every = int(args.checkpoint_interval or args.render_interval or 5000)
        while not stop.is_set() or not q.empty():
            drained = 0
            try:
                while drained < 2048:                 # ingest in bursts; never blocks the actor
                    it = q.get_nowait()
                    learner.remember(
                        it["z"].to(learn_device), it["action"], it["env_r"], it["online_nov"],
                        it["emb"].to(learn_device), it["z_next"].to(learn_device),
                        terminal=it["terminal"], boundary=it["boundary"])
                    ingested += 1; drained += 1
            except queue.Empty:
                pass

            ready = learner.memory.size >= max(args.warmup, args.batch)
            if ready and (ingested - last_train) >= args.train_interval:
                last_train = ingested
                # offline novelty must score against the LIVE density -> snapshot it
                with density_lock:
                    snap = {k: (v.clone() if torch.is_tensor(v) else v)
                            for k, v in actor.density.state.items()}
                learner.density.set_state({k: (v.to(learn_device) if torch.is_tensor(v) else v)
                                           for k, v in snap.items()})
                loss = learner.train_round(args.train_iters)
                rounds += 1
                with weights_lock:                    # publish fresh weights for the actor
                    slot["sd"] = {"q_net": _cpu_sd(learner.q_net.state_dict()),
                                  "q_env": _cpu_sd(learner.q_env.state_dict())}
                    slot["v"] += 1
                if rounds % 25 == 0:
                    ls = "n/a" if loss is None else f"{loss:.4f}"
                    print(f"        [live learner] rounds={rounds} ingested={ingested} "
                          f"buf={learner.memory.size} goal={learner.goal.size} loss={ls}", flush=True)
                if ingested - last_ckpt >= ckpt_every:
                    last_ckpt = ingested
                    p = save_checkpoint(learner, args, ingested, prefix=name)
                    print(f"        [live ckpt] {os.path.relpath(p, HERE)}", flush=True)
            elif drained == 0:
                time.sleep(0.003)                     # idle: avoid busy-spin
        print("[live learner] stopped", flush=True)

    def apply_weights():
        with weights_lock:
            if slot["sd"] is None or slot["v"] == applied["v"]:
                return
            sd = slot["sd"]; applied["v"] = slot["v"]
        actor.q_net.load_state_dict(sd["q_net"])
        actor.q_env.load_state_dict(sd["q_env"])

    # ---------------- actor (this thread, infer_device) ----------------
    def encode_reset(obs):
        actor.reset_hidden()
        img = obs_to_chw(downsample_obs(obs, obs_w, obs_h), device=infer_device)
        with density_lock:
            nov = actor.intrinsic_reward(img)
        z = actor.encode(img, nov)
        with density_lock:
            actor.update_density(img)
        return z

    lt = threading.Thread(target=learner_loop, name="live-learner", daemon=True)
    lt.start()

    obs, _ = env.reset(seed=args.seed)
    frame_sink(obs)
    z = encode_reset(obs)
    global_step = 0; t_start = time.time()
    while should_run():
        apply_weights()                                # pick up the latest learner weights
        action = actor.act(z)                          # epsilon-greedy at the (annealing) eps
        env_action = action if factored else int(action[0])
        next_obs, env_r, term, trunc, info = env.step(env_action)
        frame_sink(next_obs)                                   # stream the full render-res frame
        nimg = obs_to_chw(downsample_obs(next_obs, obs_w, obs_h), device=infer_device)
        with density_lock:
            intr = actor.intrinsic_reward(nimg)        # novelty vs live density (read)
        emb = actor.embed(nimg)
        actor.note_action(action, online_coef * float(intr) + env_reward_coef * float(env_r))
        z_next = actor.encode(nimg, intr)
        with density_lock:
            actor.update_density(nimg)                 # spawn/merge centroids (write)
            actor.step_schedule()                      # anneal eps + decay density
        # ship the transition to the learner as CPU tensors (decouples the two GPUs)
        q.put({"z": z.detach().cpu(), "action": list(action), "env_r": float(env_r),
               "online_nov": float(intr), "emb": emb.detach().cpu(),
               "z_next": z_next.detach().cpu(), "terminal": bool(term), "boundary": bool(term or trunc)})
        z = z_next
        global_step += 1

        if args.render_interval and global_step % args.render_interval == 0:
            with density_lock:
                n = actor.dump_centroids(CENTROIDS, prefix=name)
            print(f"        [live centroids] {n} atoms -> {os.path.relpath(CENTROIDS, HERE)}/ "
                  f"(step {global_step})", flush=True)
        if global_step % 1000 == 0:
            sps = global_step / max(1e-9, time.time() - t_start)
            print(f"[live actor] step={global_step} eps={actor.epsilon:.3f} "
                  f"atoms={actor.density.size} qlen={q.qsize()} ({sps:.1f} steps/s)", flush=True)
        if term or trunc:
            obs, _ = env.reset(seed=args.seed)
            frame_sink(obs)
            z = encode_reset(obs)

    stop.set()
    lt.join(timeout=30)
    try:
        env.close()
    except Exception:
        pass
    print("[live actor] stopped; env closed", flush=True)
