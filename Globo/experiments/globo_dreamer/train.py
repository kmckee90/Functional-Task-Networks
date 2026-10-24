"""Run only on the actl pod. All runtime paths are outside the source workspace."""
import argparse
import functools
import json
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
STATE = Path(os.environ.get('GLOBO_STATE', '/home/dev/globo_dreamer_state'))
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(STATE / 'cache/jax'))
sys.path.insert(0, str(STATE / 'vendor/dreamerv3'))

import elements
import embodied
import numpy as np
import ruamel.yaml

from .env import Globo, atomic_json, runtime_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='pilot')
    p.add_argument('--steps', type=int, default=0, help='0: unlimited; otherwise cumulative observations')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--world-seed', type=int, help='World generation seed; defaults to the agent seed')
    p.add_argument('--render-size', type=int, choices=(64,256), default=64)
    p.add_argument('--reduced-actions', action='store_true',
                   help='14 actions: idle, movement/look, dig and place; no inventory or sneak')
    p.add_argument('--mf-init-weights', help='Immutable weights-only warm-start file; ignored on own checkpoint resume')
    p.add_argument('--spawn', default='11,5.1,121')
    p.add_argument('--size', default='size12m')
    p.add_argument('--rssm-hidden', type=int)
    p.add_argument('--rssm-stoch', type=int)
    p.add_argument('--rssm-classes', type=int)
    p.add_argument('--encoder-depth', type=int)
    p.add_argument('--decoder-depth', type=int)
    p.add_argument('--reservoir-channels', type=int, default=64)
    p.add_argument('--reservoir-variant', choices=('legacy','tuned_v1'), default='legacy')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--length', type=int, default=32)
    p.add_argument('--train-ratio', type=float, default=16)
    p.add_argument('--world-model-only', action='store_true')
    p.add_argument('--dreamer-novelty-only', action='store_true')
    p.add_argument('--ftn', action='store_true', help='Whole-world-model BayesLoRA InputConditioned')
    p.add_argument('--plan2explore', action='store_true', help='Original embedding-disagreement objective on DreamerV3')
    p.add_argument('--p2e-reward-scale', type=float, default=10000.)
    p.add_argument('--achievements', action='store_true', help='Evaluation-only success counters; enabled only at launch')
    p.add_argument('--model-train-start', type=int, default=0)
    p.add_argument('--model-video-every', type=int, default=0)
    p.add_argument('--model-reward-weight', type=float, default=0.)
    p.add_argument('--model-feedback', action='store_true')
    p.add_argument('--model-reward-half-life', type=float, default=10000.)
    p.add_argument('--continuous-monitor', action='store_true')
    p.add_argument('--density', type=float, default=1.)
    p.add_argument('--disagreement', type=float, default=1.)
    p.add_argument('--prediction-error', type=float, default=0.)
    p.add_argument('--disagreement-power', type=float, default=1.)
    p.add_argument('--intrinsic-horizon', type=float, default=100.)
    p.add_argument('--actor-weight', type=float, default=.2)
    p.add_argument('--feedback', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--replay-value', action=argparse.BooleanOptionalAction, default=False,
                   help='Train survival value on real replay returns as well as imagination')
    p.add_argument('--episode-steps', type=int, default=0, help='0: no artificial life timeout')
    p.add_argument('--video-every', type=int, default=4000)
    p.add_argument('--smoke', action='store_true', help='Scripted engine/interface check; no claim of learned behavior')
    p.add_argument('--random', action='store_true', help='Random-action behavioral reference, separate lifetime memory')
    p.add_argument('--model-free', action='store_true', help='Existing RECODE DQN controls the shared real stream')
    p.add_argument('--model-free-only', action='store_true', help='Density explorer without any Dreamer model or replay')
    p.add_argument('--live-port', type=int, default=0, help='Optional localhost MJPEG applet for FTN-M2E')
    p.add_argument('--m2e-ftn', choices=('flat','conv'), help='Bayes FTN policy decoder; frozen reservoir retained')
    p.add_argument('--m2e-ftn-k', type=int, help='Fix active neuron count for action-space comparisons')
    p.add_argument('--frontier-ftn', choices=('frontier','waypoints'), help='Retained goal-indexed FTN policy decoders')
    p.add_argument('--model-free-device', default='cuda:0')
    p.add_argument('--mf-replay-capacity', type=int, default=4000000)
    p.add_argument('--mf-goal-capacity', type=int, default=500000)
    p.add_argument('--recode-capacity', type=int, default=20000)
    p.add_argument('--dreamer-replay-capacity', type=int, default=100000)
    p.add_argument('--checkpoint-seconds', type=float, default=900)
    p.add_argument('--mf-epsilon-end', type=float, default=.025)
    p.add_argument('--mf-gamma', type=float, default=.9)
    p.add_argument('--mf-offline-updates', type=int, default=50)
    p.add_argument('--mf-online-coef', type=float, default=1.)
    p.add_argument('--mf-offline-coef', type=float, default=1.)
    p.add_argument('--mf-frontier-scale', type=float, default=5.)
    p.add_argument('--mf-lr', type=float, default=3e-4)
    p.add_argument('--dreamer-action-probability', type=float, default=.1)
    args = p.parse_args()
    if args.mf_offline_updates < 1:p.error('--mf-offline-updates must be positive')
    if args.m2e_ftn_k is not None and (args.m2e_ftn_k <= 0 or not args.m2e_ftn or args.frontier_ftn):
        p.error('--m2e-ftn-k requires plain M2E FTN and a positive count')
    if args.m2e_ftn and (not args.model_free_only or args.mf_init_weights):
        p.error('M2E FTN requires a fresh model-free-only run (or its own checkpoint)')
    if args.frontier_ftn and args.m2e_ftn!='conv':
        p.error('Goal-indexed experiments use the matched convolutional FTN decoder')
    if args.mf_init_weights and not args.model_free_only:
        p.error('Weights-only initialization currently requires model-free-only')
    if args.plan2explore:
        if args.ftn or args.model_free or args.model_free_only or args.world_model_only:
            p.error('Plan2Explore is a standalone exploration agent')
        args.dreamer_novelty_only=True
        args.dreamer_action_probability=1.
        args.feedback=False;args.model_feedback=False
        args.density=0.;args.model_reward_weight=0.
        if not np.isfinite(args.p2e_reward_scale) or args.p2e_reward_scale<=0:
            p.error('Plan2Explore reward scale must be finite and positive')
    if args.model_free_only:
        if args.ftn or args.world_model_only or args.dreamer_novelty_only or args.model_reward_weight or args.dreamer_action_probability:
            p.error('Model-free-only requires zero Dreamer reward/actions and no Dreamer training flags')
        args.model_free=True
    if any(v is not None and v <= 0 for v in (args.rssm_hidden,args.rssm_stoch,
            args.rssm_classes,args.encoder_depth,args.decoder_depth,args.reservoir_channels)):
        p.error('Architecture dimensions must be positive')
    if args.world_model_only and (not args.model_free or args.dreamer_action_probability!=0):
        p.error('World-model-only runs require the model-free explorer and zero Dreamer action probability')
    if args.model_train_start<0 or args.model_video_every<0 or args.train_ratio<=0:
        p.error('Invalid model training schedule')
    if args.model_reward_weight<0 or args.model_reward_half_life<=0:
        p.error('Invalid model novelty reward settings')
    if args.dreamer_novelty_only and (args.model_free or args.world_model_only):
        p.error('Dreamer novelty baseline must control actions without the model-free explorer')
    if args.ftn and not args.world_model_only:
        p.error('FTN currently retains the hybrid model-only training objective')
    if args.continuous_monitor and not (args.world_model_only or args.dreamer_novelty_only or args.model_free_only):
        p.error('Continuous monitoring requires model-only or Dreamer novelty-only training')
    if not 0 <= args.mf_epsilon_end <= 1 or not 0 <= args.mf_gamma < 1:
        p.error('Invalid model-free epsilon or discount')
    if min(args.mf_online_coef, args.mf_offline_coef, args.mf_frontier_scale) < 0:
        p.error('Reward coefficients must be nonnegative')
    if min(args.density,args.disagreement,args.prediction_error,args.actor_weight)<0:
        p.error('Intrinsic reward coefficients and actor weight must be nonnegative')
    if args.disagreement_power<=0 or args.intrinsic_horizon<=1:
        p.error('Disagreement power must be positive and intrinsic horizon must exceed one')
    if args.steps < 0 or min(args.mf_replay_capacity, args.mf_goal_capacity,
                            args.recode_capacity, args.dreamer_replay_capacity,
                            args.checkpoint_seconds) <= 0:
        p.error('Capacities and checkpoint interval must be positive; steps must be nonnegative')
    if (args.smoke or args.random) and args.steps == 0:
        p.error('Scripted checks require an explicit step count')
    if not 0 <= args.dreamer_action_probability <= 1:
        p.error('dreamer-action-probability must be in [0,1]')
    if '/' in args.name or args.name in ('.', '..'):
        p.error('name must be a single directory name')
    if args.live_port and (not args.model_free_only or not args.m2e_ftn or args.frontier_ftn):
        p.error('--live-port requires --model-free-only --m2e-ftn without --frontier-ftn')
    root = runtime_path(STATE / 'runs' / args.name)
    # Single writer protects checkpoints, replay, and the one environment lifetime.
    import fcntl
    lock = (root / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    atomic_json(root / 'launch.json', dict(**vars(args), pid=os.getpid(), started=time.time()))
    env = Globo(root / 'environment', seed=args.seed if args.world_seed is None else args.world_seed, episode_steps=args.episode_steps,
                video_every=0 if args.continuous_monitor else args.video_every, spawn=args.spawn,
                bounded_artifacts=args.continuous_monitor, render_size=args.render_size,
                reduced_actions=args.reduced_actions)
    if args.achievements:env.enable_achievements()
    if args.dreamer_novelty_only:
        env.novelty_fn=lambda image:0.
        env.memory_size_fn=lambda:0
    try:
        if args.smoke or args.random:
            rng = np.random.default_rng(args.seed)
            start = time.time()
            done = True
            for i in range(args.steps):
                act = int(rng.integers(env.act_space['action'].classes)) if args.random else (1 if i % 100 < 80 else 7)
                obs = env.step(dict(reset=done, action=act))
                done = bool(obs['is_last'])
                assert all(np.isfinite(v).all() for v in obs.values())
                assert all(v.shape == env.obs_space[k].shape for k, v in obs.items())
            atomic_json(root / 'result.json', dict(kind='random' if args.random else 'scripted_smoke',
                steps=args.steps, fps=args.steps/(time.time()-start), last=env.record))
        elif args.model_free_only:
            from .model_free_only import train_model_free
            train_model_free(args,root,env)
        else:
            train(args, root, env)
    finally:
        env.close()
        lock.close()


def train(args, root, env):
    from .monitor_settings import MAX_CLIP_FRAMES
    from .agent import Agent
    if args.plan2explore:
        from .plan2explore import Plan2Explore as Agent
    from dreamerv3.main import make_replay, make_stream, make_logger
    source = STATE / 'vendor/dreamerv3/dreamerv3/configs.yaml'
    configs = ruamel.yaml.YAML(typ='safe').load(source.read_text())
    config = elements.Config(configs['defaults']).update(configs[args.size])
    overrides={key:value for key,value in (
        ('agent.dyn.rssm.hidden',args.rssm_hidden),
        ('agent.dyn.rssm.stoch',args.rssm_stoch),
        ('agent.dyn.rssm.classes',args.rssm_classes),
        ('agent.enc.simple.depth',args.encoder_depth),
        ('agent.dec.simple.depth',args.decoder_depth)) if value is not None}
    config=config.update(overrides)
    config = config.update(dict(
        logdir=str(root), seed=args.seed, task='globo_survival',
        batch_size=args.batch, batch_length=args.length, report_length=MAX_CLIP_FRAMES if args.continuous_monitor else args.length,
        replay_context=0, **{
        'logger.outputs': ['jsonl'], 'logger.filter': 'fps|loss/ensemble|loss/rew|loss/density|explore/',
        'jax.prealloc': False, 'jax.policy_devices': [0], 'jax.train_devices': [0],
        'run.envs': 1, 'run.train_ratio': args.train_ratio,
        'replay.size': args.dreamer_replay_capacity, 'replay.chunksize': 256,
        'agent.imag_length': 15, 'agent.imag_last': 8,
        'agent.repval_loss': args.replay_value, 'agent.report_gradnorms': False,
        'agent.horizon': 1000}))
    agentcfg = dict(config.agent)
    agentcfg['explore'] = dict(ensemble=5, density=args.density, disagreement=args.disagreement,
        actor_weight=args.actor_weight, feedback=args.feedback, horizon=args.intrinsic_horizon,
        prediction_error=args.prediction_error, disagreement_power=args.disagreement_power,
        model_only=args.world_model_only,novelty_only=args.dreamer_novelty_only,
        reward_feedback=(args.model_reward_weight>0 or args.model_feedback or args.dreamer_novelty_only) and not args.plan2explore,
        p2e_scale=args.p2e_reward_scale)
    jaxcfg=dict(config.jax)
    if args.ftn:jaxcfg['precompile']=False
    ac = elements.Config(**agentcfg, logdir=str(root), seed=args.seed, jax=jaxcfg,
        batch_size=args.batch, batch_length=args.length, replay_context=0,
        report_length=MAX_CLIP_FRAMES if args.continuous_monitor else args.length, replica=0, replicas=1)
    config.save(elements.Path(root / 'dreamer.yaml'))
    agent = Agent(env.obs_space, {k:v for k,v in env.act_space.items() if k != 'reset'}, ac)
    if args.ftn:
        from .ftn import BayesWorldModel
        agent=BayesWorldModel(agent,root,args.seed)
    replay = make_replay(config, 'replay')
    if args.continuous_monitor:replay.save_wait=True
    logger = make_logger(config)
    cp = elements.Checkpoint(elements.Path(root / 'ckpt'))
    cp.step, cp.agent, cp.replay = logger.step, agent, replay
    if args.world_model_only or args.dreamer_novelty_only:
        cp.model_updates = elements.Counter(0)
    live=None
    if args.continuous_monitor:
        from .continuous_monitor import ContinuousMonitor
        live=ContinuousMonitor(root/'live',agent);cp.live=live
    behavior = None
    if args.model_free:
        from .model_free import ModelFreeBehavior
        behavior = ModelFreeBehavior(args.model_free_device, args.seed,
            args.dreamer_action_probability, args.mf_replay_capacity,
            args.mf_goal_capacity, args.recode_capacity, root / 'mf_replay',
            epsilon_end=args.mf_epsilon_end, gamma=args.mf_gamma,
            online_coef=args.mf_online_coef, offline_coef=args.mf_offline_coef,
            frontier_scale=args.mf_frontier_scale, lr=args.mf_lr,
            model_reward_weight=args.model_reward_weight,model_reward_half_life=args.model_reward_half_life,
            model_feedback=args.model_feedback,reservoir_channels=args.reservoir_channels,
            reservoir_variant=args.reservoir_variant,num_actions=env.act_space['action'].classes,
            offline_updates=args.mf_offline_updates)
        env.novelty_fn = behavior.observe_frame
        env.memory_size_fn = lambda: behavior.agent.density.size
        cp.behavior = behavior
    cp.load_or_save()
    step = logger.step
    if live:live.event('process_start',int(step),pid=os.getpid())
    stream = iter(agent.stream(make_stream(config, replay, 'train')))
    carry_train = agent.init_train(args.batch)
    driver = embodied.Driver([lambda: env], parallel=False)
    driver.reset(agent.init_policy)
    agg = elements.Agg()
    batch_steps = args.batch * args.length
    due = elements.when.Ratio(args.train_ratio / batch_steps)
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    updates = int(cp.model_updates) if (args.world_model_only or args.dreamer_novelty_only) else 0
    progress = None
    if args.world_model_only and args.model_video_every:
        from .model_progress import ModelProgress
        dataset=STATE/'diagnostics'/f'config24_seed{args.seed}_canonical'/'heldout.npz'
        progress=ModelProgress(agent,dataset,root/'model_progress')
        progress.write(int(step),updates)
    next_video=(int(step)//max(1,args.model_video_every)+1)*args.model_video_every
    last_save = last_log = time.time()
    started, initial_step = time.time(), int(step)

    def on_step(transition, worker):
        nonlocal carry_train, updates
        step.increment()
        if args.ftn:agent.observe(transition)
        replay.add(transition, worker)
        live_metrics=behavior.metrics if behavior else {}
        if args.plan2explore:
            raw=float(transition['uncertainty'])
            live_metrics=dict(model_novelty=raw,model_reward=raw*args.p2e_reward_scale,
                weighted_model_novelty=raw*args.p2e_reward_scale,model_action=1.)
        elif args.dreamer_novelty_only:
            raw=float(transition['action_disagreement'][int(transition['action'])])
            novelty=float(np.clip(raw,0,1))**args.disagreement_power
            live_metrics=dict(model_novelty=raw,model_reward=novelty,
                weighted_model_novelty=novelty,model_action=1.)
        if live:live.observe(transition,env.record,int(step),live_metrics)
        if int(step) % 50 == 0 and not args.continuous_monitor:
            with (root / 'agent_feedback.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=int(step), uncertainty=float(transition['uncertainty']),
                                        density=float(transition['density'][0]))) + '\n')
        if len(replay) >= batch_steps:
            if args.world_model_only or args.dreamer_novelty_only:
                from .model_progress import target_updates
                count=max(0,target_updates(int(step),args.model_train_start,batch_steps,args.train_ratio)-updates)
            else:
                count=due(step)
            for _ in range(count):
                carry_train, outs, metrics = agent.train(carry_train, next(stream))
                if not all(np.isfinite(v).all() for v in metrics.values()):
                    raise FloatingPointError('Non-finite training metric')
                agg.add(metrics)
                updates += 1
                if args.world_model_only or args.dreamer_novelty_only:cp.model_updates.increment()
    driver.on_step(on_step)
    def policy(carry, obs, **kwargs):
        carry, actions, outputs = agent.policy(carry, obs, mode='train')
        if behavior is not None:
            import jax
            actions = behavior.policy(obs, actions,outputs.get('action_disagreement'))
            # RSSM's next observation must be conditioned on the executed action.
            carry = (*carry[:-1], {k:[jax.device_put(np.asarray(v[0]),
                carry[-1][k][0].sharding)] for k,v in actions.items()})
            agg.add({f'model_free/{k}':v for k,v in behavior.metrics.items()})
        return carry, actions, outputs
    print('Starting one sequential Globo lifetime', flush=True)
    try:
        while (args.steps == 0 or int(step) < args.steps) and not stopping[0]:
            driver(policy, steps=10)
            if live:live.render(int(step),updates)
            if progress and int(step)>=next_video:
                progress.write(int(step),updates)
                next_video=(int(step)//args.model_video_every+1)*args.model_video_every
            now = time.time()
            if now - last_log >= 30:
                metrics = agg.result()
                logger.add(metrics, prefix='train')
                logger.add(dict(fps=(int(step)-initial_step)/max(1,now-started), updates=updates))
                logger.write()
                atomic_json(root / 'status.json', dict(step=int(step), updates=updates,
                    running=True, environment=env.record, time=now))
                last_log = now
            if now - last_save >= args.checkpoint_seconds:
                cp.save()
                if behavior is not None and behavior.store:
                    behavior.store.committed()
                env.save()
                if live:
                    from .continuous_monitor import bounded_cleanup
                    bounded_cleanup(root,replay)
                last_save = now
    finally:
        if progress:progress.write(int(step),updates)
        if live:live.render(int(step),updates,force=True)
        cp.save()
        if behavior is not None and behavior.store:
            behavior.store.committed()
        env.save()
        if live:
            from .continuous_monitor import bounded_cleanup
            bounded_cleanup(root,replay)
        logger.add(agg.result(), prefix='train')
        logger.add(dict(updates=updates))
        logger.write()
        logger.close()
        atomic_json(root / 'status.json', dict(step=int(step), updates=updates,
                    running=False, environment=env.record, time=time.time()))


if __name__ == '__main__':
    main()
