"""Single-lifetime density explorer using the hybrid's exact behavior implementation."""
import os
import signal
import time
import elements
import embodied
import numpy as np
from .model_free import ModelFreeBehavior
from .continuous_monitor import ContinuousMonitor, bound_logs
from .env import atomic_json


def train_model_free(args, root, env):
    behavior=ModelFreeBehavior(args.model_free_device,args.seed,0.,
        args.mf_replay_capacity,args.mf_goal_capacity,args.recode_capacity,root/'mf_replay',
        epsilon_end=args.mf_epsilon_end,gamma=args.mf_gamma,
        online_coef=args.mf_online_coef,offline_coef=args.mf_offline_coef,
        frontier_scale=args.mf_frontier_scale,lr=args.mf_lr,
        model_reward_weight=0.,model_feedback=args.model_feedback,
        reservoir_channels=args.reservoir_channels,reservoir_variant=args.reservoir_variant,
        num_actions=env.act_space['action'].classes,offline_updates=args.mf_offline_updates)
    if args.m2e_ftn:
        from .m2e_ftn import M2EFTN
        if args.frontier_ftn:
            from .frontier_ftn import FrontierFTN
            behavior.ftn=FrontierFTN(behavior.agent,args.frontier_ftn,root)
        else:behavior.ftn=M2EFTN(behavior.agent,args.m2e_ftn,active_neurons=args.m2e_ftn_k)
        atomic_json(root/'architecture.json',behavior.ftn.q.spec)
        print('FTN architecture:',behavior.ftn.q.spec,flush=True)
    if args.mf_init_weights:
        behavior.agent.epsilon_start = behavior.agent.epsilon_end
        behavior.agent.epsilon = behavior.agent.epsilon_end
    env.novelty_fn=behavior.observe_frame
    env.memory_size_fn=lambda:behavior.agent.density.size
    live=ContinuousMonitor(root/'live')
    if args.m2e_ftn:
        live.m2e_ftn=behavior.ftn;live.last_render_time=0
    capture=None
    if os.environ.get('GLOBO_EVENT_CAPTURE')=='1':
        if not args.m2e_ftn:raise ValueError('Event capture requires the FTN observation/mask stream')
        from .event_capture import EventCapture
        capture=EventCapture(root/'event_clips',behavior.ftn)
    life_capture=None
    if os.environ.get('GLOBO_LIFE_CAPTURE')=='1':
        from .life_capture import LifeCapture
        life_capture=LifeCapture(os.environ.get('GLOBO_LIFE_VIDEO_ROOT',str(root/'life_videos')),behavior.ftn)
    cp=elements.Checkpoint(elements.Path(root/'ckpt'))
    cp.step=elements.Counter(0);cp.behavior=behavior;cp.live=live
    if args.mf_init_weights and not cp.exists():
        behavior.load_weights(args.mf_init_weights)
        atomic_json(root/'warm_start.json',dict(source=args.mf_init_weights,
            time=time.time(),weights_only=True,replay=0,frontier=0,centroids=0,
            optimizer='fresh',schedule='minimum epsilon from start',
            epsilon=behavior.agent.epsilon,step=0))
    cp.load_or_save()
    # New-world forks keep all learned state but start geometric/evaluation
    # statistics afresh. The training counter stays at its inherited step.
    from .phase2 import read
    fork=read(root/'fork.json')
    if fork.get('reset_world_metrics') and int(cp.step)==fork['step']:
        live=ContinuousMonitor(root/'live');cp.live=live
        if args.m2e_ftn:live.m2e_ftn=behavior.ftn;live.last_render_time=0
    if os.environ.get('GLOBO_SURVIVAL')=='1':
        # Loading an optimizer restores its old LR. Explicit experiment settings
        # must win, while retaining the accumulated optimizer moments.
        for group in behavior.agent.optimizer.param_groups:group['lr']=args.mf_lr
        settings=dict(step=int(cp.step),gamma=behavior.agent.gamma,gamma_env=behavior.agent.gamma_env,
            learning_rate=[g['lr'] for g in behavior.agent.optimizer.param_groups],
            offline_updates=behavior.offline_updates,novelty_scale=behavior.agent.survival_novelty_scale,
            replay_size=behavior.agent.memory.size,frontier_size=behavior.agent.goal.size,
            density_size=behavior.agent.density.size)
        atomic_json(root/'survival_settings.json',settings)
        print('Survival settings:',settings,flush=True)
    behavior.agent.capture_input = args.render_size != 64
    step=cp.step
    live.event('process_start',int(step),pid=os.getpid(),model_free_only=True)
    stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    driver=embodied.Driver([lambda:env],parallel=False)
    driver.reset(lambda batch:None)
    applet=None
    if args.live_port:
        from .live_applet import LiveApplet
        applet=LiveApplet(args.live_port, location_path=root/'live'/'goal_sightings.sqlite')
    def on_step(transition,worker):
        step.increment()
        from .video_hud import values
        transition=dict(transition, **{'log/video_hud':values(env.record)})
        if behavior.agent.capture_input:
            transition=dict(transition, **{'log/human':env.human_frame,
                'log/m2e_input':behavior.agent.last_input,
                'log/m2e_feedback':behavior.agent.last_feedback})
        if args.m2e_ftn:
            transition=dict(transition,**{'log/m2e_ftn_ema':behavior.ftn.frame()})
        if not args.frontier_ftn:
            frontier=getattr(behavior.agent,'display_frontier',None)
            h,w=behavior.agent.recode_down
            goal=(np.zeros((h,w),np.uint8) if frontier is None else
                np.uint8(np.clip(frontier.numpy().reshape(h,w)*255,0,255)))
            transition=dict(transition,**{'log/offline_goal':goal,
                'log/offline_goal_step':np.int64(getattr(behavior.agent,'display_frontier_step',-1))})
        if args.frontier_ftn:
            behavior.ftn.record_observation(env.human_frame,bool(transition['is_first']))
            transition.update({'log/goal_id':np.int32(behavior.ftn.action_goal),
                'log/frontier_id':np.int32(behavior.ftn.action_final)})
        live.observe(transition,env.record,int(step),behavior.metrics)
        if applet:applet.observe(transition,env.record,int(step),behavior)
        if capture:capture.observe(transition,env.record,int(step))
        if life_capture:life_capture.observe(transition,env.record,int(step))
    driver.on_step(on_step)
    def policy(carry,obs,**kwargs):
        # Keep the hybrid's feedback input dimensions, with the model slot zero.
        actions=behavior.policy(obs,{},np.zeros((1,env.act_space['action'].classes),np.float32))
        return carry,actions,{}
    start=last_log=last_save=time.time();initial=int(step)
    def status(running):
        atomic_json(root/'status.json',dict(step=int(step),updates=behavior.updates,
            model_updates=0,model_free_only=True,running=running,environment=env.record,time=time.time(),
            ftn={k:v for k,v in behavior.metrics.items() if k.startswith('ftn_')}))
    def save():
        cp.save()
        if behavior.store:behavior.store.committed()
        env.save();bound_logs(root)
    print(f'Starting model-free-only explorer; FTN decoder={args.m2e_ftn}; no Dreamer; model feedback fixed at zero',flush=True)
    try:
        while (args.steps==0 or int(step)<args.steps) and not stopping[0]:
            driver(policy,steps=10)
            live.render(int(step),0)
            now=time.time()
            if now-last_log>=30:
                status(True)
                print(f'Step {int(step)} DQN updates {behavior.updates} fps {(int(step)-initial)/max(1,now-start):.2f}',flush=True)
                last_log=now
            if now-last_save>=args.checkpoint_seconds:
                save();last_save=now
    finally:
        live.render(int(step),0,force=True)
        save();status(False)
        if capture:capture.close()
        if life_capture:life_capture.close()
        if applet:applet.close()
