"""Read-only model evaluation on fresh, withheld sequential trajectories. POD ONLY."""
import argparse
import json
import itertools
import os
from pathlib import Path
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
STATE = Path(os.environ.get('GLOBO_STATE', '/home/dev/globo_dreamer_state'))
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(STATE/'cache/jax'))
sys.path.insert(0, str(STATE / 'vendor/dreamerv3'))
import av
import elements
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw
import ruamel.yaml
from .agent import Agent
from .env import Globo, atomic_json, runtime_path


class ProbeAgent(Agent):
    def report(self, carry, data):
        # No loss/optimizer/normalization updates; this is solely model inference.
        obs = {k: data[k] for k in self.obs_space}
        B, T = obs['is_first'].shape
        C = T // 2
        prev = {'action': jnp.concatenate([jnp.zeros((B, 1), jnp.int32),
                                          data['action'][:, :-1]], 1)}
        enc, dyn, dec, _ = self.init_policy(B)
        reset = obs['is_first'].at[:, 0].set(True)
        _, _, tokens = self.enc(enc, obs, reset, training=False)
        _, _, features = self.dyn.observe(dyn, tokens, prev, reset, training=False)
        _, _, recon = self.dec(dec, features, reset, training=False)
        _, _, context_tokens = self.enc(enc, {k:v[:, :C] for k,v in obs.items()},
                                        reset[:, :C], training=False)
        context, _, _ = self.dyn.observe(dyn, context_tokens,
            {k:v[:, :C] for k,v in prev.items()}, reset[:, :C], training=False)
        actions = {'action': data['action'][:, C-1:T-1]}
        _, imagined, _ = self.dyn.imagine(context, actions, T-C, training=False)
        _, _, predicted = self.dec(dec, imagined, jnp.zeros((B,T-C),bool), training=False)
        wrong = {'action': (actions['action'] + 7) % 25}
        _, wrongfeat, _ = self.dyn.imagine(context, wrong, T-C, training=False)
        _, _, wrongpred = self.dec(dec, wrongfeat, jnp.zeros((B,T-C),bool), training=False)
        if hasattr(self,'image_dim'):
            forecasts=self.ensemble(features['deter'][:,1:],prev['action'][:,1:])
            targets=tokens[:,1:,-self.image_dim:]
            disagreement=forecasts.var(0).mean(-1)
        else:
            real = self.feat2tensor(features)
            forecasts = self.ensemble(real[:, :-1])
            selection = jax.nn.one_hot(data['action'][:, :-1], forecasts.shape[-2])
            forecasts = (forecasts * selection[None, ..., None]).sum(-2)
            pixels = obs['image'].astype(jnp.float32)/255
            pixels = pixels.reshape(B,T,8,8,8,8,3).mean((3,5,6)).reshape(B,T,64)
            targets = jnp.concatenate([pixels,obs['body'][...,:8]],-1)[:,1:]
            disagreement=jnp.sqrt(forecasts.var(0).mean(-1)+1e-8)
        return carry, dict(truth=obs['image'], posterior=recon['image'].pred(),
            forecast=predicted['image'].pred(), wrong_actions=wrongpred['image'].pred(),
            novelty_error=jnp.sqrt(((forecasts.mean(0)-targets)**2).mean(-1)+1e-8),
            novelty_disagreement=disagreement,
            true_body=obs['body'][:, C:], predicted_body=predicted['body'].pred())


def collect(out, batches=4, batch=8, length=32, seed=7, spawn='12,5.1,120'):
    dataset = out / 'heldout.npz'
    if dataset.exists():
        return
    env = Globo(out / 'environment', seed=seed, spawn=spawn,
                episode_steps=0, video_every=0, audit_spawn=True)
    # Match training's observation preprocessing, including its RECODE input.
    # This independent evaluation memory never loads or modifies training memory.
    from .model_free import ModelFreeBehavior
    behavior = ModelFreeBehavior(device='cpu', seed=seed, replay_capacity=1,
                                 goal_capacity=1, recode_capacity=10000)
    env.novelty_fn = behavior.observe_frame
    env.memory_size_fn = lambda: behavior.agent.density.size
    # Fixed motor probes: includes movement, turns, sky/ground looks and idle.
    # These trajectories are evaluation-only and never enter training replay.
    pattern = [1, 6, 11, 8, 9, 3, 7, 0, 5, 9, 1, 10]
    sequences, buffer = [], []
    obs = env.step(dict(reset=True, action=0))
    count = 0
    try:
        while len(sequences) < batches * batch:
            act = pattern[(count // 12) % len(pattern)]
            row = dict(obs, action=np.int32(act), uncertainty=np.float32(0),
                       consec=np.int32(0), stepid=np.zeros(20, np.uint8))
            if not obs['is_last']:
                buffer.append(row)
            if len(buffer) == length:
                sequences.append({k:np.stack([r[k] for r in buffer]) for k in row})
                buffer = []
            obs = env.step(dict(reset=bool(obs['is_last']), action=act))
            if obs['is_first']:
                buffer = []
            count += 1
        np.savez_compressed(dataset, **{k:np.stack([s[k] for s in sequences])
                                      for k in sequences[0]})
        atomic_json(out/'dataset.json',dict(seed=seed,spawn=spawn,sequences=len(sequences),
            length=length,density='Canonical explorer RECODE with fresh evaluation-only memory',
            withheld=True))
        audits = list((out/'environment/engine').glob('*/worlds/world/spawn_audit.json'))
        for i, path in enumerate(audits):
            atomic_json(out / f'spawn_audit_{i}.json', json.loads(path.read_text()))
    finally:
        env.close()


def video(path, result, index=None):
    # Select a visually varied sequence by truth alone, consistently across models.
    if index is None:
        index=int(np.argmax(result['truth'].astype(float).var(axis=(2,3)).mean(axis=(1,2))))
    truth = result['truth'][index]
    post = np.clip(result['posterior'][index]*255,0,255).astype(np.uint8)
    future = np.clip(result['forecast'][index]*255,0,255).astype(np.uint8)
    C = len(truth)//2
    masks=result.get('mask_frames')
    width=1024 if masks is not None else 768
    container = av.open(str(path), 'w')
    stream = container.add_stream('libx264', rate=6)
    stream.width, stream.height, stream.pix_fmt = width, 280, 'yuv420p'
    for t in range(len(truth)):
        canvas = Image.new('RGB',(width,280),'#171c24')
        frames = [truth[t], post[t], truth[t] if t<C else future[t-C]]
        labels = ['Actual', 'Posterior reconstruction', 'Context' if t<C else f'Open loop +{t-C+1}']
        if masks is not None:
            frames.append(masks[t]);labels.append('Live mask + EMA trails')
        for i,(frame,label) in enumerate(zip(frames,labels)):
            canvas.paste(Image.fromarray(frame).resize((256,256),Image.Resampling.NEAREST if i==3 else Image.Resampling.BICUBIC), (i*256,24))
            ImageDraw.Draw(canvas).text((i*256+6,5),label,fill='white')
        for packet in stream.encode(av.VideoFrame.from_image(canvas)):
            container.mux(packet)
        if t in (C, C+3, C+7, len(truth)-1):
            canvas.save(path.with_name(f'{path.stem}_h{t-C+1}.jpg'))
    for packet in stream.encode(): container.mux(packet)
    container.close()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',default='pilot')
    p.add_argument('--out',default='world_probe')
    p.add_argument('--collect-only',action='store_true')
    p.add_argument('--batches',type=int,default=4)
    p.add_argument('--checkpoints',nargs='+',default=['initial','phase1_snapshot/ckpt','ckpt'])
    args=p.parse_args()
    out=runtime_path(STATE/'diagnostics'/args.out)
    root=STATE/'runs'/args.run
    launch=json.loads((root/'launch.json').read_text())
    collect(out, batches=args.batches, seed=launch['seed'], spawn=launch['spawn'])
    if args.collect_only: return
    config=elements.Config(ruamel.yaml.YAML(typ='safe').load((root/'dreamer.yaml').read_text()))
    agentcfg=dict(config.agent)
    agentcfg['explore']=dict(ensemble=5,density=launch['density'],disagreement=launch['disagreement'],
        actor_weight=launch['actor_weight'],feedback=launch['feedback'],horizon=launch.get('intrinsic_horizon',100),
        prediction_error=launch.get('prediction_error',0),disagreement_power=launch.get('disagreement_power',1))
    agentcfg['explore']['model_only']=launch.get('world_model_only',False)
    ac=elements.Config(**agentcfg,logdir=str(out),seed=config.seed,jax=config.jax,
        batch_size=8,batch_length=32,replay_context=0,report_length=32,replica=0,replicas=1)
    spaces=Globo(out/'spaces',video_every=0)
    agent=ProbeAgent(spaces.obs_space,{k:v for k,v in spaces.act_space.items() if k!='reset'},ac)
    data=dict(np.load(out/'heldout.npz'))
    batches=[{k:v[start:start+8].copy() for k,v in data.items()}
             for start in range(0,len(data['image']),8)]
    prepared_stream=iter(agent.stream(itertools.cycle(batches)))
    results={}
    for checkpoint in args.checkpoints:
        if checkpoint!='initial':
            cp=elements.Checkpoint(elements.Path(root/checkpoint),write=False)
            cp.agent=agent
            cp.load(keys=['agent'])
        label=checkpoint.replace('/','_')
        all_results=[]
        for start in range(0,len(data['image']),8):
            prepared=next(prepared_stream)
            _,r=agent.report(agent.init_report(8),prepared)
            r={k:np.asarray(v) for k,v in r.items() if k!='params/summary'}
            all_results.append(r)
            if start==0:
                video(out/f'{label}.mp4',r)
                # First four contiguous probes, selected without looking at predictions.
                for index in range(min(4, len(r['truth']))):
                    video(out/f'{label}_sample{index}.mp4',r,index=index)
                np.savez_compressed(out/f'{label}_predictions.npz',**r)
        r={k:np.concatenate([x[k] for x in all_results]) for k in all_results[0]}
        C=r['truth'].shape[1]//2
        actual=r['truth'].astype(np.float32)/255
        target=actual[:,C:]
        persistence=np.broadcast_to(actual[:,C-1:C],target.shape)
        score=dict(posterior_mse=float(np.mean((r['posterior']-actual)**2)),
                   body_mse=float(np.mean((r['predicted_body']-r['true_body'])**2)),
                   sequences=len(actual),context=C,heldout=True)
        texture=.5*(np.abs(np.diff(target.mean(-1),axis=2)).mean((2,3))+
                    np.abs(np.diff(target.mean(-1),axis=3)).mean((2,3)))
        for h in [1,4,8,16]:
            e=float(np.mean((r['forecast'][:,h-1]-target[:,h-1])**2))
            base=float(np.mean((persistence[:,h-1]-target[:,h-1])**2))
            wrong=float(np.mean((r['wrong_actions'][:,h-1]-target[:,h-1])**2))
            score[f'h{h}']=dict(mse=e,persistence_mse=base,wrong_action_mse=wrong,
                skill_vs_persistence=1-e/max(base,1e-8))
            mask=texture[:,h-1]>.01
            if mask.any():
                e2=float(np.mean((r['forecast'][mask,h-1]-target[mask,h-1])**2))
                b2=float(np.mean((persistence[mask,h-1]-target[mask,h-1])**2))
                score[f'h{h}']['textured_subset']=dict(n=int(mask.sum()),mse=e2,
                    persistence_mse=b2,skill_vs_persistence=1-e2/max(b2,1e-8))
        results[checkpoint]=score
        atomic_json(out/'scores.json',results)
        print(json.dumps({checkpoint:score}),flush=True)
    gallery(out,args.run,args.checkpoints)


def gallery(out,run,checkpoints):
    cards=[]
    for checkpoint in checkpoints:
        label=checkpoint.replace('/','_')
        cards.append(f'<h2>{label}</h2>')
        cards.append('<p>Textured example: highest spatial variation in the first batch of real frames '
                     '(same selection rule for trained and untrained models).</p>'
                     f'<video controls preload="metadata" src="{label}.mp4"></video>'
                     f'<p><a href="{label}_h16.jpg">16-step forecast still</a></p>')
        for index in range(4):
            cards.append(f'<p>Fixed probe {index+1}</p><video controls preload="metadata" '
                         f'src="{label}_sample{index}.mp4"></video>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<style>body{background:#141820;color:#eee;font:16px system-ui;margin:24px} '
        'video{width:min(100%,768px)}a{color:#9bd0ff}</style>'
        f'<h1>World model: {run}</h1>'
        '<p>Left: real frames. Middle: reconstruction with access to each frame. '
        'Right: 16 context frames followed by 16 open-loop predictions using the recorded actions, '
        'without seeing future frames. Playback is slowed to 6 fps. '
        'These scripted trajectories were withheld from training. First four probes are shown without selection.</p>'
        '<p><a href="scores.json">Metrics, including persistence and wrong-action comparisons</a></p>'
        +''.join(cards))


if __name__=='__main__': main()
