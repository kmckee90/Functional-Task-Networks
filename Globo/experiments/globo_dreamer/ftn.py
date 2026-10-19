"""Whole-world-model BayesLoRA, ported from ftn/projects/bayesian_adv.

One atom ID denotes rank-one factors for EVERY learned world-model tensor.
Backbone parameters are immutable. Mask search differentiates only logits;
weight training differentiates only atom banks. Replay runs under recorded masks.
"""
from collections import deque,OrderedDict
import json
import math
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

SIDE=int(os.environ.get('GLOBO_FTN_SIDE','48'))
H=SIDE*SIDE
K=int(os.environ.get('GLOBO_FTN_K','256'))
ALPHA=float(os.environ.get('GLOBO_FTN_ALPHA','64'))
if not (1<=K<=H<=65535 and math.isfinite(ALPHA) and ALPHA>0):
    raise ValueError('Invalid FTN dimensions or alpha')
ROOTS={'enc','dyn','dec','rew','con','density','ensemble','pol','val','ival'}


def hard_mask(logits):
    ids=jax.lax.top_k(logits,K)[1]
    return jnp.zeros(H,jnp.float32).at[ids].set(1.)


def bernoulli_kl(logits,prior):
    q=jnp.clip(jax.nn.sigmoid(logits),1e-6,1-1e-6)
    p=jnp.clip(jax.nn.sigmoid(prior),1e-6,1-1e-6)
    return jnp.mean(q*jnp.log(q/p)+(1-q)*jnp.log((1-q)/(1-p)))


def features(obs):
    image=np.asarray(obs['image'],np.float32).reshape(-1,64,64,3)/255.
    pixels=image.reshape(-1,8,8,8,8,3).mean((2,4,5)).reshape(-1,64)
    body=np.asarray(obs['body'],np.float32).reshape(len(image),-1)[:,:8]
    return np.concatenate([pixels,body],-1)


class InputRouter:
    """Append-only PCA projectors and posterior snapshots, as InputConditioned.

    Features are fixed pixels/body, never a drifting learned embedding. The most
    recent near-tied route wins. Projectors are immutable once archived.
    """
    def __init__(self):
        self.entries=[];self.cov=np.zeros((72,72),np.float64);self.samples=0
        self.projector=None;self.calls=0;self.hits=0;self.changes=0
    def observe(self,x):
        self.cov+=x.T@x;self.samples+=len(x)
        if self.samples>=4 and (self.projector is None or self.calls%8==0):
            _,v=np.linalg.eigh(self.cov);self.projector=v[:,-2:]@v[:,-2:].T
        self.calls+=1
    @staticmethod
    def error(x,p):
        return float(np.square(x-x@p).sum()/max(np.square(x).sum(),1e-12))
    def novel(self,x):
        return self.samples>=20 and self.projector is not None and self.error(x,self.projector)>=.10
    def archive(self,logits):
        if self.projector is not None:
            self.entries.append(dict(projector=self.projector.copy(),logits=np.array(logits),samples=self.samples))
        self.cov.fill(0);self.samples=0;self.projector=None;self.changes+=1
    def match(self,x):
        if not self.entries:return None
        errors=[self.error(x[:4],e['projector']) for e in self.entries]
        best=min(errors)
        if best>=.10:return None
        idx=next(i for i in reversed(range(len(errors))) if errors[i]<=best+.001)
        self.hits+=1
        return self.entries[idx]['logits'].copy()
    def save(self):return dict(entries=self.entries,cov=self.cov,samples=self.samples,projector=self.projector,calls=self.calls,hits=self.hits,changes=self.changes)
    def load(self,d):self.__dict__.update(d)


class BayesWorldModel:
    """Embodied-compatible host wrapper; all numerical computation is on pod GPU."""
    ftn_enabled=True
    mask_side=SIDE
    atom_count=H
    def __init__(self,agent,root,seed):
        # This single-device wrapper deliberately exchanges the small routing
        # state with the host; upstream's sharded wrapper disallows implicit IO.
        jax.config.update('jax_transfer_guard','allow')
        self.agent=agent;self.root=Path(root);self.seed=seed
        self.rng=np.random.default_rng(seed);self.key=jax.random.PRNGKey(seed+719)
        # Upstream optimizer state is not part of the frozen backbone.
        self.base={k:v for k,v in agent.params.items() if not k.startswith('opt/')}
        self.keys=tuple(k for k,v in self.base.items() if k.split('/')[0] in ROOTS and jnp.issubdtype(v.dtype,jnp.floating))
        self.banks={}
        for key in self.keys:
            shape=self.base[key].shape
            if len(shape)>=2:
                fan=math.prod(shape[:-1]);out=shape[-1]
                a=jax.random.uniform(self.nextkey(),(H,fan),minval=-1/math.sqrt(fan),maxval=1/math.sqrt(fan))
                b=.01*jax.random.normal(self.nextkey(),(H,out))
                self.banks[key]={'a':a,'b':b}
            else:
                self.banks[key]={'b':.01*jax.random.normal(self.nextkey(),(H,self.base[key].size))}
        self.optimizer=optax.chain(optax.clip_by_global_norm(10),optax.adam(4e-5))
        self.optstate=self.optimizer.init(self.banks)
        self.train_logits=jnp.zeros(H,jnp.float32);self.action_logits=jnp.zeros(H,jnp.float32)
        self.action_mask=hard_mask(jax.random.normal(self.nextkey(),(H,)))
        self.train_mask=self.action_mask
        self.router=InputRouter();self.recent=deque(maxlen=256)
        self.interactions=0;self.updates=0;self.last_reconfig=-8
        self.loss_mean=None;self.loss_var=0.;self.loss_since=0
        self.action_changes=0;self.action_reconfigs=0;self.train_changes=0
        self.prediction_next=None;self.last_forecasts=None
        self.online_mean=None;self.online_var=0.;self.online_since=0
        self.replay_updates=0;self.replay_candidates=0;self.repair_attempts=0;self.repair_accepted=0
        self.ema=np.zeros((H,3),np.float32);self.frames=deque(maxlen=192)
        self.layout=np.arange(H);self.layout_updates=0;self.last_viz=0
        self.layout_method='SpatialPermuteRandomCenter_v1';self.layout_initialized=False
        self.mask_history=deque(maxlen=64)
        self.learned_masks=OrderedDict()
        self._pure_loss=nj.pure(agent.model.loss)
        self._pure_init=nj.pure(agent.model.init_policy)
        # Canonical single-GPU layout prevents recompilation when recent NumPy
        # support alternates with upstream replay's explicitly sharded arrays.
        sharding=agent.train_mirrored
        def compile(fn,**kw):
            compiled=jax.jit(fn,in_shardings=sharding,out_shardings=sharding,**kw)
            def call(*args):
                # Canonicalize before JIT dispatch as well as inside its ABI:
                # explicit in_shardings alone still permits dispatch variants.
                return compiled(*jax.device_put(args,sharding))
            call.clear_cache=compiled.clear_cache
            call._cache_size=compiled._cache_size
            return call
        self.materialize=compile(self._materialize)
        self.search=compile(self._search)
        self.update=compile(self._update,donate_argnums=(0,1))
        self.evaluate=compile(self._evaluate)
        self.recontext=compile(self._recontext)
        self.forecast=compile(self._forecast)
        count=sum(x.size for x in jax.tree.leaves(self.banks))
        spec=dict(atoms=H,side=SIDE,active=K,rank=1,alpha=ALPHA,scale=ALPHA/K,
            trainable_atom_parameters=count,backbone_parameters=sum(self.base[k].size for k in self.keys),
            coverage={k:list(self.base[k].shape) for k in self.keys},
            train_reconfiguration_steps=2,action_reconfiguration_steps=2,action_interval=8,
            history_repair_observations=32,replay_overlap=[.2,.8],
            replay_masks='Checkpointed learned-mask sidecar indexed by ftn_index in Dreamer replay; endpoints select sequence masks',
            repair_every_updates=512,repair_fit_frames=16,repair_validation_frames=16,
            router='InputConditioned PCA rank 2; fixed 72-D input features; threshold 0.10',
            spatial_permute='Display only; atom IDs and mask overlaps are unchanged')
        (self.root/'ftn_spec.json').write_text(json.dumps(spec,indent=2))
        print('FTN atom parameters:',count,'covered tensors:',len(self.keys),flush=True)
        self.sync_policy()

    def nextkey(self):
        self.key,key=jax.random.split(self.key);return key
    def _materialize(self,banks,mask):
        params=dict(self.base)
        for key,bank in banks.items():
            if 'a' in bank:
                delta=(bank['a']*mask[:,None]).T@bank['b']
            else:delta=mask@bank['b']
            params[key]=self.base[key]+(ALPHA/K)*delta.reshape(self.base[key].shape)
        return params
    def _loss(self,banks,mask,data,key):
        params=self._materialize(banks,mask)
        B=data['image'].shape[0]
        _,carry=self._pure_init(params,B,seed=key)
        obs={k:data[k] for k in self.agent.obs_space}
        obs=dict(obs,is_first=obs['is_first'].at[:,0].set(True))
        prev={'action':jnp.concatenate([jnp.zeros((B,1),jnp.int32),data['action'][:,:-1]],1)}
        _,(loss,aux)=self._pure_loss(params,carry[:3],obs,prev,training=True,seed=key)
        return loss,aux[-1]
    def _evaluate(self,banks,mask,data,key):return self._loss(banks,mask,data,key)[0]
    def _forecast(self,params,dyn,key):
        def predict():return self.agent.model.ensemble(self.agent.model.feat2tensor(dyn)).mean(0)
        return nj.pure(predict)(params,seed=key)[1]
    def _recontext(self,params,data,key):
        def observe():
            B=data['image'].shape[0]
            enc,dyn,dec,_=self.agent.model.init_policy(B)
            obs={k:data[k] for k in self.agent.obs_space}
            reset=obs['is_first'].at[:,0].set(True)
            prev={'action':jnp.concatenate([jnp.zeros((B,1),jnp.int32),data['action'][:,:-1]],1)}
            enc,_,tokens=self.agent.model.enc(enc,obs,reset,training=False)
            dyn,_,_=self.agent.model.dyn.observe(dyn,tokens,prev,reset,training=False)
            return enc,dyn,dec,{'action':data['action'][:,-1]}
        return nj.pure(observe)(params,seed=key)[1]
    def _search(self,banks,prior,data,key,kl_weight):
        # Reset Adam moments each support search, matching BayesMaskReconfig.
        logits=prior;m=jnp.zeros_like(logits);v=jnp.zeros_like(logits)
        for i in range(2):
            key,noise_key,loss_key=jax.random.split(key,3)
            noise=jax.random.gumbel(noise_key,(H,))
            def objective(z):
                soft=jax.nn.sigmoid(z)
                hard=hard_mask(z/math.exp(-i/10)+noise)
                mask=soft+jax.lax.stop_gradient(hard-soft)
                return self._loss(banks,mask,data,loss_key)[0]+kl_weight*bernoulli_kl(z,prior)
            loss,g=jax.value_and_grad(objective)(logits)
            m=.9*m+.1*g;v=.999*v+.001*g*g
            logits=logits-.3*(m/(1-.9**(i+1)))/(jnp.sqrt(v/(1-.999**(i+1)))+1e-8)
        return logits,hard_mask(logits+jax.random.gumbel(key,(H,))),loss
    def _update(self,banks,optstate,mask,data,replay_mask,replay_data,replay_valid,key):
        key,other=jax.random.split(key)
        def objective(b):
            loss,metrics=self._loss(b,mask,data,key)
            # The replay mask is preserved, not replaced by the current mask.
            replay_loss=jax.lax.cond(replay_valid>0,
                lambda:self._loss(b,replay_mask,replay_data,other)[0],lambda:jnp.float32(0))
            return loss+replay_valid*replay_loss,dict(metrics,ftn_primary_loss=loss,ftn_replay_loss=replay_loss*replay_valid)
        (loss,metrics),grad=jax.value_and_grad(objective,has_aux=True)(banks)
        updates,optstate=self.optimizer.update(grad,optstate,banks)
        active=jnp.maximum(mask,replay_valid*replay_mask)
        # Adam momentum must not move inactive atoms.
        updates=jax.tree.map(lambda x:x*active[:,None],updates)
        banks=optax.apply_updates(banks,updates)
        return banks,optstate,dict(metrics,ftn_total_loss=loss,ftn_grad_norm=optax.global_norm(grad))
    def sync_policy(self):
        params=self.materialize(self.banks,self.action_mask)
        self.agent.params.update(params)
        self.agent.policy_params={k:self.agent.params[k] for k in self.agent.policy_keys}
        self.agent.pending_sync=None
    def __getattr__(self,key):return getattr(self.agent,key)
    def train(self,carry,data):
        started=time.perf_counter()
        data=dict(data);data.pop('seed',None)
        saved=np.asarray(data.pop('ftn_mask'),np.uint16)
        indices=np.asarray(data.pop('ftn_index',np.zeros(saved.shape[:2],np.uint32)))
        data={k:data[k] for k in [*self.agent.obs_space,'action']}
        replay_pool=data
        # Current support follows the one real trajectory; uniform historical
        # replay remains the separate overlap-filtered preservation term.
        used_recent=len(self.recent)>=32
        if used_recent:
            rows=list(self.recent);windows=[]
            for offset in range(0,32,4):
                end=max(32,len(rows)-offset);window=rows[end-32:end]
                windows.append({k:np.stack([r[k] for r in window]) for k in data})
            data={k:np.stack([w[k] for w in windows]) for k in data}
        # Replay sequence windows are recomputed from zero hidden state under
        # their endpoint's recorded mask, including earlier frames as support.
        x=features(data)
        routed=self.router.match(x)
        prior=self.train_logits if routed is None else jnp.asarray(routed)
        pre=float(self.evaluate(self.banks,hard_mask(prior),data,self.nextkey()))
        change=self.loss_mean is not None and self.loss_since>=20 and pre>self.loss_mean+5*math.sqrt(max(self.loss_var,1e-8))
        if change:
            prior=jnp.zeros_like(prior) if routed is None else prior
            self.train_changes+=1
        self.train_logits,self.train_mask,search_loss=self.search(self.banks,prior,data,self.nextkey(),jnp.float32(0 if change else 1))
        mask=np.asarray(self.train_mask)
        searched=time.perf_counter()
        stored=np.zeros((len(saved),H),np.float32)
        for i,ids in enumerate(saved[:,-1]):
            # Prefer the mask that actually learned this endpoint/window. The
            # compact sidecar is checkpointed alongside the existing replay;
            # it avoids mutating already-written upstream replay chunks.
            learned=self.learned_masks.get(int(indices[i,-1]))
            if learned is not None:stored[i,learned]=1
        overlap=stored@mask/K;eligible=np.flatnonzero((overlap>.2)&(overlap<=.8))
        self.replay_candidates+=len(eligible)
        index=int(self.rng.choice(eligible)) if len(eligible) else 0
        replay={k:v[index:index+1] for k,v in replay_pool.items()}
        self.banks,self.optstate,metrics=self.update(self.banks,self.optstate,self.train_mask,data,
            jnp.asarray(stored[index]),replay,jnp.float32(bool(len(eligible))),self.nextkey())
        self.updates+=1;self.replay_updates+=int(bool(len(eligible)))
        self.agent.n_updates.increment()
        learned=np.flatnonzero(mask).astype(np.uint16)
        trained_indices=([int(row['ftn_index']) for row in list(self.recent)[-60:] if 'ftn_index' in row]
            if used_recent else indices.ravel())
        for index in trained_indices:
            if index:self.learned_masks[int(index)]=learned
        while len(self.learned_masks)>self.agent.config.get('ftn_replay_capacity',100000):self.learned_masks.popitem(last=False)
        value=float(metrics['ftn_primary_loss'])
        trained=time.perf_counter()
        if self.loss_mean is None or change:self.loss_mean=value;self.loss_var=0.;self.loss_since=0
        else:
            delta=value-self.loss_mean;self.loss_mean=.95*self.loss_mean+.05*value;self.loss_var=.95*self.loss_var+.05*delta*delta
        self.loss_since+=1
        self.ema=np.maximum(self.ema*np.array([.95,.30,.975]),mask[:,None])
        if self.updates%8==0:self.frames.append(self.ema.copy())
        if self.updates%32==0:self.mask_history.append(mask.copy())
        if self.updates%512==0:self.repair_route(replay)
        self.render_masks()
        metrics=dict(metrics,ftn_search_loss=search_loss,ftn_active=mask.sum(),ftn_replay_used=float(bool(len(eligible))),
            ftn_train_changes=float(self.train_changes),ftn_action_reconfigs=float(self.action_reconfigs),
            ftn_action_changes=float(self.action_changes),ftn_route_hits=float(self.router.hits),ftn_routes=float(len(self.router.entries)))
        metrics.update(ftn_search_seconds=searched-started,ftn_update_seconds=trained-searched,
            ftn_repair_attempts=float(self.repair_attempts),ftn_repair_accepted=float(self.repair_accepted))
        return carry,{},jax.device_get(metrics)
    def repair_route(self,replay):
        """Append-only InputConditioned repair with disjoint temporal validation.

        Uses the same two-step support-search budget, once per 512 updates.
        Neither the live action posterior nor recorded replay masks are changed.
        """
        if not self.router.entries:return
        x=features(replay)
        errors=[self.router.error(x[:4],e['projector']) for e in self.router.entries]
        best=min(errors)
        if best>=.10:return
        index=next(i for i in reversed(range(len(errors))) if errors[i]<=best+.001)
        entry=self.router.entries[index]
        fit={k:v[:,:16] for k,v in replay.items()};valid={k:v[:,16:] for k,v in replay.items()}
        prior=jnp.asarray(entry['logits']);key=self.nextkey()
        old=float(self.evaluate(self.banks,hard_mask(prior),valid,key))
        candidate,mask,_=self.search(self.banks,prior,fit,self.nextkey(),jnp.float32(1))
        new=float(self.evaluate(self.banks,mask,valid,key));self.repair_attempts+=1
        if new<old:
            logits=np.where(np.asarray(mask)>.5,4.,-4.).astype(np.float32)
            self.router.entries.append(dict(projector=entry['projector'].copy(),logits=logits,samples=entry['samples']))
            self.repair_accepted+=1
    def support_batch(self,current=None):
        rows=list(self.recent)[-32:]
        if current is not None:
            row={k:np.asarray(v[0]) for k,v in current.items()}
            row['action']=np.int32(0)
            rows=(rows+[row])[-32:]
        if not rows:return None
        rows=[rows[0]]*(32-len(rows))+rows
        return {k:np.stack([r[k] for r in rows])[None] for k in [*self.agent.obs_space,'action']}
    def observe(self,transition):
        if bool(transition['is_first']):self.recent.clear()
        self.recent.append({k:np.array(v,copy=True) for k,v in transition.items()})
        if self.last_forecasts is not None:
            self.prediction_next=self.last_forecasts[int(transition['action'])].copy()
    def policy(self,carry,obs,mode='train'):
        x=features(obs);self.interactions+=1
        change=self.router.novel(x)
        if self.prediction_next is not None and not bool(np.asarray(obs['is_first']).any()):
            error=float(np.mean(np.square(x[0]-self.prediction_next)))
            loss_change=(self.online_mean is not None and self.online_since>=20 and
                error>self.online_mean+5*math.sqrt(max(self.online_var,1e-8)))
            change=change or loss_change
            if self.online_mean is None or change:
                self.online_mean=error;self.online_var=0.;self.online_since=0
            else:
                delta=error-self.online_mean;self.online_mean=.95*self.online_mean+.05*error
                self.online_var=.95*self.online_var+.05*delta*delta
            self.online_since+=1
        due=self.interactions-self.last_reconfig>=8
        if self.recent and (due or change):
            if change:self.router.archive(np.asarray(self.train_logits));self.action_changes+=1
            routed=self.router.match(x)
            prior=self.action_logits if routed is None else jnp.asarray(routed)
            if change and routed is None:prior=jnp.zeros(H,jnp.float32)
            self.action_logits,self.action_mask,_=self.search(self.banks,prior,self.support_batch(obs),self.nextkey(),jnp.float32(0 if change else 1))
            self.last_reconfig=self.interactions;self.action_reconfigs+=1
            # Rebuild recent latent history under the selected model rather
            # than carrying incompatible hidden state or discarding context.
            params=self.materialize(self.banks,self.action_mask)
            carry=self.agent._split(self.recontext(params,self.support_batch(),self.nextkey()))
        self.router.observe(x)
        self.sync_policy()
        carry,acts,outs=self.agent.policy(carry,obs,mode)
        dyn=self.agent._stack(carry)[1]
        self.last_forecasts=np.asarray(self.forecast(self.agent.params,dyn,self.nextkey()))[0]
        ids=np.flatnonzero(np.asarray(self.action_mask)).astype(np.uint16)
        outs['ftn_mask']=np.repeat(ids[None],len(x),axis=0)
        outs['ftn_index']=np.full(len(x),self.interactions,np.uint32)
        return carry,acts,outs
    def report(self,carry,data):
        data={k:v for k,v in data.items() if k not in ('ftn_mask','ftn_index','log/ftn_ema')}
        self.sync_policy()
        return self.agent.report(carry,data)
    def render_masks(self,force=False):
        if not force and time.time()-self.last_viz<60:return
        from PIL import Image
        from .ftn_spatial_permute import spatial_permutation
        from .env import atomic_json
        out=self.root/'live';out.mkdir(exist_ok=True)
        masks=[np.asarray(hard_mask(jnp.asarray(e['logits']))) for e in self.router.entries[-16:]]+list(self.mask_history)[-8:]
        if masks:
            # Exact base spatial_permute algorithm; bounded recent masks make
            # live display cost independent of run length. No training mutation.
            self.layout=spatial_permutation(np.array(masks[-16:]),(SIDE,SIDE),
                self.layout if self.layout_initialized else None, sweeps=8,
                initialization='random_blobs', center=0.25, roundness=2.0,
                radius=7, temperature=0.05, seed=0)
            self.layout_initialized=True
            self.layout_updates+=1
        frames=[Image.fromarray(np.uint8(np.clip(e[self.layout].reshape(SIDE,SIDE,3),0,1)*255)).resize((384,384),Image.Resampling.NEAREST) for e in self.frames]
        if not frames:frames=[Image.fromarray(np.uint8(self.ema[self.layout].reshape(SIDE,SIDE,3)*255)).resize((384,384),Image.Resampling.NEAREST)]
        frames[-1].save(out/'bw_pending.png');(out/'bw_pending.png').replace(out/'bw.png')
        frames[0].save(out/'bw_pending.gif',save_all=True,append_images=frames[1:],duration=100,loop=0);(out/'bw_pending.gif').replace(out/'bw.gif')
        atomic_json(out/'ftn.json',dict(step=self.interactions,updates=self.updates,active=K,atoms=H,rank=1,alpha=ALPHA,
            routes=len(self.router.entries),route_hits=self.router.hits,action_changes=self.action_changes,
            action_reconfigs=self.action_reconfigs,train_changes=self.train_changes,replay_updates=self.replay_updates,
            replay_candidates=self.replay_candidates,repair_attempts=self.repair_attempts,repair_accepted=self.repair_accepted,
            layout_updates=self.layout_updates,layout_method=self.layout_method,time=time.time()))
        self.last_viz=time.time()
    def save(self):
        return dict(version=1,alpha=ALPHA,base=jax.device_get(self.base),banks=jax.device_get(self.banks),optstate=jax.device_get(self.optstate),
            train_logits=np.asarray(self.train_logits),action_logits=np.asarray(self.action_logits),action_mask=np.asarray(self.action_mask),
            router=self.router.save(),key=np.asarray(self.key),rng=self.rng.bit_generator.state,
            agent_counters={k:int(getattr(self.agent,k)) for k in ['n_updates','n_batches','n_actions']},
            counters={k:getattr(self,k) for k in ['interactions','updates','last_reconfig','loss_mean','loss_var','loss_since',
                'action_changes','action_reconfigs','train_changes','online_mean','online_var','online_since',
                'replay_updates','replay_candidates','repair_attempts','repair_accepted','layout_updates']},
            ema=self.ema,layout=self.layout,layout_method=self.layout_method,
            layout_initialized=self.layout_initialized,
            mask_history=list(self.mask_history),learned_masks=list(self.learned_masks.items()))
    def load(self,d):
        if d['version']!=1:raise ValueError('FTN checkpoint version mismatch')
        if np.shape(d['action_mask'])!=(H,) or int(np.asarray(d['action_mask']).sum())!=K:
            raise ValueError('FTN checkpoint atom count or active count mismatch')
        if d.get('alpha',64.)!=ALPHA:raise ValueError('FTN checkpoint alpha mismatch')
        self.base=jax.device_put(d['base']);self.banks=jax.device_put(d['banks']);self.optstate=jax.device_put(d['optstate'])
        for k in ['train_logits','action_logits','action_mask','key']:setattr(self,k,jnp.asarray(d[k]))
        self.router.load(d['router']);self.rng.bit_generator.state=d['rng']
        for k,v in d['agent_counters'].items():getattr(self.agent,k).value=int(v)
        for k,v in d['counters'].items():setattr(self,k,v)
        self.ema=d['ema'];self.layout=d['layout'];self.mask_history=deque(d['mask_history'],maxlen=64)
        # Old sector layouts get a one-time blob initialization. New checkpoints
        # retain their evolving layout; atom IDs, learned weights and EMA persist.
        self.layout_initialized=(d.get('layout_method')==self.layout_method
                                 and d.get('layout_initialized',False))
        self.learned_masks=OrderedDict(d['learned_masks'])
        # JIT closures capture the frozen backbone, so discard initialization
        # executables before installing a checkpoint's backbone.
        for fn in [self.materialize,self.search,self.update,self.evaluate,self.recontext,self.forecast]:fn.clear_cache()
        self.sync_policy()
