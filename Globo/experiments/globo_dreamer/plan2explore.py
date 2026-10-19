"""Plan2Explore exploration objective on the pinned JAX DreamerV3 backbone.

Reference: ramanans1/plan2explore f8ba930, modeltype_1. Five independent
two-layer predictors fit encoder image embeddings from pre-observation deter
and preceding action. Original mean variance reward (scale 10000), no clipping.
DreamerV3 categorical RSSM, REINFORCE actor and return normalization are retained.
"""
import elements
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
from dreamerv3.agent import imag_loss, sg, sample, concat, prefix
from .agent import Agent


def disagreement(predictions):
    return jnp.var(predictions.astype(jnp.float32),axis=0).mean(-1)


def bootstrap_loss(predictions,targets,valid,indices):
    """Resample whole sequences independently per head; exclude reset targets."""
    errors=((predictions-sg(targets)[None])**2).mean(-1)
    errors=jax.vmap(lambda x,i:x[i])(errors,indices)
    weights=valid[indices].astype(jnp.float32)
    return ((errors*weights).sum((1,2))/jnp.maximum(1,weights.sum((1,2)))).sum()


class EmbeddingEnsemble(nj.Module):
    def __init__(self,count,actions,dimension,width=None):
        self.count=count;self.actions=actions;self.dimension=dimension
        self.width=width or dimension

    def __call__(self,deter,action):
        state=nn.cast(sg(deter))
        act=jax.nn.one_hot(action,self.actions,dtype=state.dtype)
        predictions=[]
        for i in range(self.count):
            x=state
            for layer in range(2):
                x=jnp.concatenate([x,act],-1)
                x=jax.nn.relu(self.sub(f'head{i}_hidden{layer}',nn.Linear,self.width)(x))
            x=jnp.concatenate([x,act],-1)
            predictions.append(self.sub(f'head{i}_out',nn.Linear,self.dimension)(x))
        return jnp.stack(predictions).astype(jnp.float32)


class Plan2Explore(Agent):
    def __init__(self,obs_space,act_space,config):
        super().__init__(obs_space,act_space,config)
        enc=config.enc.simple
        if enc.outer or enc.strided:
            raise ValueError('Plan2Explore image target currently requires the standard pooled encoder')
        h,w,_=obs_space['image'].shape
        self.image_dim=(h//2**len(enc.mults))*(w//2**len(enc.mults))*enc.depth*enc.mults[-1]
        self.ensemble=EmbeddingEnsemble(5,int(act_space['action'].classes),self.image_dim,name='p2e_ensemble')
        self.reward_scale=config.explore.get('p2e_scale',10000.)
        self.modules=[self.enc,self.dyn,self.dec,self.rew,self.con,self.pol,self.val,self.ensemble]
        from embodied.jax import Optimizer
        self.opt=Optimizer(self.modules,self._make_opt(**config.opt),summary_depth=1,name='p2e_opt')

    @property
    def policy_keys(self):return '^(enc|dyn|dec|pol|p2e_ensemble)/'

    def policy_input(self,feat,density=None):
        return self.feat2tensor(feat),None

    def policy(self,carry,obs,mode='train'):
        enc,dyn,dec,previous=carry
        enc,_,tokens=self.enc(enc,obs,obs['is_first'],training=False,single=True)
        dyn,_,feat=self.dyn.observe(dyn,tokens,previous,obs['is_first'],training=False,single=True)
        dist=self.pol(self.feat2tensor(feat),1)
        action=sample(dist) if mode=='train' else jax.tree.map(lambda d:d.pred(),dist)
        novelty=disagreement(self.ensemble(feat['deter'],previous['action']))
        novelty=jnp.where(obs['is_first'],0.,novelty)
        return (enc,dyn,dec,action),action,dict(uncertainty=novelty)

    def loss(self,carry,obs,prevact,training):
        enc,dyn,dec=carry;reset=obs['is_first'];B,T=reset.shape
        enc,ee,tokens=self.enc(enc,obs,reset,training)
        dyn,de,losses,feat,metrics=self.dyn.loss(dyn,tokens,prevact,reset,training)
        losses=dict(losses)
        dec,ce,recons=self.dec(dec,feat,reset,training)
        real=self.feat2tensor(feat)
        # Diagnostic reward head cannot shape the world representation or actor.
        losses['rew']=self.rew(sg(real),2).loss(obs['reward'])
        cont=(~obs['is_terminal']).astype(jnp.float32)
        if self.config.contdisc:cont*=1-1/self.config.horizon
        losses['con']=self.con(real,2).loss(cont)
        for key,recon in recons.items():
            value=obs[key].astype(jnp.float32)
            if self.obs_space[key].dtype==jnp.uint8:value/=255
            losses[key]=recon.loss(sg(value))
        # Encoder concatenates vector then image features. Do not predict body
        # or density inputs. deter_t already includes action_(t-1), not image_t.
        target=sg(tokens[...,-self.image_dim:])
        predictions=self.ensemble(feat['deter'],prevact['action'])
        indices=jax.random.randint(nj.seed(),(5,B),0,B)
        eloss=bootstrap_loss(predictions,target,~reset,indices)
        K=min(self.config.imag_last or T,T);H=self.config.imag_length
        starts=self.dyn.starts(de,dyn,K)
        policyfn=lambda f:sample(self.pol(self.feat2tensor(f),1))
        _,future,acts=self.dyn.imagine(starts,policyfn,H,training)
        first=jax.tree.map(lambda x:x[:,-K:].reshape((B*K,1,*x.shape[2:])),feat)
        imagined=concat([sg(first),sg(future)],1)
        last=policyfn(jax.tree.map(lambda x:x[:,-1],imagined))
        actions=concat([acts,jax.tree.map(lambda x:x[:,None],last)],1)
        # DreamerV3 returns use reward[:,1:]: arriving state and action that
        # produced it must align, rather than the next action at that state.
        raw=disagreement(self.ensemble(future['deter'],acts['action']))
        reward=jnp.concatenate([jnp.zeros_like(raw[:,:1]),sg(raw)*self.reward_scale],1)
        x=self.feat2tensor(imagined);con=self.con(x,2).prob(1)
        if self.config.contdisc:con=jnp.clip(con/(1-1/self.config.horizon),0,1)
        ac,_,am=imag_loss(actions,reward,con,self.pol(x,2),self.val(x,2),self.slowval(x,2),
            self.retnorm,self.valnorm,self.advnorm,update=training,contdisc=False,
            horizon=self.config.explore.horizon,**self.config.imag_loss)
        total=sum(self.scales[k]*v.mean() for k,v in losses.items())+eloss+ac['value'].mean()+ac['policy'].mean()
        metrics.update({f'loss/{k}':v.mean() for k,v in losses.items()})
        metrics.update(prefix(am,'novelty_actor'))
        metrics.update({'loss/ensemble':eloss,'loss/novelty_policy':ac['policy'].mean(),
            'loss/novelty_value':ac['value'].mean(),'explore/disagreement':raw.mean(),
            'explore/novelty_reward':reward[:,1:].mean(),'explore/dreamer_action_fraction':jnp.float32(1)})
        return total,((enc,dyn,dec),(ee,de,ce),dict(tokens=tokens,repfeat=feat,losses=losses),metrics)
