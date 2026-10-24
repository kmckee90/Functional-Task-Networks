"""One cortical sheet of isolated deep neurons for the frozen-reservoir explorer.

Bayesian mask inference is vendored unchanged from ftn/projects/bayesian_adv.
TD mask search and masked cross-policy replay follow rl/robotics/train_grid_dqn.
No trainable preprocessing, inter-neuron weights, or recurrent weight learning.
"""
from collections import deque
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .legacy.bayes_reconfig import BayesMaskReconfig
from .scalable_replay import cpu_tree, trajectory


class Sheet(nn.Module):
    def __init__(self, shape, actions, variant, budget, width=8, depth=8, active_neurons=None):
        super().__init__()
        self.shape=tuple(shape);self.actions=actions;self.variant=variant
        c,h,w=shape
        self.positions=1 if variant=='flat' else h*w
        inputs=c*h*w if variant=='flat' else c*3*3
        sizes=[inputs]+[width]*depth+[1]
        neuron=sum((a+1)*b for a,b in zip(sizes,sizes[1:]))
        # Both existing value streams count toward the old trainable budget.
        per_atom=neuron+self.positions*actions*2
        self.k=max(1,round(budget/per_atom)) if active_neurons is None else int(active_neurons)
        if self.k < 1:raise ValueError('FTN active neuron count must be positive')
        self.H=self.k*8
        self.weights=nn.ParameterList()
        self.biases=nn.ParameterList()
        for a,b in zip(sizes,sizes[1:]):
            weight=nn.Parameter(torch.empty(self.H,a,b))
            nn.init.uniform_(weight,-math.sqrt(6/a),math.sqrt(6/a))
            self.weights.append(weight)
            bias=nn.Parameter(torch.empty(self.H,b))
            nn.init.uniform_(bias,-.1/math.sqrt(a),.1/math.sqrt(a))
            self.biases.append(bias)
        # Equivalent to a linear layer on flattened masked neuron maps.
        # Axis zero is atom ID for every parameter, including the decoder.
        self.decoder=nn.Parameter(torch.empty(self.H,self.positions,2*actions))
        nn.init.uniform_(self.decoder,-1/math.sqrt(self.k*self.positions),
                         1/math.sqrt(self.k*self.positions))
        self.register_buffer('mask',torch.zeros(self.H))
        self.mask[:self.k]=1
        self.spec=dict(variant=variant,depth=depth,width=width,k=self.k,neurons=self.H,
            baseline_parameters=budget,active_parameters=self.k*per_atom,
            bank_parameters=self.H*per_atom,relative_error=self.k*per_atom/budget-1,
            reservoir_shape=list(shape),positions=self.positions,parameter_isolation=True,
            mask_count=1,kernel=3 if variant=='conv' else None)

    def features(self,z,ids=None):
        if ids is None:ids=torch.arange(self.H,device=z.device)
        if self.variant=='flat':x=z.flatten(1)[:,None,:]
        else:x=F.unfold(z,kernel_size=3,padding=1).transpose(1,2)
        # B,P,N,W. Neuron N never receives another neuron's activations.
        x=torch.einsum('bpd,ndw->bpnw',x,self.weights[0][ids])+self.biases[0][ids]
        x=F.relu(x)
        for i in range(1,len(self.weights)):
            x=torch.einsum('bpni,nij->bpnj',x,self.weights[i][ids])+self.biases[i][ids]
            if i<len(self.weights)-1:x=F.relu(x)
        return torch.tanh(x[...,0]).permute(0,2,1)

    def forward(self,z,mask=None):
        mask=self.mask if mask is None else mask
        ids=torch.nonzero(mask,as_tuple=False).flatten()
        return torch.einsum('bnp,npa->ba',self.features(z,ids)*mask[ids][None,:,None],self.decoder[ids])

    @torch.no_grad()
    def contributions(self,z):
        return torch.einsum('bnp,npa->bna',self.features(z),self.decoder)


class Stream(nn.Module):
    def __init__(self,sheet,index):
        super().__init__();self.sheet=sheet;self.index=index
    def forward(self,z):
        a=self.sheet.actions
        return [self.sheet(z)[:,self.index*a:(self.index+1)*a]]


class IsolatedRMSprop(torch.optim.Optimizer):
    """No momentum/decay updates to inactive neurons or their optimizer state."""
    def __init__(self,params,lr):super().__init__(params,dict(lr=lr))
    @torch.no_grad()
    def step(self,ids):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue
                state=self.state[p]
                if not state:state['square_avg']=torch.zeros_like(p)
                g=p.grad[ids];avg=state['square_avg'][ids]*.99+g.square()*.01
                state['square_avg'].index_copy_(0,ids,avg)
                p.index_add_(0,ids,-group['lr']*g/(avg.sqrt()+1e-8))


class Selector(BayesMaskReconfig,nn.Module):
    def __init__(self,H,k,actions,gamma,gamma_env):
        nn.Module.__init__(self)
        self.hidden_size=H;self.k=k;self.actions=actions
        self.gamma=gamma;self.gamma_env=gamma_env
        self.reconfig_steps=2;self.reconfig_sample_size=64;self.reconfig_noise_scale=1.
        self.register_buffer('m',torch.zeros(H))
        self._init_bayes(bayes_reconfig_steps=2,bayes_changepoint_reconfig_steps=2)
        self.m.copy_(self._sample_mask(self.mask_logits,1.,True))
        self.loss=self.td_loss
    def _topk_mask(self,scores):
        return torch.zeros_like(scores).scatter_(0,scores.topk(self.k).indices,1.)
    def _run(self,x,mask,detach):return torch.einsum('bna,n->ba',x,mask)
    def td_loss(self,pred,y):
        a=self.actions
        qi,qe,ni,ne=pred.split(a,dim=-1)
        with torch.no_grad():
            nxt=(ni+ne).argmax(-1,keepdim=True)
            ti=y[:,1]+self.gamma*y[:,3]*ni.gather(1,nxt)[:,0]
            te=y[:,2]+self.gamma_env*y[:,3]*ne.gather(1,nxt)[:,0]
        action=y[:,0].long()[:,None]
        return F.mse_loss(qi.gather(1,action)[:,0],ti)+F.mse_loss(qe.gather(1,action)[:,0],te)


class M2EFTN:
    """Install a sheet into M2E with bounded-dwell frontier selection."""
    def __init__(self,agent,variant,active_neurons=None):
        self.agent=agent
        from .frontier_rotation import FrontierRotation
        agent.frontier_rotation=FrontierRotation()
        budget=sum(p.numel() for p in agent._trainable_params)
        self.q=Sheet(agent.hidden_shape,agent.head_sizes[0],variant,budget,
            active_neurons=active_neurons).to(agent.device)
        self.target=deepcopy(self.q).requires_grad_(False)
        self.selector=Selector(self.q.H,self.q.k,self.q.actions,agent.gamma,agent.gamma_env).to(agent.device)
        self.q.mask.copy_(self.selector.m);self.target.mask.copy_(self.selector.m)
        agent.q_net=Stream(self.q,0);agent.q_env=Stream(self.q,1)
        agent.q_target=Stream(self.target,0);agent.q_env_target=Stream(self.target,1)
        agent._trainable_params=list(self.q.parameters())
        agent.optimizer=IsolatedRMSprop(agent._trainable_params,agent.optimizer.param_groups[0]['lr'])
        # Bounded replay of supports and the assemblies they trained, independent
        # of frontier eviction. No task labels and no unbounded mask registry.
        self.replay=[];self.replay_updates=0;self.updates=0
        self.searches=0;self.changepoints=0;self.loss_mean=0.;self.loss_var=0.;self.loss_n=0
        self.mask_turnover=0.;self.visited_atoms=torch.zeros(self.q.H,dtype=torch.bool,device=agent.device)
        self.action_ema=np.zeros((self.q.H,3),np.float32)
        self.mask_history=deque(maxlen=16)
        h=math.isqrt(self.q.H)
        while self.q.H%h:h-=1
        self.grid=(h,self.q.H//h);self.layout=np.arange(self.q.H);self.layout_initialized=False
        self.original_act=agent.act;agent.act=self.act;agent._td_update=self.update
        self.action_mask=self.selector.m.clone()

    @torch.no_grad()
    def data(self,B,idx,is_goal):
        d={k:v.detach().clone() for k,v in B.gather(idx).items()}
        d['is_goal']=is_goal
        d['frontier_endpoint']=bool(is_goal)
        return d

    @torch.no_grad()
    def labels(self,d):
        a=self.agent;nov=a._novelty(d['emb'])
        reward=a.online_coef*d['online_nov']+a.offline_coef*nov
        bonus_index=-1 if d.get('frontier_endpoint',False) else nov.argmax()
        reward[bonus_index]*=a.goal_sparse_reward_scale if d['is_goal'] else a.main_sparse_reward_scale
        env=a.env_reward_coef*d['env_r']-a.death_penalty*d['terminal'].float()
        if getattr(a,'survival',False):
            # Keep the boosted frontier reward; quarter other novelty rewards.
            # Match death cost to that batch's original positive goal target.
            goal_reward=reward[bonus_index].clone().clamp_min(0)
            reward*=getattr(a,'survival_novelty_scale',.25)
            reward[bonus_index]=goal_reward
            env=-d['env_r'].lt(0).float()*goal_reward
        return torch.stack((d['action'][:,0].float(),reward,env,(~d['terminal']).float()),-1)

    def search(self,d):
        # Cache full-bank detached contributions once; two mask steps are cheap.
        with torch.no_grad():
            x=torch.cat((self.q.contributions(d['z'].float()),
                         self.target.contributions(d['z_next'].float())),-1)
            y=self.labels(d)
            err=float(self.selector.td_loss(self.selector._run(x,self.selector.m,True),y))
        cp=self.loss_n>=32 and err>self.loss_mean+3*max(math.sqrt(self.loss_var),1e-4)
        previous=self.selector.m.clone()
        if cp:self.selector.reconfigure_changepoint(x,y);self.changepoints+=1
        else:self.selector.reconfigure(x,y)
        self.mask_turnover=1-float((previous*self.selector.m).sum())/self.q.k
        self.visited_atoms|=self.selector.m.bool()
        delta=err-self.loss_mean
        self.loss_mean+=.03*delta;self.loss_var=.97*(self.loss_var+.03*delta*delta);self.loss_n+=1
        self.q.mask.copy_(self.selector.m);self.target.mask.copy_(self.selector.m)
        self.searches+=1

    def loss(self,d,mask):
        current=self.q(d['z'].float(),mask)
        with torch.no_grad():nxt=self.target(d['z_next'].float(),mask)
        return self.selector.td_loss(torch.cat((current,nxt),-1),self.labels(d))

    def update(self,B,idx,is_goal):
        if not len(idx):return None
        d=self.data(B,idx,is_goal);self.search(d)
        mask=self.selector.m.detach().clone()
        opt=self.agent.optimizer;opt.zero_grad(set_to_none=True)
        loss=self.loss(d,mask);value=float(loss.detach());loss.backward()
        active=mask.bool()
        overlap_value=0.
        if self.replay:
            # Same 0.2–0.8 overlap filter as the grid DQN experiment.
            masks=torch.stack([v['mask'] for v in self.replay])
            overlap=(masks*mask).sum(-1)/self.q.k
            eligible=torch.nonzero((overlap>=.2)&(overlap<=.8)).flatten()
            if eligible.numel():
                j=int(eligible[torch.randint(len(eligible),(1,),device=eligible.device)])
                old=self.replay[j]
                replay_loss=self.loss(old['data'],old['mask']);replay_loss.backward()
                active|=old['mask'].bool();self.replay_updates+=1;overlap_value=float(overlap[j])
        torch.nn.utils.clip_grad_norm_(self.q.parameters(),self.agent.grad_clip)
        opt.step(torch.nonzero(active).flatten())
        with torch.no_grad():
            for p,t in zip(self.q.parameters(),self.target.parameters()):
                t.lerp_(p,self.agent.tau)
        self.updates+=1
        # Sampling every 8 updates avoids filling the archive with one burst.
        if self.updates%8==0:
            d['z']=d['z'].half();d['z_next']=d['z_next'].half()
            entry=dict(mask=mask,data=d)
            if len(self.replay)<256:self.replay.append(entry)
            else:
                # Uniform lifetime reservoir, not just the last training burst.
                slot=int(torch.randint(self.updates//8,(1,)))
                if slot<256:self.replay[slot]=entry
        self.agent._last_metrics.update(ftn_loss=value,ftn_replay_updates=self.replay_updates,
            ftn_overlap=overlap_value,ftn_changepoints=self.changepoints,ftn_searches=self.searches)
        return value

    def act(self,z,greedy=False):
        a=self.agent
        if a.cur_step>=500 and a.cur_step%8==0 and a.memory.size:
            idx=trajectory(a.memory,(a.memory.ptr-1)%a.memory.capacity,64)
            self.search(self.data(a.memory,idx,False))
        self.action_mask=self.selector.m.detach().clone()
        with torch.no_grad():q=self.q(z[None]).reshape(1,2,self.q.actions).sum(1)
        if not greedy and torch.rand(1).item()<a.epsilon:
            return [int(torch.randint(self.q.actions,(1,)))]
        return [int(q.argmax())]

    def frame(self):
        mask=self.action_mask.cpu().numpy()
        self.action_ema=np.maximum(self.action_ema*np.array([.95,.30,.975],np.float32),mask[:,None])
        if not self.mask_history or not np.array_equal(mask,self.mask_history[-1]):
            self.mask_history.append(mask.copy())
        return np.uint8(np.clip(self.action_ema,0,1)*255)

    def consolidate(self):
        from .ftn_spatial_permute import spatial_permutation
        if self.mask_history:
            self.layout=spatial_permutation(np.array(self.mask_history),self.grid,
                self.layout if self.layout_initialized else None,sweeps=8,
                initialization='random_blobs',center=.25,roundness=2.,radius=7,temperature=.05,seed=0)
            self.layout_initialized=True

    def save(self):
        return cpu_tree(dict(frontier_rotation=self.agent.frontier_rotation.save(),selector=self.selector.state_dict(),replay=list(self.replay),
            updates=self.updates,replay_updates=self.replay_updates,searches=self.searches,
            changepoints=self.changepoints,loss_mean=self.loss_mean,loss_var=self.loss_var,loss_n=self.loss_n,
            mask_turnover=self.mask_turnover,visited_atoms=self.visited_atoms,
            action_ema=self.action_ema,mask_history=list(self.mask_history),layout=self.layout,
            layout_initialized=self.layout_initialized,spec=self.q.spec))

    def load(self,s):
        if s['spec']!=self.q.spec:raise ValueError('FTN architecture changed on resume')
        self.agent.frontier_rotation.load(s.get('frontier_rotation'))
        self.selector.load_state_dict(s['selector'])
        self.mask_turnover=s.get('mask_turnover',0.)
        self.visited_atoms.copy_(s.get('visited_atoms',self.selector.m.bool()).to(self.agent.device))
        def device_tree(v):
            if isinstance(v,torch.Tensor):return v.to(self.agent.device)
            if isinstance(v,dict):return {k:device_tree(x) for k,x in v.items()}
            return v
        self.replay=[device_tree(x) for x in s['replay']]
        self.mask_history=deque(s['mask_history'],maxlen=16)
        for k in ('updates','replay_updates','searches','changepoints','loss_mean','loss_var','loss_n',
                  'action_ema','layout','layout_initialized'):setattr(self,k,s[k])
        self.q.mask.copy_(self.selector.m);self.target.mask.copy_(self.selector.m)


def render(root,rows,step,ftn):
    """Synchronized original pixels and actual action mask, one two-panel movie."""
    import av
    from PIL import Image,ImageDraw
    from .env import atomic_json
    ftn.consolidate()
    out=av.open(str(root/'obs_pending.mp4'),'w');stream=out.add_stream('libx264',rate=6)
    goals=hasattr(ftn,'options')
    offline_goals='log/offline_goal' in rows[0][1]
    width=768 if goals or offline_goals else 512
    stream.width=width;stream.height=280;stream.pix_fmt='yuv420p'
    for n,row in rows:
        canvas=Image.new('RGB',(width,280),'#171c24')
        canvas.paste(Image.fromarray(row['log/human']),(0,24))
        mask=row['log/m2e_ftn_ema'][ftn.layout].reshape(*ftn.grid,3)
        canvas.paste(Image.fromarray(mask).resize((256,256),Image.Resampling.NEAREST),(256,24))
        draw=ImageDraw.Draw(canvas);draw.text((6,5),f'Step {n:,}',fill='white')
        draw.text((262,5),'FTN mask / EMA',fill='white')
        if offline_goals:
            draw.text((518,5),'Offline frontier (density view)',fill='white')
            goal_step=int(row['log/offline_goal_step'])
            if goal_step>=0:
                canvas.paste(Image.fromarray(row['log/offline_goal']).convert('RGB').resize((256,128),Image.Resampling.NEAREST),(512,80))
                draw.text((518,240),f'Observed at step {goal_step:,}',fill='white')
            else:draw.text((518,100),'Collecting initial experience',fill='white')
        if goals:
            import io
            goal=int(row.get('log/goal_id',0));final=int(row.get('log/frontier_id',0))
            draw.text((518,5),'Goal observation',fill='white')
            if ftn.mode=='frontier':
                if goal:canvas.paste(Image.open(io.BytesIO(ftn.options[goal]['image'])),(512,24))
                draw.text((518,30),f'Goal {goal}' if goal else 'Density exploration',fill='white',stroke_width=1,stroke_fill='black')
            else:
                for label,g,y in (('Waypoint',goal,24),('Frontier',final,152)):
                    if g:canvas.paste(Image.open(io.BytesIO(ftn.options[g]['image'])).resize((128,128)),(512,y))
                    draw.text((646,y+8),f'{label} {g}' if g else 'Exploring',fill='white')
        for packet in stream.encode(av.VideoFrame.from_image(canvas)):out.mux(packet)
    for packet in stream.encode():out.mux(packet)
    out.close();(root/'obs_pending.mp4').replace(root/'obs.mp4')
    atomic_json(root/'video.json',dict(step=step,time=__import__('time').time(),
        frame_start=rows[0][0],frame_end=rows[-1][0],frame_age=step-rows[-1][0],frames=len(rows),
        playback_seconds=len(rows)/6,panels=['original 256x256 observation','concurrent action mask with EMA'],
        model_free_only=True,m2e_ftn=True,frontier_mode=getattr(ftn,'mode',None),
        goal_panels=goals or offline_goals,offline_goal=offline_goals,layout='SpatialPermuteRandomCenter',grid=ftn.grid))
