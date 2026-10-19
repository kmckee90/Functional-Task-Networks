"""Goal-indexed Bayesian FTN options: direct frontiers or composed waypoints.

Goal keys, detectors, support and posteriors are retained. Novelty changes the
goal's priority, never its reaching reward. No position or engine state is used.
"""
from collections import OrderedDict
import heapq
import io
import math
import random
import torch
from torch.nn import functional as F
from .m2e_ftn import M2EFTN
from .scalable_replay import cpu_tree, trajectory


class FrontierFTN(M2EFTN):
    capacity=1024
    def __init__(self,agent,mode,root):
        super().__init__(agent,'conv')
        self.mode=mode;self.root=root;self.rng=random.Random(9107)
        self.q.spec.update(goal_mode=mode,goal_capacity=self.capacity,goal_gamma=.98)
        self.options=[];self.images=OrderedDict();self.edges={}
        self.training_goal=0;self.active_goal=0;self.final_goal=0
        self.action_goal=0;self.action_final=0;self.route=[]
        self.goal_started=0;self.explore_until=500;self.last_decision=-1
        self.goal_switches=0;self.goal_retrievals=0;self.goal_successes=0;self.goal_timeouts=0
        self.plans=0;self.planned_hops=0;self.mask_accepts=0;self.mask_searches=0
        self.last_goal_metrics={};self.last_goal_publish=-1
        self.last_final_choices=[];self.final_started=0
        self.search_positive=False
        self.options.append(self.new_option(None,None,None,None))  # density exploration
        agent.train_round=self.train_round
        # Termination on reaching a goal gives bounded [0,1] reaching values.
        self.selector.loss=self.search_loss

    def new_option(self,emb,z,image,radius):
        state=self.selector.get_reconfig_state()
        if emb is not None:
            state['mask_logits'].zero_()
            state['m']=self.selector._sample_mask(state['mask_logits'],1.,True)
        return dict(emb=emb,context=None if z is None else self.context(z),image=image,
            radius=radius,posterior=state,supports=[],seen=0,updates=0,frontier=False,
            mean=0.,var=0.,n=0,visits=0,attempts=0,successes=0,timeouts=0,
            last_attempt=-100000,last_seen=self.agent.cur_step)

    @staticmethod
    def context(z):return F.normalize(z.float().mean((-2,-1)),dim=-1)

    def record_observation(self,image,is_first):
        # Same model_step key written by ModelFreeBehavior.remember(). Store only
        # a bounded recent image cache; registered goal images are retained apart.
        if is_first or not self.agent.memory.size:return
        from PIL import Image
        b=io.BytesIO();Image.fromarray(image).save(b,format='JPEG',quality=85)
        self.images[self.agent.cur_step-1]=b.getvalue()
        while len(self.images)>20000:self.images.popitem(last=False)

    def on_reset(self):
        self.active_goal=self.final_goal=0;self.route=[]
        self.explore_until=max(self.agent.cur_step+16,500)

    @torch.no_grad()
    def matches(self,emb,z):
        if len(self.options)==1:return None,None
        prototypes=torch.stack([o['emb'] for o in self.options[1:]])
        contexts=torch.stack([o['context'] for o in self.options[1:]])
        distances=(prototypes-emb).square().mean(-1).sqrt()
        similarity=contexts@self.context(z)
        return distances,similarity

    @torch.no_grad()
    def register(self,d,end,frontier=False):
        emb=d['emb'][end].detach().clone();z=d['z_next'][end].float()
        # Reject nearly uniform views rather than assigning a sky-return option.
        if float(emb.std())<.025:return 0
        distance,similarity=self.matches(emb,z)
        goal=0
        if distance is not None:
            radii=distance.new_tensor([min(.055,o['radius']) for o in self.options[1:]])
            candidates=torch.nonzero((distance<=radii)&(similarity>=.5)).flatten()
            if candidates.numel():goal=int(candidates[distance[candidates].argmin()])+1
        if not goal:
            image=self.images.get(int(d['model_step'][end]))
            if image is None or len(self.options)>=self.capacity:return 0
            # Fixed detector: conservative tolerance from the approach's final
            # observations. Never adapt the target to follow the agent.
            tail=d['emb'][max(0,end-3):end+1]
            radius=float((tail-emb).square().mean(-1).sqrt().median().clamp(.025,.06))
            self.options.append(self.new_option(emb,z,image,radius));goal=len(self.options)-1
        o=self.options[goal];o['frontier']|=frontier;o['last_seen']=self.agent.cur_step
        start=max(0,end-(64 if self.mode=='frontier' else 16)+1)
        support={k:v[start:end+1].detach().clone() if isinstance(v,torch.Tensor) else v for k,v in d.items()}
        support['z']=support['z'].half();support['z_next']=support['z_next'].half()
        support['positive']=True
        o['seen']+=1
        if len(o['supports'])<3:o['supports'].append(support)
        else:
            # Keep one lifetime sample, the original successful approach, and
            # the latest approach (new start contexts improve the same policy).
            if self.rng.randrange(o['seen'])<2:o['supports'][1]=support
            o['supports'][2]=support
        return goal

    def register_path(self,d):
        n=len(d['emb'])
        if self.mode=='frontier':return self.register(d,n-1,True)
        previous=0
        endpoints=list(range(0,n,8))
        if not endpoints or endpoints[-1]!=n-1:endpoints.append(n-1)
        for end in endpoints:
            goal=self.register(d,end,end==n-1)
            if previous and goal and previous!=goal:
                self.edges.setdefault(previous,{})[goal]=self.edges.get(previous,{}).get(goal,0)+1
            if goal:previous=goal
        return previous

    @torch.no_grad()
    def labels(self,d):
        if self.training_goal==0:return super().labels(d)
        o=self.options[self.training_goal]
        delta=(d['emb']-o['emb']).square().mean(-1).sqrt()
        similarity=self.context(d['z_next'])@o['context']
        hit=(delta<=o['radius'])&(similarity>=.5)
        terminal=d['terminal']|hit
        return torch.stack((d['action'][:,0].float(),hit.float(),torch.zeros_like(delta),(~terminal).float()),-1)

    def search_loss(self,pred,y):
        if self.training_goal==0:return self.selector.td_loss(pred,y)
        a=self.q.actions;qi,qe,ni,ne=pred.split(a,dim=-1)
        with torch.no_grad():target=y[:,1]+.98*y[:,3]*ni.max(-1).values.clamp(0,1)
        action=y[:,0].long()[:,None]
        loss=F.mse_loss(qi.gather(1,action)[:,0],target)+F.mse_loss(qe.gather(1,action)[:,0],torch.zeros_like(target))
        # Conservative offline TD discourages unsupported high-valued actions.
        loss+=.02*(torch.logsumexp(qi,dim=-1)-qi.gather(1,action)[:,0]).mean()
        if self.search_positive:loss+=.05*F.cross_entropy(qi/.2,y[:,0].long())
        return loss

    def restore(self,goal):
        self.training_goal=goal
        self.selector.set_reconfig_state(self.options[goal]['posterior'])
        self.q.mask.copy_(self.selector.m);self.target.mask.copy_(self.selector.m)

    def search(self,d):
        o=self.options[self.training_goal]
        self.search_positive=d.get('positive',False)
        with torch.no_grad():
            x=torch.cat((self.q.contributions(d['z'].float()),self.target.contributions(d['z_next'].float())),-1)
            y=self.labels(d);old=self.selector.m.clone()
            before=float(self.search_loss(self.selector._run(x,old,True),y))
        cp=o['n']>=32 and before>o['mean']+3*max(math.sqrt(o['var']),1e-4)
        if cp:self.selector.reconfigure_changepoint(x,y);self.changepoints+=1
        else:self.selector.reconfigure(x,y)
        with torch.no_grad():
            # Search can explore, but published goal masks are deterministic and
            # change only when the new assembly improves the support objective.
            candidate=self.selector._topk_mask(self.selector.mask_logits)
            after=float(self.search_loss(self.selector._run(x,candidate,True),y))
            accept=after<before-1e-6
            self.selector.m.copy_(candidate if accept else old)
        self.mask_accepts+=int(accept);self.mask_searches+=1
        delta=before-o['mean'];o['mean']+=.03*delta;o['var']=.97*(o['var']+.03*delta*delta);o['n']+=1
        o['posterior']=self.selector.get_reconfig_state()
        self.mask_turnover=1-float((old*self.selector.m).sum())/self.q.k
        self.visited_atoms|=self.selector.m.bool()
        self.q.mask.copy_(self.selector.m);self.target.mask.copy_(self.selector.m)
        self.searches+=1;self.loss_mean=o['mean']

    def loss(self,d,mask):
        current=self.q(d['z'].float(),mask)
        with torch.no_grad():nxt=self.target(d['z_next'].float(),mask)
        previous=self.search_positive;self.search_positive=d.get('positive',False)
        value=self.search_loss(torch.cat((current,nxt),-1),self.labels(d))
        self.search_positive=previous
        return value

    def remember_support(self,goal,d):
        if goal:return
        copy={k:(v.half() if k in ('z','z_next') else v).detach().clone() if isinstance(v,torch.Tensor) else v for k,v in d.items()}
        o=self.options[0];o['seen']+=1
        if len(o['supports'])<3:o['supports'].append(copy)
        else:o['supports'][self.rng.randrange(3)]=copy

    def update_option(self,goal,d):
        self.restore(goal);self.search(d)
        mask=self.selector.m.clone();active=mask.bool();opt=self.agent.optimizer
        opt.zero_grad(set_to_none=True)
        loss=self.loss(d,mask);value=float(loss.detach());loss.backward()
        # Protect all overlapping retained goals, including nearly identical
        # masks. Weight updates on shared neurons see both objectives.
        candidates=[i for i,o in enumerate(self.options) if i!=goal and o['updates'] and o['supports']]
        if candidates:
            sampled=self.rng.sample(candidates,min(32,len(candidates)))
            overlaps=torch.stack([(self.options[i]['posterior']['m']*mask).sum() for i in sampled])
            valid=torch.nonzero(overlaps>0).flatten()
            if valid.numel():
                chosen=int(torch.multinomial(overlaps,1));other=sampled[chosen]
                self.training_goal=other;o=self.options[other]
                other_mask=o['posterior']['m']
                replay=self.rng.choice(o['supports'])
                self.loss(replay,other_mask).backward();active|=other_mask.bool();self.replay_updates+=1
        self.training_goal=goal
        torch.nn.utils.clip_grad_norm_(self.q.parameters(),self.agent.grad_clip)
        opt.step(torch.nonzero(active).flatten())
        with torch.no_grad():
            for p,t in zip(self.q.parameters(),self.target.parameters()):t.lerp_(p,self.agent.tau)
        self.options[goal]['updates']+=1;self.updates+=1
        self.remember_support(goal,d)
        return value

    def train_round(self,n_updates):
        a=self.agent;M=a.memory
        if M.size<a.minibatch_size:return None
        with torch.no_grad():
            # New goal images come from a bounded recent cache. Old registered
            # destinations remain selectable and trainable indefinitely.
            count=min(M.size,20000);end=(M.ptr-1)%M.capacity
            scan=(end-torch.randint(count,(min(4096,count),),device=a.device))%M.capacity
            from .frontier_rotation import candidates
            pool=candidates(a,M,'memory',scan)
            rotation=a.frontier_rotation;old=rotation.current
            valid=old is not None and old['index']<M.size and int(M.model_step[old['index']])==old['id']
            selected=rotation.choose(pool,a.cur_step,valid)
            front=selected['index']
        idx=trajectory(M,front,64);d=self.data(M,idx,True)
        a.goal.insert(d)
        discovered=self.register_path(d)
        losses=[]
        for i in range(n_updates):
            goals=[j for j,o in enumerate(self.options) if j and o['supports']]
            if i%5==0 or not goals:
                endpoint=(M.ptr-1)%M.capacity if i%2 else int(torch.randint(M.size,(1,)))
                batch=self.data(M,trajectory(M,endpoint,64),False)
                losses.append(self.update_option(0,batch));continue
            if i%3==0 and self.active_goal:goal=self.active_goal
            elif i%3==1 and discovered:goal=discovered
            else:goal=self.rng.choice(goals)
            batch=self.rng.choice(self.options[goal]['supports'])
            losses.append(self.update_option(goal,batch))
        # Offline sampling never changes which policy the acting agent requested.
        self.restore(self.active_goal)
        self.publish()
        a._last_metrics=dict(frontier_novelty=selected['novelty'],goal_buffer=float(a.goal.size),
            **rotation.metrics(a.cur_step),**self.metrics())
        return sum(losses)/len(losses)

    def metrics(self):
        return dict(ftn_goals=len(self.options)-1,ftn_active_goal=self.action_goal,
            ftn_frontier_goal=self.action_final,ftn_goal_retrievals=self.goal_retrievals,
            ftn_goal_successes=self.goal_successes,ftn_goal_timeouts=self.goal_timeouts,
            ftn_goal_switches=self.goal_switches,ftn_plans=self.plans,ftn_planned_hops=self.planned_hops,
            ftn_goal_mask_accepts=self.mask_accepts,ftn_goal_searches=self.mask_searches)

    @torch.no_grad()
    def reached(self,goal,emb,z):
        if not goal:return False
        o=self.options[goal]
        return bool((emb-o['emb']).square().mean().sqrt()<=o['radius'] and self.context(z)@o['context']>=.5)

    def shortest_paths(self,start):
        # Directed edges exist only for observed trajectories; edge cost includes
        # subsequently measured option failures, not a geometric oracle.
        distance={start:0.};paths={start:[]};queue=[(0.,start)]
        while queue:
            cost,node=heapq.heappop(queue)
            if cost!=distance[node]:continue
            for nxt,count in self.edges.get(node,{}).items():
                o=self.options[nxt]
                if not o['updates']:continue
                nc=cost+1.+2*o['timeouts']/max(1,o['attempts'])
                if nc<distance.get(nxt,float('inf')):
                    distance[nxt]=nc;paths[nxt]=paths[node]+[nxt];heapq.heappush(queue,(nc,nxt))
        return distance,paths

    @torch.no_grad()
    def choose(self,z,emb):
        eligible=[i for i,o in enumerate(self.options) if i and o['updates']>=2 and o['frontier']]
        if not eligible:return []
        contributions=self.q.contributions(z[None])[0,:,:self.q.actions]
        masks=torch.stack([self.options[i]['posterior']['m'] for i in eligible])
        values=(masks@contributions).max(-1).values.clamp(.03,1.)
        novelty=self.agent._novelty(torch.stack([self.options[i]['emb'] for i in eligible]))
        priority=(novelty+1e-4)/(novelty.max()+1e-4)
        paths={i:[i] for i in eligible};costs={i:1. for i in eligible}
        if self.mode=='waypoints':
            d,s=self.matches(emb,z)
            nearby=torch.nonzero((d<.10)&(s>.3)).flatten()
            if not nearby.numel():return []
            start=int(nearby[d[nearby].argmin()])+1
            costs,paths=self.shortest_paths(start)
        scores=[]
        for j,i in enumerate(eligible):
            o=self.options[i]
            if i not in paths or not paths[i] or self.reached(i,emb,z):scores.append(-1.);continue
            if self.agent.cur_step-o['last_attempt']<64:scores.append(-1.);continue
            success=(o['successes']+1)/(o['attempts']+2)
            # Novelty decides desirability. Reach values or observed graph paths
            # decide accessibility. Neither changes the option's reward labels.
            accessibility=float(values[j]) if self.mode=='frontier' else 1/math.sqrt(costs[i])
            scores.append(float(priority[j])*(.2+accessibility)*success)
        ranked=sorted((j for j,s in enumerate(scores) if s>=0),key=lambda j:scores[j],reverse=True)[:5]
        if not ranked:return []
        # A failed/completed option should not immediately select the same final
        # destination again. Explore instead if no alternative is accessible.
        if self.last_final_choices:
            ranked=[j for j in ranked if eligible[j]!=self.last_final_choices[-1]]
        if not ranked:return []
        fresh=[j for j in ranked if eligible[j] not in self.last_final_choices[-4:]]
        best=self.rng.choice(fresh or ranked)
        self.last_final_choices=(self.last_final_choices+[eligible[best]])[-4:]
        self.final_started=self.agent.cur_step
        self.plans+=int(self.mode=='waypoints');self.planned_hops+=len(paths[eligible[best]])
        return paths[eligible[best]].copy()

    def activate(self,goal):
        o=self.options[goal]
        self.goal_retrievals+=int(o['attempts']>0);o['attempts']+=1
        o['last_attempt']=self.agent.cur_step
        self.active_goal=goal;self.goal_started=self.agent.cur_step;self.goal_switches+=1
        self.restore(goal)

    def act(self,z,greedy=False):
        a=self.agent;step=a.cur_step
        emb=a.embed(self.current_image)
        if self.active_goal and step-self.final_started>=1000:
            self.options[self.active_goal]['timeouts']+=1;self.goal_timeouts+=1
            self.active_goal=self.final_goal=0;self.route=[]
            self.explore_until=step+32;self.restore(0)
        if self.active_goal and step-self.goal_started>=4:
            hit=self.reached(self.active_goal,emb,z)
            timeout=step-self.goal_started>=(128 if self.mode=='frontier' else 64)
            if hit or timeout:
                o=self.options[self.active_goal]
                o['successes']+=int(hit);o['timeouts']+=int(timeout and not hit)
                self.goal_successes+=int(hit);self.goal_timeouts+=int(timeout and not hit)
                if hit and self.route:self.activate(self.route.pop(0))
                else:
                    self.active_goal=self.final_goal=0;self.route=[]
                    self.explore_until=step+32;self.restore(0)
        if not self.active_goal and step>=self.explore_until and step!=self.last_decision:
            self.last_decision=step
            path=self.choose(z,emb)
            if path:
                self.final_goal=path[-1];self.route=path[1:];self.activate(path[0])
            else:self.explore_until=step+16
        self.restore(self.active_goal)
        self.action_goal=self.active_goal;self.action_final=self.final_goal
        self.action_mask=self.selector.m.clone()
        with torch.no_grad():
            output=self.q(z[None]);q=output[:,:self.q.actions]
            if not self.active_goal:q=q+output[:,self.q.actions:]
        self.agent._last_metrics.update(self.metrics())
        if self.active_goal!=self.last_goal_publish:
            self.publish();self.last_goal_publish=self.active_goal
        if not greedy and self.rng.random()<a.epsilon:return [self.rng.randrange(self.q.actions)]
        return [int(q.argmax())]

    def publish(self):
        from .env import atomic_json
        live=self.root/'live';live.mkdir(exist_ok=True)
        for key,goal in (('active_goal',self.active_goal),('frontier_goal',self.final_goal)):
            if goal:
                path=live/(key+'.jpg');tmp=path.with_suffix('.tmp');tmp.write_bytes(self.options[goal]['image']);tmp.replace(path)
        atomic_json(live/'goal_status.json',dict(step=self.agent.cur_step,mode=self.mode,
            active_goal=self.active_goal,frontier_goal=self.final_goal,remaining_route=self.route,
            nodes=len(self.options)-1,edges=sum(map(len,self.edges.values())),**self.metrics()))

    def save(self):
        s=super().save()
        s['frontiers']=cpu_tree(dict(mode=self.mode,options=self.options,edges=self.edges,
            last_final_choices=self.last_final_choices,final_started=self.final_started,
            rng=self.rng.getstate(),active_goal=self.active_goal,final_goal=self.final_goal,route=self.route,
            goal_started=self.goal_started,explore_until=self.explore_until,last_decision=self.last_decision,
            goal_switches=self.goal_switches,goal_retrievals=self.goal_retrievals,goal_successes=self.goal_successes,
            goal_timeouts=self.goal_timeouts,plans=self.plans,planned_hops=self.planned_hops,
            mask_accepts=self.mask_accepts,mask_searches=self.mask_searches))
        return s

    def load(self,s):
        super().load(s)
        def move(v):
            if isinstance(v,torch.Tensor):return v.to(self.agent.device)
            if isinstance(v,dict):return {k:move(x) for k,x in v.items()}
            if isinstance(v,list):return [move(x) for x in v]
            return v
        values=move(s['frontiers'])
        if values['mode']!=self.mode:raise ValueError('Frontier mode changed on resume')
        self.rng.setstate(values.pop('rng'))
        for key,value in values.items():setattr(self,key,value)
        self.restore(self.active_goal)
