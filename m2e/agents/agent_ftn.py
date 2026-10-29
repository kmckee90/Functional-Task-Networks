"""RECODE M2E with Bayesian FTN readouts; reservoirs and DDPG stay unchanged.

Actor/critic mask separation follows robotics/train_cw.py. Persistent Bayesian
TD mask inference, isolated atom updates and overlapping-mask replay follow
Globo's m2e_ftn.py. No task labels, extra rewards, or learned reservoirs.
"""
from copy import deepcopy
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .agent_recode import RDDPG_Dual as RECODEAgent
from .ftn_support.bayes_reconfig import BayesMaskReconfig


class MaskPosterior(BayesMaskReconfig, nn.Module):
    def __init__(self, H, k, normalize, cfg):
        nn.Module.__init__(self)
        self.hidden_size, self.k, self.normalize = H, k, normalize
        self.reconfig_steps = cfg['ftn_reconfig_steps']
        self.reconfig_sample_size = cfg['ftn_support_size']
        self.reconfig_noise_scale = cfg['ftn_reconfig_noise_scale']
        self.loss = F.mse_loss
        self.register_buffer('m', torch.zeros(H))
        self._init_bayes(
            bayes_reconfig_steps=cfg['ftn_reconfig_steps'],
            bayes_reconfig_lr=cfg['ftn_reconfig_lr'],
            bayes_kl_weight=cfg['ftn_kl_weight'],
            bayes_temperature=cfg['ftn_temperature'],
            bayes_prior_decay=cfg['ftn_prior_decay'],
            bayes_stochastic_mask=cfg['ftn_stochastic_mask'],
            bayes_reconfig_take_best=cfg['ftn_reconfig_take_best'],
            bayes_changepoint_reconfig_steps=cfg['ftn_changepoint_steps'],
            bayes_changepoint_reconfig_lr=(cfg['ftn_reconfig_lr'] if cfg['ftn_changepoint_lr'] is None
                                          else cfg['ftn_changepoint_lr']),
            bayes_changepoint_reconfig_sample_size=cfg['ftn_changepoint_support_size'],
            bayes_changepoint_temperature=cfg['ftn_changepoint_temperature'],
            bayes_changepoint_prior_decay=cfg['ftn_changepoint_prior_decay'],
            bayes_changepoint_kl_weight=cfg['ftn_changepoint_kl_weight'],
            bayes_changepoint_stochastic_mask=cfg['ftn_changepoint_stochastic_mask'],
            sample_type=cfg['ftn_sample_type'],
            overlap_mode=cfg['ftn_overlap_mode'], overlap_lambda=cfg['ftn_overlap_lambda'],
            overlap_threshold=cfg['ftn_overlap_threshold'], overlap_temperature=cfg['ftn_overlap_temperature'])
        self.m.copy_(self._sample_mask(self.mask_logits, cfg['ftn_initial_temperature'],
                                      cfg['ftn_initial_stochastic_mask']))

    def _topk_mask(self, scores):
        return torch.zeros_like(scores).scatter_(0, scores.topk(self.k).indices, 1.)

    def _run(self, x, mask, detach):
        y = torch.einsum('bna,n->ba', x, mask)
        return y.tanh() if self.normalize else y


class FTNDecoder(nn.Module):
    """Independent deep neurons, masked linear readout; selected atoms only."""
    def __init__(self, inputs, outputs, budget, normalize, cfg):
        super().__init__()
        sizes = [inputs] + [cfg['ftn_width']] * cfg['ftn_depth'] + [1]
        per_atom = sum((a + 1) * b for a, b in zip(sizes, sizes[1:])) + outputs
        k = cfg['ftn_k'] or max(1, round(budget / per_atom))
        H = cfg['ftn_hidden_size'] or k * cfg['ftn_bank_multiplier']
        if not 1 <= k <= H:raise ValueError('FTN requires 1 <= k <= hidden_size')
        self.k, self.H, self.normalize = k, H, normalize
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for a, b in zip(sizes, sizes[1:]):
            w = nn.Parameter(torch.empty(H, a, b))
            nn.init.uniform_(w, -math.sqrt(6/a), math.sqrt(6/a))
            self.weights.append(w)
            self.biases.append(nn.Parameter(torch.empty(H, b).uniform_(-.1/math.sqrt(a), .1/math.sqrt(a))))
        self.readout = nn.Parameter(torch.empty(H, outputs).uniform_(-1/math.sqrt(k), 1/math.sqrt(k)))
        self.register_buffer('mask', torch.zeros(H))
        self.mask[:k] = 1
        self.spec = dict(k=k, neurons=H, active_parameters=k*per_atom,
                         bank_parameters=H*per_atom, baseline_parameters=budget)

    def contributions(self, x, ids=None):
        if ids is None:ids = torch.arange(self.H, device=x.device)
        h = torch.einsum('bd,ndw->bnw', x, self.weights[0][ids]) + self.biases[0][ids]
        h = F.relu(h)
        for i in range(1, len(self.weights)):
            h = torch.einsum('bni,nij->bnj', h, self.weights[i][ids]) + self.biases[i][ids]
            if i < len(self.weights)-1:h = F.relu(h)
        return h.tanh() * self.readout[ids][None]

    def forward(self, x, mask=None):
        mask = self.mask if mask is None else mask
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        y = (self.contributions(x, ids) * mask[ids][None, :, None]).sum(1)
        return y.tanh() if self.normalize else y


class IsolatedRMSprop(torch.optim.Optimizer):
    """Inactive neurons and their RMSprop state stay bitwise unchanged."""
    def __init__(self, params, lr):super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, active):
        ids = active.nonzero().flatten()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue
                state = self.state[p]
                if not state:state['square_avg'] = torch.zeros_like(p)
                g = p.grad[ids]
                avg = .99*state['square_avg'][ids] + .01*g.square()
                state['square_avg'].index_copy_(0, ids, avg)
                p.index_add_(0, ids, -group['lr']*g/(avg.sqrt()+1e-8))


FTN_DEFAULTS = dict(
    ftn_k=None, ftn_hidden_size=None, ftn_bank_multiplier=8,
    ftn_width=8, ftn_depth=8, ftn_reconfig_steps=2, ftn_reconfig_lr=.3,
    ftn_support_size=64, ftn_kl_weight=1., ftn_temperature=1.,
    ftn_reconfig_take_best=False, ftn_sample_type='gumbel', ftn_reconfig_noise_scale=1.,
    ftn_initial_temperature=1., ftn_initial_stochastic_mask=True,
    ftn_prior_decay=1., ftn_stochastic_mask=True, ftn_changepoint_steps=2,
    ftn_changepoint_lr=None, ftn_changepoint_support_size=None,
    ftn_changepoint_temperature=1., ftn_changepoint_prior_decay=0.,
    ftn_changepoint_kl_weight=0., ftn_changepoint_stochastic_mask=True,
    ftn_changepoint_sd=3., ftn_changepoint_warmup=32, ftn_loss_ema=.03,
    ftn_action_reconfig_interval=8, ftn_replay_capacity=256,
    ftn_replay_every=8, ftn_overlap_low=.2, ftn_overlap_high=.8,
    ftn_overlap_mode='bump', ftn_overlap_lambda=0.,
    ftn_overlap_threshold=.5, ftn_overlap_temperature=.2,
    ftn_grad_clip=1., ftn_reactive=False, ftn_reactive_sd=5., ftn_reactive_ema=.9)


class RDDPG_Dual(RECODEAgent):
    def __init__(self, observation_size, action_size, action_bounds, reservoir_type='Local', **kwargs):
        cfg = dict(FTN_DEFAULTS, **kwargs)
        for key in ('ftn_bank_multiplier','ftn_width','ftn_depth','ftn_reconfig_steps',
                    'ftn_support_size','ftn_changepoint_steps','ftn_replay_every'):
            if not isinstance(cfg[key], int) or cfg[key] < 1:raise ValueError(f'{key} must be positive')
        for key in ('ftn_k','ftn_hidden_size','ftn_changepoint_support_size'):
            if cfg[key] is not None and (not isinstance(cfg[key], int) or cfg[key] < 1):
                raise ValueError(f'{key} must be positive or None')
        for key in ('ftn_initial_temperature', 'ftn_changepoint_lr'):
            value = cfg[key]
            if key == 'ftn_changepoint_lr' and value is None:continue
            if not math.isfinite(value) or value <= 0:raise ValueError(f'{key} must be finite and positive')
        if not math.isfinite(cfg['ftn_reconfig_noise_scale']) or cfg['ftn_reconfig_noise_scale'] < 0:
            raise ValueError('ftn_reconfig_noise_scale must be finite and non-negative')
        if not 0 <= cfg['ftn_overlap_low'] <= cfg['ftn_overlap_high'] <= 1:
            raise ValueError('Invalid FTN overlap interval')
        if not 0 < cfg['ftn_loss_ema'] <= 1:raise ValueError('Invalid FTN loss EMA')
        for key in ('ftn_replay_capacity','ftn_action_reconfig_interval','ftn_changepoint_warmup'):
            if not isinstance(cfg[key], int) or cfg[key] < 0:raise ValueError(f'Invalid {key}')
        if not math.isfinite(cfg['ftn_changepoint_sd']) or cfg['ftn_changepoint_sd'] < 0:
            raise ValueError('Invalid FTN changepoint SD')
        if not math.isfinite(cfg['ftn_grad_clip']) or cfg['ftn_grad_clip'] <= 0:
            raise ValueError('Invalid FTN gradient clip')
        if not 0 <= cfg['ftn_reactive_ema'] < 1:raise ValueError('Invalid FTN reactive EMA')
        if not math.isfinite(cfg['ftn_reactive_sd']) or cfg['ftn_reactive_sd'] < 0:
            raise ValueError('Invalid FTN reactive SD')
        super().__init__(observation_size, action_size, action_bounds, reservoir_type, **cfg)
        self.posteriors = nn.ModuleDict()
        self.ftn_spec = {}
        for key, inputs, outputs in [('o',self.rnn_size_o+action_size,1),
                                    ('f',self.rnn_size_f+action_size,1),
                                    ('a',self.rnn_size_o+self.rnn_size_f,action_size)]:
            pred, target = self._names(key)
            budget = sum(p.numel() for p in getattr(self,pred).parameters())
            decoder = FTNDecoder(inputs, outputs, budget, key=='a', cfg)
            posterior = MaskPosterior(decoder.H, decoder.k, key=='a', cfg)
            decoder.mask.copy_(posterior.m)
            setattr(self,pred,decoder)
            setattr(self,target,deepcopy(decoder).requires_grad_(False))
            setattr(self,'optimizer_'+key,IsolatedRMSprop(decoder.parameters(), cfg['lr']))
            self.posteriors[key] = posterior
            self.ftn_spec[key] = decoder.spec
        self.ftn_archive = []
        self.ftn_updates = self.ftn_archive_seen = self.ftn_searches = self.ftn_changepoints = 0
        self.ftn_replay_updates = 0
        self.ftn_loss_mean = self.ftn_loss_var = 0.
        self._last_action_search = -1
        self.ftn_reactive_mean = None
        self.ftn_reactive_var = 0.
        self.ftn_reactive_since = self.ftn_reactive_checks = 0

    @staticmethod
    def _names(key):
        return ('act_pred','act_target') if key=='a' else ('q_pred_'+key,'q_target_'+key)

    def _masks(self):return {k:p.m.detach().clone() for k,p in self.posteriors.items()}

    def _install(self, masks):
        for k,m in masks.items():
            for name in self._names(k):getattr(self,name).mask.copy_(m)

    @torch.no_grad()
    def _targets(self, d):
        next_action = self.act_target(torch.cat((d['no'],d['nf']),-1))
        to = self._q_targets_o(d['ro'],torch.cat((d['no'],next_action),-1),d['done'])
        if d['goal'] and self.config['a'] > 0:
            to = (1-self.config['a'])*to + self.config['a']*self._empirical_returns(d['ro'],d['done'])
        tf = self._q_targets_f(d['rf'],torch.cat((d['nf'],next_action),-1),d['done'])
        return to.reshape(-1,1), tf.reshape(-1,1)

    def _actor_loss(self, actions, states):
        zo,zf = states.split((self.rnn_size_o,self.rnn_size_f),-1)
        return -(self.q_target_o(torch.cat((zo,actions),-1)).mean()*self.config['OCP'] +
                 self.q_target_f(torch.cat((zf,actions),-1)).mean()*self.config['FCP'])

    def _search(self, d):
        # Critic masks minimize TD error; actor mask maximizes the original dual Q.
        # Fixed bootstrap labels are detached, as in ordinary DDPG regression.
        to,tf = self._targets(d)
        with torch.no_grad():
            inputs={'o':torch.cat((d['o'],d['action']),-1),'f':torch.cat((d['f'],d['action']),-1),
                    'a':torch.cat((d['o'],d['f']),-1)}
            err=self._critic_error(inputs,to,tf)
        cfg=self.config
        cp=(self.ftn_searches>=cfg['ftn_changepoint_warmup'] and
            err>self.ftn_loss_mean+cfg['ftn_changepoint_sd']*max(math.sqrt(self.ftn_loss_var),1e-4))
        reactive=cfg.get('ftn_reactive',False)
        if reactive:
            self.ftn_reactive_checks+=1
            warmup=math.ceil(1/max(1-cfg['ftn_reactive_ema'],1e-8))
            initial=self.ftn_reactive_mean is None
            cp=(not initial and self.ftn_reactive_since>=warmup and
                err>self.ftn_reactive_mean+cfg['ftn_reactive_sd']*math.sqrt(max(self.ftn_reactive_var,0.)))
            if not initial and not cp:
                self._observe_reactive_loss(err)
                return False
        with torch.no_grad():
            contributions={k:getattr(self,self._names(k)[0]).contributions(v) for k,v in inputs.items()}
        for k,y in [('o',to),('f',tf),('a',inputs['a'])]:
            if k!='a' and not cfg['OCP' if k=='o' else 'FCP']:continue
            posterior=self.posteriors[k]
            if k=='a':posterior.loss=self._actor_loss
            try:
                with torch.enable_grad():
                    method=posterior.reconfigure_changepoint if cp else posterior.reconfigure
                    method(contributions[k],y)
            finally:posterior.loss=F.mse_loss
            self._install({k:posterior.m})
        delta=err-self.ftn_loss_mean;rate=cfg['ftn_loss_ema']
        self.ftn_loss_mean+=rate*delta
        self.ftn_loss_var=(1-rate)*(self.ftn_loss_var+rate*delta*delta)
        self.ftn_searches+=1;self.ftn_changepoints+=int(cp)
        if reactive:
            # Match ftn/run_experiments: track the post-search loss, never the
            # pre-search spike; suppress triggers while the EMA re-forms.
            self.ftn_reactive_since=0
            self._observe_reactive_loss(self._critic_error(inputs,to,tf))
        return True

    @torch.no_grad()
    def _critic_error(self, inputs, to, tf):
        return sum(float(F.mse_loss(getattr(self,self._names(k)[0])(inputs[k]),y))
                   for k,y in [('o',to),('f',tf)] if self.config['OCP' if k=='o' else 'FCP'])

    def _observe_reactive_loss(self, loss):
        alpha=self.config['ftn_reactive_ema']
        if self.ftn_reactive_mean is None:self.ftn_reactive_mean=loss
        else:
            delta=loss-self.ftn_reactive_mean
            self.ftn_reactive_mean=alpha*self.ftn_reactive_mean+(1-alpha)*loss
            self.ftn_reactive_var=alpha*self.ftn_reactive_var+(1-alpha)*delta*delta
        self.ftn_reactive_since+=1

    def _backward(self, d):
        to,tf=self._targets(d)
        loss=0.
        for k,y in [('o',to),('f',tf)]:
            if not self.config['OCP' if k=='o' else 'FCP']:continue
            pred=getattr(self,self._names(k)[0])(torch.cat((d[k],d['action']),-1))
            term=F.mse_loss(pred,y);term.backward();loss+=float(term.detach())
        states=torch.cat((d['o'],d['f']),-1)
        actor=self._actor_loss(self.act_pred(states),states)
        actor.backward()
        return loss,float(actor.detach())

    def _train_support(self, d):
        self._search(self._trim(d))
        masks=self._masks();active={k:m.bool() for k,m in masks.items()}
        for k in masks:getattr(self,'optimizer_'+k).zero_grad(set_to_none=True)
        critic,actor=self._backward(d)
        # One old support, under its own masks, sums gradients on shared neurons.
        eligible=self._eligible_archive_indices(masks)
        if eligible:
            old=self.ftn_archive[eligible[int(torch.randint(len(eligible),()))]]
            old_masks={k:m.to(masks[k].device) for k,m in old['masks'].items()}
            self._install(old_masks)
            self._backward(self._to_device(old['data']))
            for k in masks:active[k]|=old_masks[k].bool()
            self.ftn_replay_updates+=1
        self._install(masks)
        for k in masks:
            if k!='a' and not self.config['OCP' if k=='o' else 'FCP']:continue
            pred,target=(getattr(self,n) for n in self._names(k))
            nn.utils.clip_grad_norm_(pred.parameters(),self.config['ftn_grad_clip'])
            getattr(self,'optimizer_'+k).step(active[k])
            self._polyak_update(target.parameters(),pred.parameters())
        self.ftn_updates+=1
        capacity=self.config['ftn_replay_capacity']
        if capacity and self.ftn_updates%self.config['ftn_replay_every']==0:
            self.ftn_archive_seen+=1
            entry=dict(data=self._trim(d,cpu=True),masks={k:m.cpu() for k,m in masks.items()})
            if len(self.ftn_archive)<capacity:self.ftn_archive.append(entry)
            else:
                slot=int(torch.randint(self.ftn_archive_seen,()))
                if slot<capacity:self.ftn_archive[slot]=entry
        self._metrics.update(ftn_critic_loss=critic,ftn_actor_loss=actor,ftn_updates=self.ftn_updates,
            ftn_searches=self.ftn_searches,ftn_changepoints=self.ftn_changepoints,
            ftn_replay_updates=self.ftn_replay_updates,ftn_archived_supports=len(self.ftn_archive))

    def _eligible_archive_indices(self, masks):
        """Same eligibility/order as the scalar scan, with three transfers total.

        Archive masks already live on CPU. Accumulate exact binary overlaps in
        batches there, then divide in float64 to match Python float thresholds.
        """
        if not self.ftn_archive:return []
        overlaps=[]
        for k,m in masks.items():
            stored=torch.stack([entry['masks'][k] for entry in self.ftn_archive])
            counts=(stored*m.detach().cpu()).sum(-1).double()
            overlaps.append(counts/self.posteriors[k].k)
        overlaps=torch.stack(overlaps,-1)
        eligible=((overlaps>=self.config['ftn_overlap_low']) &
                  (overlaps<=self.config['ftn_overlap_high'])).any(-1)
        return eligible.nonzero().flatten().tolist()

    def _trim(self,d,cpu=False):
        # Keep the full sequence when blending empirical returns; truncating it
        # would change those targets. Ordinary TD supports can be capped safely.
        n=len(d['o']) if d['goal'] and self.config['a']>0 else self.config['ftn_support_size']
        return {k:(v[:n].detach().clone().cpu() if cpu else v[:n].detach())
                if isinstance(v,torch.Tensor) else v for k,v in d.items()}

    def _to_device(self,d):
        device=self.act_pred.mask.device
        return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in d.items()}

    def forward(self,x,f):
        interval=self.config['ftn_action_reconfig_interval']
        if interval and self.cur_step%interval==0 and self.cur_step!=self._last_action_search and self.memory.length()>=self.config['minibatch_size']:
            models=(self.state_model_o,self.state_model_f,self.state_model_o_targ,self.state_model_f_targ)
            hidden=[m.hidden.clone() for m in models]
            try:
                idx=torch.arange(max(0,self.memory.length()-self.config['ftn_support_size']),self.memory.length(),device=x.device)
                obs,fb,no,nf,act,rew,done,_=self.memory.recall(idx)
                with torch.no_grad():
                    d=dict(o=models[0](obs*self.config['OCP']),f=models[1](fb*self.config['FCP']),
                           no=models[2](no),nf=models[3](nf),action=act,ro=rew,rf=rew,done=done,goal=False)
                self._search(d);self._last_action_search=self.cur_step
            finally:
                for model,state in zip(models,hidden):model.hidden=state
        return super().forward(x,f)

    def get_extra_state(self):
        return {k:getattr(self,k) for k in ('ftn_archive','ftn_updates','ftn_archive_seen',
            'ftn_searches','ftn_changepoints','ftn_replay_updates','ftn_loss_mean','ftn_loss_var','_last_action_search',
            'ftn_reactive_mean','ftn_reactive_var','ftn_reactive_since','ftn_reactive_checks')}

    def set_extra_state(self,state):
        for k,v in state.items():setattr(self,k,v)

    def reset_memory(self,**kwargs):
        super().reset_memory(**kwargs)
        self.ftn_archive=[];self.ftn_archive_seen=0;self._last_action_search=-1
    def replay(self):
        if self.memory.length() < self.config["minibatch_size"]:
            return

        # Save the real-time context to reinstate later.
        hidden_f = self.state_model_f.hidden.clone()
        hidden_o = self.state_model_o.hidden.clone()

        # Calculate the reward function for main memories:
        main_reward = self.get_reward(self.memory.next_obs[:])# + self.memory.reward[:].squeeze()  # type: ignore
        main_best = main_reward.argmax()
        main_reward[main_best] *= self.config["main_sparse_reward_scale"]
        best_main_reward = main_reward[main_best]

        # Calculate the reward function for goal memories:
        if self.goal_memory.length() > 3 * self.config["minibatch_size"]:
            goal_reward = self.get_reward(self.goal_memory.next_obs[:])#  + self.goal_memory.reward[:].squeeze() # type: ignore
            goal_best = goal_reward.argmax()
            goal_reward[goal_best] *= self.config["goal_sparse_reward_scale"]
            best_goal_reward = goal_reward[goal_best]

            # Keep the goal location for rendering
            self.goal_location = self.goal_memory.next_obs[goal_best]
        else:
            best_goal_reward = 0

        self.best_main_reward = best_main_reward.item()
        self.best_goal_reward = best_goal_reward.item() if isinstance(best_goal_reward, Tensor) else best_goal_reward

        # If a better reward was computed from main, move that episode to goals
        if best_main_reward > best_goal_reward:
            obs, fb, next_obs, next_fb, act, rew, done, _ = self._prioritized_recall(self.memory, "goal")
            for i in range(len(obs)):
                self.goal_memory.remember(obs[i], fb[i], next_obs[i], next_fb[i], act[i], rew[i], done[i])

            # Have to redo this because goal memory changed
            goal_reward = self.get_reward(self.goal_memory.next_obs[:])# + self.goal_memory.reward[:].squeeze()  # type: ignore
            goal_best = goal_reward.argmax()
            goal_reward[goal_best] *= self.config["goal_sparse_reward_scale"]
            best_goal_reward = goal_reward[goal_best]

            # Keep the goal location for rendering
            self.goal_location = self.goal_memory.next_obs[goal_best]

        # Prioritized replay
        if self.config["prioritized_replay"]:
            replay_types = ["goal", "last", "random"]
        else:
            replay_types = ["random"]
        for _ in range(self.config["n_optimizer_epochs"]):            
            for step in range(self.config["n_optimizer_steps"]):
                # Reinstate real-world context
                # self.state_model_f.hidden = hidden_f
                # self.state_model_o.hidden = hidden_o

                if step < len(replay_types):
                    recall_mode = replay_types[step]
                else:
                    recall_mode = "random"

                # Train on either the goal buffer or the main buffer
                goal_replay = self.goal_memory.length() > 4 * self.config["minibatch_size"] and recall_mode == "goal"
                if goal_replay:
                    obs, feedback, next_obs, next_feedback, action, reward, done, idx = self._prioritized_recall(
                        self.goal_memory, "goal"
                    )
                    rewards_o = goal_reward[idx].reshape(-1, 1)  # type: ignore
                else:
                    obs, feedback, next_obs, next_feedback, action, reward, done, idx = self._prioritized_recall(
                        self.memory, recall_mode
                    )
                    rewards_o = reward

                states_o = self.state_model_o(obs * self.config["OCP"])
                next_states_o = self.state_model_o_targ(next_obs)

                states_f = self.state_model_f(feedback *  self.config["FCP"])
                next_states_f = self.state_model_f_targ(next_feedback)

                self._train_support(dict(o=states_o.detach(),f=states_f.detach(),
                    no=next_states_o.detach(),nf=next_states_f.detach(),action=action,
                    ro=rewards_o,rf=reward,done=done,goal=goal_replay))

        self.state_model_f.hidden = hidden_f
        self.state_model_o.hidden = hidden_o

