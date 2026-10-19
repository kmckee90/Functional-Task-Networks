"""Use the existing Globo RECODE DQN as the model-free behavior policy.

The same real transition trains this policy and Dreamer. No imagined transition
or predicted reward is needed for the DQN's density objective.
"""
from pathlib import Path
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from .legacy.agent import GloboExplorerDQN


class ModelFreeBehavior:
    def __init__(self, device='cuda:0', seed=7, dreamer_probability=.1,
                 replay_capacity=20000, goal_capacity=20000, recode_capacity=2000,
                 checkpoint_root=None, epsilon_end=.025, gamma=.9,
                 online_coef=1., offline_coef=1., frontier_scale=5., lr=3e-4,
                 model_reward_weight=0., model_reward_half_life=10000., model_feedback=False,
                 reservoir_channels=64, reservoir_variant='legacy',
                 num_actions=25, in_h=32, in_w=64, reservoir_stride=7,
                 telemetry_dims=0):
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        self.agent = GloboExplorerDQN(num_actions=num_actions, in_h=in_h, in_w=in_w,
            reservoir_channels=reservoir_channels, reservoir_stride=reservoir_stride,
            reservoir_kernel=reservoir_stride,
            reservoir_variant=reservoir_variant,
            recode_down_h=16, recode_down_w=32, recode_memory_size=recode_capacity,
            use_online_novelty=True, use_offline_novelty=True,
            env_reward_coef=0., death_penalty=0., gamma=gamma,
            online_coef=online_coef, offline_coef=offline_coef,
            main_sparse_reward_scale=frontier_scale, goal_sparse_reward_scale=frontier_scale,
            lr=lr,
            memory_length=replay_capacity, goal_memory_length=goal_capacity, minibatch_size=64,
            epsilon_end=epsilon_end, epsilon_decay_steps=10000, grad_clip=1.,
            target_update_tau=.01, device=device, seed=seed, model_feedback=model_feedback,
            telemetry_dims=telemetry_dims)
        from .novelty_scale import NoveltyScale
        storage_dtype=os.environ.get('GLOBO_MF_REPLAY_DTYPE','float32')
        if storage_dtype not in ('float32','float16'):raise ValueError('Invalid replay storage dtype')
        for buffer in (self.agent.memory,self.agent.goal):
            buffer.storage_dtype=getattr(torch,storage_dtype)
        self.novelty_scale = NoveltyScale()
        self.model_feedback = model_feedback
        self.probability = dreamer_probability
        self.agent.model_reward_weight=model_reward_weight
        self.agent.model_reward_half_life=model_reward_half_life
        self.rng = np.random.default_rng(seed)
        self.pending = None
        self.updates = 0
        self.metrics = {}
        from .scalable_replay import install_scalable_sampling, ChunkStore
        install_scalable_sampling(self.agent)
        self.store = ChunkStore(checkpoint_root, dict(memory=self.agent.memory,
            goal=self.agent.goal)) if checkpoint_root else None

    def observe_frame(self, image):
        image = torch.from_numpy(np.array(image, copy=True)).to(self.agent.device)
        image = image.permute(2,0,1).float() / 255
        self.image = F.interpolate(image[None], size=(self.agent.in_h,self.agent.in_w), mode='area')[0]
        novelty = self.agent.intrinsic_reward(self.image)
        self.agent.update_density(self.image)
        return float(novelty)

    def policy(self, obs, dreamer_action, model_novelty=None):
        assert len(obs['image']) == 1, 'Exactly one real environment stream'
        first, last = bool(obs['is_first'][0]), bool(obs['is_last'][0])
        nov = float(obs['density'][0,0])
        if first:
            self.agent.reset_hidden()
            if hasattr(self,'ftn') and hasattr(self.ftn,'on_reset'):self.ftn.on_reset()
            self.pending = None
        if self.pending is not None:
            self.agent.note_action(self.pending[1], nov)
            self.agent._model_feedback = self.pending[3]
        z = self.agent.encode(self.image, torch.tensor(nov, device=self.agent.device))
        if self.pending is not None:
            old_z, action, bonus, feedback = self.pending
            self.agent.remember(old_z, action, float(obs['reward'][0]), nov,
                self.agent.embed(self.image), z, bool(obs['is_terminal'][0]), last,
                model_nov=bonus,model_step=self.agent.cur_step)
            self.agent.step_schedule()
        if hasattr(self,'ftn'):self.ftn.current_image=self.image
        selected = self.agent.act(z)[0]
        from_model = self.rng.random() < self.probability
        if from_model:
            selected = int(dreamer_action['action'][0])
        if (self.agent.model_reward_weight or self.model_feedback) and model_novelty is None:
            raise ValueError('Missing action-conditioned model novelty')
        raw=float(np.asarray(model_novelty)[0,selected]) if model_novelty is not None else 0.
        feedback, bonus = self.novelty_scale.transform(raw)
        self.pending = None if last else (z, selected, bonus, feedback)
        loss = None
        if self.agent.memory.size >= 500 and self.agent.cur_step % 50 == 0:
            loss = self.agent.train_round(50)
            self.updates += 50
            if loss is not None and not np.isfinite(loss):
                raise FloatingPointError('Non-finite model-free density loss')
        self.metrics = dict(epsilon=self.agent.epsilon, updates=self.updates,
            novelty=nov, model_action=float(from_model), model_novelty=raw,
            model_input=feedback, model_reward=bonus,
            model_reward_scale=self.novelty_scale.reward_scale,
            model_input_scale=max(.001,self.novelty_scale.second**.5),
            weighted_model_novelty=self.agent.model_reward_weight*bonus, **self.agent._last_metrics)
        if loss is not None:
            self.metrics['loss'] = loss
        if hasattr(self,'ftn'):
            self.metrics.update({f'ftn_{k}':v for k,v in self.agent.frontier_rotation.metrics(self.agent.cur_step).items()})
            self.metrics.update(ftn_searches=self.ftn.searches,ftn_updates=self.ftn.updates,
                ftn_replay_updates=self.ftn.replay_updates,ftn_changepoints=self.ftn.changepoints,
                ftn_mask_turnover=self.ftn.mask_turnover,ftn_td_ema=self.ftn.loss_mean,
                ftn_atoms_used=int(self.ftn.visited_atoms.sum()))
        return dict(action=np.array([selected], np.int32))

    def save(self):
        from .scalable_replay import cpu_tree
        # Save real replay too, unlike the standalone explorer's demo checkpoint.
        buffers = {}
        for name in ('memory','goal'):
            if self.store:
                break
            buf = getattr(self.agent,name)
            buffers[name] = dict(size=buf.size, ptr=buf.ptr,
                tensors={k:v[:buf.size].cpu() for k,v in vars(buf).items()
                         if isinstance(v,torch.Tensor)})
        return dict(agent=cpu_tree(self.agent.state_dict()), buffers=buffers,
            ftn=self.ftn.save() if hasattr(self,'ftn') else None,
            novelty_scale=self.novelty_scale.save(), model_feedback=self.model_feedback,
            chunks=self.store.save() if self.store else None,
            rng=self.rng.bit_generator.state, updates=self.updates,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())

    def load_weights(self, path):
        """Transfer parameters only; all memories, optimizer and schedules stay fresh."""
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['reservoir_config'] != self.agent.reservoir_config:
            raise ValueError('Warm-start reservoir architecture mismatch')
        for key in ('q_net','q_target','q_env','q_env_target','reservoir','feedback_to_channel'):
            getattr(self.agent,key).load_state_dict(state[key],strict=True)
        self.agent.reset_hidden()
        assert self.agent.memory.size == self.agent.goal.size == self.agent.density.size == 0
        assert self.updates == self.agent.cur_step == 0 and not self.agent.optimizer.state

    def load(self, state):
        if state.get('model_feedback',False) != self.model_feedback:
            raise ValueError('Feedback architecture changed; use a fresh run')
        if 'novelty_scale' in state:self.novelty_scale.load(state['novelty_scale'])
        self.agent.load_state_dict(state['agent'])
        if hasattr(self,'ftn'):self.ftn.load(state['ftn'])
        if state.get('chunks'):
            self.store.load(state['chunks'])
        for name,saved in state['buffers'].items():
            buf=getattr(self.agent,name)
            if saved['size']:
                buf._lazy(saved['tensors']['z'][0])
                for key,value in saved['tensors'].items():
                    getattr(buf,key)[:saved['size']].copy_(value.to(buf.device))
            buf.size,buf.ptr=saved['size'],saved['ptr']
            if self.store:
                buf.dirty_chunks.update(range((buf.size + self.store.chunk_size - 1) // self.store.chunk_size))
        self.rng.bit_generator.state=state['rng']
        self.updates=state['updates']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        self.pending=None
