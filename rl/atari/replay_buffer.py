"""
replay_buffer.py - Replay buffer for masked DQN continual learning.

Two-tier design:
  1. Standard DQN replay buffer for the current game (large, circular).
  2. Mask-correlated replay buffer that persists across games — stores a small
     number of transitions per game, tagged with the mask used for that game.
     During training, selectively replays transitions whose mask partially
     overlaps the current mask (correlation between CORR_LOW and CORR_HIGH),
     exactly as in the parent project's ReplayBuffer.
"""

import random

import numpy as np
import torch


class ReplayBuffer:
    """
    Standard circular replay buffer for DQN transitions.

    Stores (state, action, reward, next_state, done) tuples as numpy arrays
    for memory efficiency (states stay as uint8).
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """Add a transition. state/next_state should be numpy uint8 arrays."""
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch. Returns tensors on the current default device."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        device = torch.empty(0).device
        states = torch.from_numpy(np.stack(states)).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.from_numpy(np.stack(next_states)).to(device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class MaskReplayBuffer:
    """
    Replay buffer storing transitions tagged with per-game masks.

    Used for selective cross-game replay: samples transitions whose stored
    mask has partial overlap with the current game's mask, targeting shared
    neurons at risk of interference (same logic as parent project).
    """

    def __init__(self, max_per_game):
        self.max_per_game = max_per_game
        self.transitions = []  # list of (state, action, reward, next_state, done)
        self.masks = []        # list of mask tensors, one per transition
        self.count = 0

    def add(self, state, action, reward, next_state, done, mask):
        """Store a transition with its associated mask."""
        if self.count < self.max_per_game:
            self.transitions.append((state, action, reward, next_state, done))
            self.masks.append(mask.detach().clone())
        self.count += 1

    def new_game(self):
        """Reset per-game counter so the next game gets its own quota."""
        self.count = 0

    def sample_correlated(self, current_mask, n_samples, corr_low=0.2, corr_high=0.8):
        """
        Sample transitions whose masks partially overlap with current_mask.

        Returns None if no qualifying transitions exist.
        Returns (states, actions, rewards, next_states, dones) tensors.
        """
        if len(self.masks) == 0:
            return None

        stored = torch.stack(self.masks)
        mask_sum = current_mask.sum()
        if mask_sum == 0:
            return None

        corrs = (stored * current_mask).sum(-1) / mask_sum
        keep = (corrs > corr_low) & (corrs <= corr_high)
        overlap_idx = torch.where(keep)[0]

        if len(overlap_idx) == 0:
            return None

        chosen = overlap_idx[torch.randint(0, len(overlap_idx), (n_samples,))]

        device = torch.empty(0).device
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in chosen:
            s, a, r, ns, d = self.transitions[idx.item()]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        states = torch.from_numpy(np.stack(states)).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.from_numpy(np.stack(next_states)).to(device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones, self.masks[chosen[0].item()]

    def __len__(self):
        return len(self.transitions)
