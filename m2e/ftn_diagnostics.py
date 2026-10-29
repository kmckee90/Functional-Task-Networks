"""Retrospective, isolated fixed-mask route diagnostics for the maze FTN.

No live agent method is called during evaluation. Only copied actor/reservoir
modules and a rollout-local copy of the density estimator are used. All RNG
streams are restored, including on failures, before training resumes.
"""

from collections import Counter
from contextlib import contextmanager
from copy import copy, deepcopy
import json
from math import ceil, isqrt
from pathlib import Path
import random

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import numpy as np
import torch

from agents.ftn_support.spatial_permute import spatial_permutation


BANKS = ("a", "o", "f")
LABELS = {"a": "Actor", "o": "OCP", "f": "FCP"}
COLORS = {"a": "#28b9d6", "o": "#c889e8", "f": "#efaa56"}
SPATIAL_OPTIONS = dict(initialization="random_blobs", center=.25,
                       roundness=2., radius=7, temperature=.05, seed=0)


@contextmanager
def isolated_rng():
    """Keep diagnostics from advancing the Python, NumPy or PyTorch RNGs."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def clone_maze(env):
    """Keep this episode's geometry, without sharing mutable episode state.

    Avoid copying pygame surfaces or opening/closing a pygame display: either
    would interfere with the live training renderer. Resets of randomized
    mazes keep the recorded geometry so comparisons isolate the mask.
    """
    result = copy(env)
    result.walls = env.walls.copy()
    result.randomized = False
    result._screen = None
    result._all_pos, result._last_pos = [], []
    result._pos = None if env.pos is None else env.pos.copy()
    result._goal = None
    result._np_random = deepcopy(getattr(env, "_np_random", None))
    return result


class FTNRouteDiagnostic:
    def __init__(self, output_dir, obs_scale, normalizer, *, block_size=20,
                 rollouts=32, seed=0, spatial_sweeps=8, artifact_logger=None):
        if not 1 <= block_size <= 20 or rollouts < 1:
            raise ValueError("Diagnostics require 1..20 masks and positive rollouts")
        self.output_dir = Path(output_dir)
        self.obs_scale, self.normalizer = obs_scale, normalizer
        self.block_size, self.rollouts = block_size, rollouts
        self.seed, self.spatial_sweeps = seed, spatial_sweeps
        self.artifact_logger = artifact_logger
        self.masks = {k: [] for k in BANKS}
        self._mask_lookup = {k: {} for k in BANKS}
        self.layouts, self.grid_shapes, self._layout_counts = {}, {}, {}
        self.pending = []
        self.current = None
        self._last_search = None
        self._current_ids = None

    def begin_episode(self, episode, env):
        if self.current is not None:
            raise RuntimeError("Previous diagnostic episode was not finished")
        self.current = dict(episode=episode, env=clone_maze(env),
                            start_pos=env.pos.copy(), segments=[], steps=0)

    def record_step(self, agent):
        """Call after forward/mask search, before stepping the training maze."""
        if self.current is None:
            raise RuntimeError("begin_episode must precede record_step")
        # Masks only change during a search (or a new agent/load). Avoid three
        # GPU-to-CPU transfers at every action when the masks are unchanged.
        version = (id(agent), agent.ftn_searches)
        if self._last_search != version:
            ids = {}
            for k in BANKS:
                name = agent._names(k)[0]
                mask = getattr(agent, name).mask.detach().cpu().numpy().astype(np.uint8)
                signature = mask.tobytes()
                if signature not in self._mask_lookup[k]:
                    self._mask_lookup[k][signature] = len(self.masks[k])
                    self.masks[k].append(mask.copy())
                ids[k] = self._mask_lookup[k][signature]
            self._current_ids, self._last_search = ids, version
        segments = self.current["segments"]
        if not segments or segments[-1]["mask_ids"] != self._current_ids:
            segments.append(dict(start_step=self.current["steps"], steps=0,
                                 mask_ids=dict(self._current_ids)))
        segments[-1]["steps"] += 1
        self.current["steps"] += 1

    def end_episode(self):
        episode = self.current
        if episode is None or not episode["segments"]:
            raise RuntimeError("Cannot finish an episode without actions")
        # Choose the most-used ACTOR mask, then the most-used complete mask
        # tuple associated with it. This avoids critic-only switches splitting
        # the count for an otherwise identical behavior policy. Ties use first.
        actor_counts = Counter()
        tuple_counts = Counter()
        for segment in episode["segments"]:
            actor_counts[segment["mask_ids"]["a"]] += segment["steps"]
            tuple_counts[tuple(segment["mask_ids"][k] for k in BANKS)] += segment["steps"]
        actor_id = actor_counts.most_common(1)[0][0]
        candidates = {ids: n for ids, n in tuple_counts.items() if ids[0] == actor_id}
        episode["mask_ids"] = dict(zip(BANKS, max(candidates, key=candidates.get)))
        episode["representative_steps"] = actor_counts[actor_id]
        self.pending.append(episode)
        self.current = None

    @property
    def ready(self):
        return len(self.pending) == self.block_size

    def consolidate(self):
        """Refine one shared display permutation per independent neuron bank."""
        for k in BANKS:
            if self._layout_counts.get(k) == len(self.masks[k]):
                continue
            masks = np.stack(self.masks[k])
            size = masks.shape[1]
            height = isqrt(size)
            shape = (height, ceil(size / height))
            padded = np.pad(masks, ((0, 0), (0, shape[0] * shape[1] - size)))
            self.layouts[k] = spatial_permutation(
                padded, shape, previous=self.layouts.get(k),
                sweeps=self.spatial_sweeps, **SPATIAL_OPTIONS)
            self.grid_shapes[k] = shape
            self._layout_counts[k] = len(self.masks[k])

    @torch.no_grad()
    def _rollout(self, policy, episode, masks, cfg, max_reward, rollout):
        actor, observation_model, feedback_model, base_density = policy
        # Common noise seeds across masks make comparisons reproducible. Each
        # rollout starts from the same block-end density and empty RNN state.
        rollout_seed = (self.seed + rollout) % (2**32)
        random.seed(rollout_seed)
        np.random.seed(rollout_seed)
        torch.manual_seed(rollout_seed)
        density = deepcopy(base_density)
        env = clone_maze(episode["env"])
        obs, _ = env.reset(seed=rollout_seed)
        env.pos = episode["start_pos"].copy()
        obs[:2] = env.pos
        device = actor.mask.device
        actor.mask.copy_(torch.as_tensor(masks["a"], device=device))
        observation_model.hidden.zero_()
        feedback_model.hidden.zero_()
        feedback = torch.zeros(feedback_model.rnn.input_size, device=device)
        zeros = torch.zeros(cfg["extra_zeros"], device=device)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device) / self.obs_scale
        positions = [env.pos.copy()]
        done = truncated = False
        while not (done or truncated):
            zo = observation_model(obs.unsqueeze(0) * cfg["OCP"])
            zf = feedback_model(feedback.unsqueeze(0) * cfg["FCP"])
            action = actor(torch.cat((zo, zf), -1)).reshape(-1)
            next_obs, _, done, truncated, _ = env.step(action.cpu().numpy().copy())
            positions.append(env.pos.copy())
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device) / self.obs_scale
            reward = density.get_reward(obs.unsqueeze(0)).reshape(-1)
            density.update(obs)
            density.step()
            norm_reward = self.normalizer(reward)
            max_reward = max(max_reward, float(norm_reward.item()))
            bins = cfg["num_negdensity_bins"]
            cutpoints = torch.arange(1, bins + 1, device=device) / bins
            bin_index = (norm_reward / max_reward <= cutpoints).int().argmax()
            binned = torch.nn.functional.one_hot(bin_index, num_classes=bins)
            feedback = torch.cat((action, reward, binned, zeros))
        return np.asarray(positions, dtype=np.float32)

    def evaluate(self, agent, max_reward):
        """Evaluate and save a completed block, leaving the live agent untouched."""
        if not self.ready:
            raise RuntimeError("A complete diagnostic block is required")
        policy = (deepcopy(agent.act_pred).eval(),
                  deepcopy(agent.state_model_o).eval(),
                  deepcopy(agent.state_model_f).eval(),
                  deepcopy(agent.density_estimator))
        return self.evaluate_policy(policy, agent.config, max_reward, agent.cur_step)

    def evaluate_policy(self, policy, cfg, max_reward, training_step):
        """Evaluate an owned inference snapshot, synchronously or in a worker."""
        if not self.ready:
            raise RuntimeError("A complete diagnostic block is required")
        with isolated_rng():
            for model in policy[1:3]:
                model.rnn.flatten_parameters()
            self.consolidate()
            visitations = []
            for episode in self.pending:
                masks = {k: self.masks[k][episode["mask_ids"][k]] for k in BANKS}
                visitations.append([
                    self._rollout(policy, episode, masks, cfg,
                                  max(max_reward, 1e-6), r)
                    for r in range(self.rollouts)])
            paths = self._save(visitations, training_step)
            if self.artifact_logger:
                for path in paths:
                    if path.suffix.lower() == ".png":
                        self.artifact_logger(path)
        self.pending.clear()
        return paths

    def _save(self, visitations, training_step):
        first, last = self.pending[0]["episode"], self.pending[-1]["episode"]
        stem = f"episodes_{first:04d}_{last:04d}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.output_dir / f"{stem}.png"
        data_path = self.output_dir / f"{stem}.npz"
        meta_path = self.output_dir / f"{stem}.json"
        arrays = {}
        for k in BANKS:
            arrays[f"masks_{k}"] = np.stack(self.masks[k])
            arrays[f"permutation_{k}"] = self.layouts[k]
            arrays[f"grid_shape_{k}"] = np.array(self.grid_shapes[k])
        episodes = []
        for episode, routes in zip(self.pending, visitations):
            number = episode["episode"]
            arrays[f"walls_{number}"] = episode["env"].walls
            arrays[f"start_pos_{number}"] = episode["start_pos"]
            for r, route in enumerate(routes):
                arrays[f"visits_{number}_{r}"] = route
            episodes.append({k: episode[k] for k in (
                "episode", "steps", "mask_ids", "representative_steps", "segments")})
        np.savez_compressed(data_path, **arrays)
        metadata = dict(episodes=episodes, training_step=training_step,
                        rollouts_per_mask=self.rollouts, visitation_color="red", seed=self.seed,
                        representative="most-used actor mask; first wins ties",
                        policy="block-end weights; fixed mask; epsilon=0; no mask search",
                        recurrent_state="zero at each diagnostic episode start",
                        density="independent block-end copy per rollout; local updates only",
                        feedback="zero at start; local density rewards and normalization thereafter",
                        maze="recorded episode walls and start position; no regeneration",
                        layout="SpatialPermuteRandomCenter", spatial_options=SPATIAL_OPTIONS,
                        spatial_sweeps=self.spatial_sweeps,
                        cumulative_mask_counts={k: len(v) for k, v in self.masks.items()})
        metadata["execution"] = getattr(self, "execution", {"mode": "synchronous"})
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        self._plot(visitations, image_path, first, last)
        return [image_path, data_path, meta_path]

    def _plot(self, visitations, path, first, last):
        figure = Figure(figsize=(25, 16), facecolor="#f1f4f8")
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(4, 5, left=.02, right=.99, bottom=.035,
                                  top=.90, hspace=.36, wspace=.18)
        try:
            for i, (episode, routes) in enumerate(zip(self.pending, visitations)):
                cell = grid[i // 5, i % 5].subgridspec(
                    3, 2, width_ratios=(2.7, 1), hspace=.48, wspace=.08)
                maze_ax = figure.add_subplot(cell[:, 0])
                maze_ax.set_facecolor("#101822")
                maze_ax.add_collection(LineCollection(episode["env"].walls,
                                                      colors="#dce6ef", linewidths=.65))
                for route in routes:
                    maze_ax.scatter(route[:, 0], route[:, 1], s=2.5,
                                    color="red", alpha=.55, linewidths=0)
                maze_ax.scatter(*episode["start_pos"], s=25, marker="+", color="white", linewidths=1)
                maze_ax.set(xlim=(-13, 13), ylim=(-13, 13), aspect="equal", xticks=[], yticks=[])
                count, steps = episode["representative_steps"], episode["steps"]
                maze_ax.set_title(f"Episode {episode['episode']} · {self.rollouts} rollouts\n"
                                  f"dominant actor: {count}/{steps} steps", fontsize=9)
                for row, k in enumerate(BANKS):
                    axis = figure.add_subplot(cell[row, 1])
                    mask_id = episode["mask_ids"][k]
                    mask = self.masks[k][mask_id]
                    n = len(self.layouts[k])
                    padded = np.pad(mask, (0, n - len(mask)))
                    # Padded display sites remain visibly distinct from neurons.
                    values = np.where(self.layouts[k] < len(mask),
                                      padded[self.layouts[k]], np.nan)
                    axis.imshow(values.reshape(self.grid_shapes[k]), vmin=0, vmax=1,
                                interpolation="nearest", cmap=ListedColormap(["#18232f", COLORS[k]]))
                    axis.set_title(f"{LABELS[k]} #{mask_id}", fontsize=7, pad=2)
                    axis.set_axis_off()
            figure.suptitle(f"FTN fixed-mask routes · training episodes {first}–{last}",
                            fontsize=19, fontweight="bold", y=.98)
            figure.text(.5, .947,
                        f"Most-used actor mask per episode · {self.rollouts} independent rollouts · "
                        "block-end weights · shared cumulative spatial layouts",
                        ha="center", fontsize=10)
            figure.savefig(path, dpi=120, facecolor=figure.get_facecolor())
        finally:
            figure.clear()
