"""Minimal 2D top-down survival grid world (gymnasium).

A bounded grid containing:
  * static random obstacle cells,
  * a few NPCs that wander randomly and damage the agent on contact,
  * a few nutrients that occasionally respawn and sate the agent's hunger.

The agent must simply survive. It has two internal scalars:
  * ``health``  in (0, 1]  -- starts at 1; the agent dies when it reaches 0.
  * ``hunger``  in [0, 1]  -- rises linearly each step; at 1 it bleeds health.

Dynamics per step:
  1. The agent moves one cell (up/down/left/right) unless blocked by an
     obstacle or the world border, in which case it stays put.
  2. Hunger increases by ``1 / sate_steps``.
  3. If the agent stands on a nutrient it eats it: hunger drops by ``eat_value``
     and the nutrient is removed.
  4. While hunger == 1 the agent starves: health drops by ``1 / starve_steps``.
  5. NPCs each take a random step. Any NPC sharing the agent's cell deals
     ``npc_damage`` to health (handles both "agent walks into NPC" and "NPC
     walks into agent").
  6. Nutrients respawn stochastically up to ``n_nutrients``.
  7. The agent dies (terminated) when health <= 0, yielding the only extrinsic
     reward: ``death_reward`` (default -1). Every surviving step rewards 0.

Observation (all in [0, 1], float32, flattened):
  * a ``(2*view_radius+1)^2 * 3`` egocentric one-hot view kernel centred on the
    agent, channels = (obstacle | wall, npc, nutrient). The world border reads
    as an obstacle. The centre cell is the agent and is never encoded.
  * health  (in (0, 1])
  * hunger  (the "nutrient counter", in [0, 1])
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Cell channel indices in the egocentric view.
CH_OBSTACLE = 0
CH_NPC = 1
CH_NUTRIENT = 2
N_CHANNELS = 3

# Action -> (drow, dcol). Matches WASD in the human-play wrapper.
ACTIONS = {
    0: (-1, 0),  # up    / W
    1: (1, 0),   # down  / S
    2: (0, -1),  # left  / A
    3: (0, 1),   # right / D
}


class SurvivalGridEnv(gym.Env):
    """A bounded top-down survival grid world with an egocentric observation."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 8}

    def __init__(
        self,
        size: int = 20,
        view_radius: int = 3,
        n_npcs: int = 0,
        n_nutrients: int = 8,
        obstacle_density: float = 0.2,
        sate_steps: int = 100,      # steps for hunger to climb 0 -> 1
        starve_steps: int = 100,    # steps to die of starvation once hunger == 1
        eat_value: float = 0.5,     # hunger removed by eating one nutrient
        npc_damage: float = 0.2,   # health lost per NPC contact
        nutrient_respawn_prob: float = 0.05,
        max_steps: int = 5000,
        death_reward: float = -1.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        assert size > 2 * view_radius, "world must be larger than the view kernel"
        self.size = size
        self.view_radius = view_radius
        self.view_dim = 2 * view_radius + 1
        self.n_npcs = n_npcs
        self.n_nutrients = n_nutrients
        self.obstacle_density = obstacle_density
        self.hunger_rate = 1.0 / sate_steps
        self.starve_rate = 1.0 / starve_steps
        self.eat_value = eat_value
        self.npc_damage = npc_damage
        self.nutrient_respawn_prob = nutrient_respawn_prob
        self.max_steps = max_steps
        self.death_reward = death_reward
        self.render_mode = render_mode

        view_size = self.view_dim * self.view_dim * N_CHANNELS
        self.obs_dim = view_size + 2  # + health + hunger
        self.observation_space = spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))

        # World state (allocated on reset).
        self.obstacles = np.zeros((size, size), dtype=bool)
        self.nutrients = np.zeros((size, size), dtype=bool)
        self.npcs: list[tuple[int, int]] = []
        self.agent = (0, 0)
        self.health = 1.0
        self.hunger = 0.0
        self.steps = 0

    # ---- world helpers -------------------------------------------------

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def _free(self, r: int, c: int) -> bool:
        """A cell an agent/NPC may stand on: in-bounds and not an obstacle."""
        return self._in_bounds(r, c) and not self.obstacles[r, c]

    def _random_free_cell(self) -> tuple[int, int]:
        while True:
            r = self.np_random.integers(self.size)
            c = self.np_random.integers(self.size)
            if self._free(r, c):
                return int(r), int(c)

    def _spawn_nutrients(self) -> None:
        """Top nutrients up toward ``n_nutrients`` (stochastically per step)."""
        while self.nutrients.sum() < self.n_nutrients:
            if self.np_random.random() >= self.nutrient_respawn_prob:
                break
            r, c = self._random_free_cell()
            if (r, c) != self.agent:
                self.nutrients[r, c] = True

    # ---- gymnasium API -------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Static obstacles, with a guaranteed-clear border-less interior.
        self.obstacles = (
            self.np_random.random((self.size, self.size)) < self.obstacle_density
        )

        self.nutrients = np.zeros((self.size, self.size), dtype=bool)
        self.agent = self._random_free_cell()
        self.obstacles[self.agent] = False  # never spawn on an obstacle

        self.npcs = []
        for _ in range(self.n_npcs):
            cell = self._random_free_cell()
            if cell != self.agent:
                self.npcs.append(cell)

        # Fill nutrients to target (ignore respawn prob on the initial fill).
        for _ in range(self.n_nutrients):
            r, c = self._random_free_cell()
            if (r, c) != self.agent:
                self.nutrients[r, c] = True

        self.health = 1.0
        self.hunger = 0.0
        self.steps = 0
        return self._obs(), self._info()

    def step(self, action: int):
        action = int(action)
        self.steps += 1

        # 1. Move the agent (blocked moves are no-ops).
        dr, dc = ACTIONS[action]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc
        if self._free(nr, nc):
            self.agent = (nr, nc)

        # 2. Hunger rises.
        self.hunger = min(1.0, self.hunger + self.hunger_rate)

        # 3. Eat any nutrient underfoot.
        if self.nutrients[self.agent]:
            self.nutrients[self.agent] = False
            self.hunger = max(0.0, self.hunger - self.eat_value)

        # 4. Starve while fully hungry.
        if self.hunger >= 1.0:
            self.health -= self.starve_rate

        # 5. NPCs wander; contact damages the agent.
        self._move_npcs()
        contacts = sum(1 for npc in self.npcs if npc == self.agent)
        if contacts:
            self.health -= self.npc_damage * contacts

        # 6. Respawn nutrients.
        self._spawn_nutrients()

        # 7. Death / timeout.
        self.health = min(1.0, self.health)
        terminated = self.health <= 0.0
        truncated = self.steps >= self.max_steps
        reward = self.death_reward if terminated else 0.0
        if terminated:
            self.health = 0.0

        return self._obs(), reward, terminated, truncated, self._info()

    def _move_npcs(self) -> None:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        new = []
        for r, c in self.npcs:
            dr, dc = moves[self.np_random.integers(len(moves))]
            tr, tc = r + dr, c + dc
            new.append((tr, tc) if self._free(tr, tc) else (r, c))
        self.npcs = new

    # ---- observation ---------------------------------------------------

    def _obs(self) -> np.ndarray:
        view = np.zeros((self.view_dim, self.view_dim, N_CHANNELS), np.float32)
        ar, ac = self.agent
        rad = self.view_radius
        for i in range(self.view_dim):
            for j in range(self.view_dim):
                r, c = ar + (i - rad), ac + (j - rad)
                if not self._in_bounds(r, c) or self.obstacles[r, c]:
                    view[i, j, CH_OBSTACLE] = 1.0  # border reads as obstacle
                if self._in_bounds(r, c) and self.nutrients[r, c]:
                    view[i, j, CH_NUTRIENT] = 1.0
        # NPCs (placed after so they sit "on top" in the view).
        for npr, npc in self.npcs:
            i, j = npr - ar + rad, npc - ac + rad
            if 0 <= i < self.view_dim and 0 <= j < self.view_dim:
                view[i, j, CH_NPC] = 1.0
        return np.concatenate(
            [view.ravel(), np.array([self.health, self.hunger], np.float32)]
        )

    def _info(self) -> dict:
        return {
            "health": self.health,
            "hunger": self.hunger,
            "steps": self.steps,
            "agent": self.agent,
            "n_nutrients": int(self.nutrients.sum()),
        }

    # ---- rendering -----------------------------------------------------

    def render(self):
        if self.render_mode in ("ansi", "human", None):
            return self.render_full() + "\n\n" + self.render_view()
        return None

    def _glyph(self, r: int, c: int) -> str:
        if (r, c) == self.agent:
            return "@"
        if (r, c) in self._npc_set():
            return "N"
        if self.obstacles[r, c]:
            return "#"
        if self.nutrients[r, c]:
            return "*"
        return "."

    def _npc_set(self) -> set:
        return set(self.npcs)

    def render_full(self) -> str:
        """ASCII of the whole world."""
        npcs = self._npc_set()
        rows = []
        for r in range(self.size):
            line = []
            for c in range(self.size):
                if (r, c) == self.agent:
                    line.append("@")
                elif (r, c) in npcs:
                    line.append("N")
                elif self.obstacles[r, c]:
                    line.append("#")
                elif self.nutrients[r, c]:
                    line.append("*")
                else:
                    line.append(".")
            rows.append(" ".join(line))
        header = (
            f"step {self.steps:4d}   "
            f"health {_bar(self.health)} {self.health:4.2f}   "
            f"hunger {_bar(self.hunger)} {self.hunger:4.2f}"
        )
        return header + "\n" + "\n".join(rows)

    def render_view(self) -> str:
        """ASCII of the egocentric 7x7 kernel the agent actually observes."""
        view = self._obs()[: self.view_dim * self.view_dim * N_CHANNELS]
        view = view.reshape(self.view_dim, self.view_dim, N_CHANNELS)
        rad = self.view_radius
        rows = []
        for i in range(self.view_dim):
            line = []
            for j in range(self.view_dim):
                if (i, j) == (rad, rad):
                    line.append("@")
                elif view[i, j, CH_NPC]:
                    line.append("N")
                elif view[i, j, CH_OBSTACLE]:
                    line.append("#")
                elif view[i, j, CH_NUTRIENT]:
                    line.append("*")
                else:
                    line.append(".")
            rows.append(" ".join(line))
        return "egocentric view (obs):\n" + "\n".join(rows)


def _bar(x: float, width: int = 10) -> str:
    x = max(0.0, min(1.0, x))
    filled = int(round(x * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# Optional gymnasium registration so `gym.make("SurvivalGrid-v0")` works.
try:
    gym.register(id="SurvivalGrid-v0", entry_point=__name__ + ":SurvivalGridEnv")
except Exception:
    pass
