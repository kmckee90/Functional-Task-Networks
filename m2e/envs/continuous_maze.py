# MIT License

# Copyright (c) 2022 Quentin Gallouédec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Forked from https://github.com/qgallouedec/gym-continuous-maze/blob/main/gym_continuous_maze/gym_continuous_maze.py

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
import torch
from gymnasium import spaces
from pygame import gfxdraw

_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_RED = (255, 0, 0)
_DARKRED = (120, 0, 0)
_YELLOW = (255, 255, 0)
_BLUE = (0, 0, 255)
_DARKBLUE = (0, 0, 128)
_GREEN = (0, 255, 0)
_TEAL = (0, 255, 255)


def detect_in_spot(pos, x_loc: float, y_loc: float, radius: float):
    lb_x = x_loc - radius
    ub_x = x_loc + radius
    lb_y = y_loc - radius
    ub_y = y_loc + radius
    return bool(np.all((pos[0] > lb_x) * (pos[0] < ub_x)) and np.all((pos[1] > lb_y) * (pos[1] < ub_y)))


def remove_square(walls, x_loc: float, y_loc: float, radius: float):
    lb_x = x_loc - radius
    ub_x = x_loc + radius
    lb_y = y_loc - radius
    ub_y = y_loc + radius
    removed_walls = []
    for i, wall in enumerate(walls):
        wall_arr = np.array(wall)
        x = wall_arr[[0, 2]]
        y = wall_arr[[1, 3]]
        if np.all((x > lb_x) * (x < ub_x)) and np.all((y > lb_y) * (y < ub_y)):
            removed_walls.append(i)
    new_walls = [wall for idx, wall in enumerate(walls) if idx not in removed_walls]
    return new_walls


def generate_maze(n, center_box_radius: float, noisy_spots: bool, noisy_spot_radius: float):
    # Initialize the maze grid, walls and the set of visited cells
    maze = np.zeros((n, n), dtype=bool)  # True means part of the maze
    walls = []  # List of walls, where walls are ((x1, y1), (x2, y2))
    walls_to_remove = []

    # Set the outer boundary as part of the maze
    maze[0, :] = True
    maze[-1, :] = True
    maze[:, 0] = True
    maze[:, -1] = True

    # Start from a random cell inside the boundary
    start_x, start_y = np.random.randint(1, n - 2), np.random.randint(1, n - 2)
    maze[start_x, start_y] = True
    add_walls(walls, start_x, start_y, n)

    while walls:
        # Randomly select a wall
        if len(walls) == 1:
            wall = walls.pop(0)
        else:
            wall = walls.pop(np.random.randint(0, len(walls) - 1))
        x1, y1, x2, y2 = wall

        # Check if the new cell is inside boundaries and not part of the maze
        if 0 < x2 < n - 1 and 0 < y2 < n - 1 and not maze[x2, y2]:
            # Make the new cell part of the maze
            maze[x2, y2] = True
            # Remove the wall by adding it to the final list
            walls_to_remove.append(wall)
            # Add the walls of the new cell to the list
            add_walls(walls, x2, y2, n)

    # Convert walls to remove into the format (n, 2, 2)
    walls_to_remove.append((0, 0, 0, n - 1))
    walls_to_remove.append((0, 0, n - 1, 0))
    walls_to_remove.append((0, n - 1, n - 1, n - 1))
    walls_to_remove.append((n - 1, 0, n - 1, n - 1))

    possible_closes = [
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (0, n - 2, 1, n - 2),
        (1, n - 1, 1, n - 2),
        (n - 2, 0, n - 2, 1),
        (n - 1, 1, n - 2, 1),
        (n - 1, n - 2, n - 2, n - 2),
        (n - 2, n - 1, n - 2, n - 2),
    ]
    walls_to_remove.append(possible_closes[np.random.choice(range(8))])

    walls_to_remove = remove_square(walls_to_remove, x_loc=(n - 1) / 2, y_loc=(n - 1) / 2, radius=center_box_radius)

    if noisy_spots:
        walls_to_remove = remove_square(walls_to_remove, x_loc=(n - 1) / 4, y_loc=(n - 1) / 4, radius=noisy_spot_radius)
        walls_to_remove = remove_square(
            walls_to_remove, x_loc=(n - 1) * 3 / 4, y_loc=(n - 1) * 3 / 4, radius=noisy_spot_radius
        )
        walls_to_remove = remove_square(
            walls_to_remove, x_loc=(n - 1) / 4, y_loc=(n - 1) * 3 / 4, radius=noisy_spot_radius
        )
        walls_to_remove = remove_square(
            walls_to_remove, x_loc=(n - 1) * 3 / 4, y_loc=(n - 1) / 4, radius=noisy_spot_radius
        )

    walls_array = np.array([[[x1, y1], [x2, y2]] for (x1, y1, x2, y2) in walls_to_remove])
    # walls_array = walls_array*2 - n + 1
    walls_array = np.float32(walls_array)
    walls_array /= walls_array.max()
    walls_array *= 24
    walls_array -= 12
    return walls_array


def add_walls(walls, x, y, n):
    # Helper function to add walls to the list if next cell is inside boundary
    if x > 1:
        walls.append((x, y, x - 1, y))  # Wall to the left
    if x < n - 2:
        walls.append((x, y, x + 1, y))  # Wall to the right
    if y > 1:
        walls.append((x, y, x, y - 1))  # Wall above
    if y < n - 2:
        walls.append((x, y, x, y + 1))  # Wall below


def _intersection(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray | None:
    """
    Get the intersection of [A, B] and [C, D]. Return False if segment don't cross.

    :param A: Point of the first segment
    :param B: Point of the first segment
    :param C: Point of the second segment
    :param D: Point of the second segment
    :return: The intersection if any, otherwise None.
    """
    det = (B[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (B[1] - A[1])
    if det == 0:
        # Parallel
        return None
    else:
        t1 = ((C[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (C[1] - A[1])) / det
        t2 = ((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / det
        if t1 > 1 or t1 < 0 or t2 > 1 or t2 < 0:
            # not intersect
            return None
        else:
            xi = A[0] + t1 * (B[0] - A[0])
            yi = A[1] + t1 * (B[1] - A[1])
            return np.array([xi, yi])


class CoverageCalculator:

    low = -12
    high = 12

    def __init__(self, resolution, device):
        self._bin_width = torch.full([2], (self.high - self.low) / resolution, device=device)
        self._covered = torch.zeros([resolution, resolution], device=device, dtype=torch.float32)

    def calculate(self, kernel_size: int | None = None) -> float:
        covered = self.values(kernel_size)
        return covered.sum().item() / covered.numel()

    def update(self, locations: torch.Tensor) -> None:
        indices = ((locations[:,:2] - self.low) / self._bin_width).int()
        indices = torch.clamp(indices, 0, self._covered.shape[0] - 1)
        self._covered[indices[:, 0], indices[:, 1]] = 1

    def values(self, kernel_size: int | None = None) -> torch.Tensor:
        if kernel_size is None or kernel_size == 1:
            return self._covered
        else:
            maxpool = torch.nn.MaxPool2d(kernel_size)
            return maxpool(self._covered.unsqueeze(0)).squeeze()


class ContinuousMaze(gym.Env):
    """Continuous maze environment with no rewards (goal-less).

    Arguments:
        max_steps (int): Max number of steps that can be taken in the environment before completion.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}
    action_space = spaces.Box(-1, 1, (2,))

    old_walls = np.array(
        [
            [[-12.0, -12.0], [-12.0, 12.0]],
            [[-10.0, 8.0], [-10.0, 10.0]],
            [[-10.0, 0.0], [-10.0, 6.0]],
            [[-10.0, -4.0], [-10.0, -2.0]],
            [[-10.0, -10.0], [-10.0, -6.0]],
            [[-8.0, 4.0], [-8.0, 8.0]],
            [[-8.0, -4.0], [-8.0, 0.0]],
            [[-8.0, -8.0], [-8.0, -6.0]],
            [[-6.0, 8.0], [-6.0, 10.0]],
            [[-6.0, 4.0], [-6.0, 6.0]],
            [[-6.0, 0.0], [-6.0, 2.0]],
            [[-6.0, -6.0], [-6.0, -4.0]],
            [[-4.0, 2.0], [-4.0, 8.0]],
            [[-4.0, -2.0], [-4.0, 0.0]],
            [[-4.0, -10.0], [-4.0, -6.0]],
            [[-2.0, 8.0], [-2.0, 12.0]],
            [[-2.0, 2.0], [-2.0, 6.0]],
            [[-2.0, -4.0], [-2.0, -2.0]],
            [[0.0, 6.0], [0.0, 12.0]],
            [[0.0, 2.0], [0.0, 4.0]],
            [[0.0, -8.0], [0.0, -6.0]],
            [[2.0, 8.0], [2.0, 10.0]],
            [[2.0, -8.0], [2.0, 6.0]],
            [[4.0, 10.0], [4.0, 12.0]],
            [[4.0, 4.0], [4.0, 6.0]],
            [[4.0, 0.0], [4.0, 2.0]],
            [[4.0, -6.0], [4.0, -2.0]],
            [[4.0, -10.0], [4.0, -8.0]],
            [[6.0, 10.0], [6.0, 12.0]],
            [[6.0, 6.0], [6.0, 8.0]],
            [[6.0, 0.0], [6.0, 2.0]],
            [[6.0, -8.0], [6.0, -6.0]],
            [[8.0, 10.0], [8.0, 12.0]],
            [[8.0, 4.0], [8.0, 6.0]],
            [[8.0, -4.0], [8.0, 2.0]],
            [[8.0, -10.0], [8.0, -8.0]],
            [[10.0, 10.0], [10.0, 12.0]],
            [[10.0, 4.0], [10.0, 8.0]],
            [[10.0, -2.0], [10.0, 0.0]],
            [[12.0, -12.0], [12.0, 12.0]],
            [[-12.0, 12.0], [12.0, 12.0]],
            [[-12.0, 10.0], [-10.0, 10.0]],
            [[-8.0, 10.0], [-6.0, 10.0]],
            [[-4.0, 10.0], [-2.0, 10.0]],
            [[2.0, 10.0], [4.0, 10.0]],
            [[-8.0, 8.0], [-2.0, 8.0]],
            [[2.0, 8.0], [8.0, 8.0]],
            [[-10.0, 6.0], [-8.0, 6.0]],
            [[-6.0, 6.0], [-2.0, 6.0]],
            [[6.0, 6.0], [8.0, 6.0]],
            [[0.0, 4.0], [6.0, 4.0]],
            [[-10.0, 2.0], [-6.0, 2.0]],
            [[-2.0, 2.0], [0.0, 2.0]],
            [[8.0, 2.0], [10.0, 2.0]],
            [[-4.0, 0.0], [-2.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0]],
            [[6.0, 0.0], [8.0, 0.0]],
            [[-6.0, -2.0], [2.0, -2.0]],
            [[4.0, -2.0], [10.0, -2.0]],
            [[-12.0, -4.0], [-8.0, -4.0]],
            [[-4.0, -4.0], [-2.0, -4.0]],
            [[0.0, -4.0], [6.0, -4.0]],
            [[8.0, -4.0], [10.0, -4.0]],
            [[-8.0, -6.0], [-6.0, -6.0]],
            [[-2.0, -6.0], [0.0, -6.0]],
            [[6.0, -6.0], [10.0, -6.0]],
            [[-12.0, -8.0], [-6.0, -8.0]],
            [[-2.0, -8.0], [2.0, -8.0]],
            [[4.0, -8.0], [6.0, -8.0]],
            [[8.0, -8.0], [10.0, -8.0]],
            [[-10.0, -10.0], [-8.0, -10.0]],
            [[-4.0, -10.0], [4.0, -10.0]],
            [[-12.0, -12.0], [12.0, -12.0]],
        ]
    )

    def __init__(
        self,
        size: float = 12.0,
        max_steps: int = 100,
        randomized: bool = False,
        seed: int | None = None,
        old_walls: bool = False,
        center_box_radius: float = 1.0,
        noisy_spots: bool = False,
        noisy_spot_radius: float = 2.0,
    ) -> None:
        self.noisy_spots = noisy_spots
        self.observation_space = spaces.Box(-12, 12, (2 + int(noisy_spots),))

        self._screen = None
        self._isopen = True
        self._all_pos: list | deque = []
        self._last_pos: list | deque = []
        self.max_steps = max_steps
        self._cur_step: int | None = None
        self._pos: np.ndarray | None = None
        self._goal: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.value_coords: np.ndarray | None = None
        self.randomized = randomized
        self.center_box_radius = center_box_radius
        self.noisy_spot_radius = noisy_spot_radius

        self.size = size
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        self.walls = generate_maze(size, self.center_box_radius, self.noisy_spots, self.noisy_spot_radius)
        if old_walls:
            self.walls = self.old_walls

    @property
    def pos(self):
        return self._pos

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, goal: np.ndarray | None):
        self._goal = goal

    @pos.setter
    def pos(self, pos: np.ndarray | None):
        self._pos = pos

    @property
    def all_pos(self):
        return self._all_pos

    @all_pos.setter
    def all_pos(self, all_pos: list[Any]):
        self._all_pos = all_pos

    def step(self, action: np.ndarray | int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Convert discrete inputs to continuous ones:
        if isinstance(action, int):
            if action == 0:
                action = np.array([0.0, 1.0])
            elif action == 1:
                action = np.array([0.0, -1.0])
            elif action == 2:
                action = np.array([1.0, 0.0])
            elif action == 3:
                action = np.array([-1.0, 0.0])
        assert isinstance(action, np.ndarray)

        # action *= 0.5
        action += np.random.randn(2) * 0.05

        assert self._pos is not None
        new_pos = self._pos + action
        for wall in self.walls:
            intersection = _intersection(wall[0], wall[1], self._pos, new_pos)
            if intersection is not None:
                new_pos = self._pos
        self._pos = new_pos
        self._all_pos.append(self._pos.copy())
        self._last_pos.append(self._pos.copy())
        assert self._cur_step is not None
        self._cur_step += 1
        done = self._cur_step >= self.max_steps

        if self.noisy_spots:
            n = 24.0
            gen_noise = False
            if detect_in_spot(self._pos, n / 4 - n / 2, n / 4 - n / 2, radius=self.noisy_spot_radius):
                gen_noise = True
            if detect_in_spot(self._pos, n * 3 / 4 - n / 2, n * 3 / 4 - n / 2, radius=self.noisy_spot_radius):
                gen_noise = True
            if detect_in_spot(self._pos, n / 4 - n / 2, n * 3 / 4 - n / 2, radius=self.noisy_spot_radius):
                gen_noise = True
            if detect_in_spot(self._pos, n * 3 / 4 - n / 2, n / 4 - n / 2, radius=self.noisy_spot_radius):
                gen_noise = True
            if gen_noise:
                noise = np.random.rand(1) * 2 - 1
            else:
                noise = np.zeros(1)
            obs = np.concatenate((self._pos.copy(), noise))
        else:
            obs = self._pos.copy()
        return obs, 0.0, done, False, {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if self.randomized:
            self.walls = generate_maze(self.size, self.center_box_radius, self.noisy_spots, self.noisy_spot_radius)

        self._pos = np.zeros(2)
        self._all_pos.append(self._pos.copy())
        self._cur_step = 0
        self._last_pos.clear()

        if self.noisy_spots:
            n = 24.0
            gen_noise = False
            if detect_in_spot(self._pos, x_loc=n / 4 - n / 2, y_loc=n / 4 - n / 2, radius=self.noisy_spot_radius):
                gen_noise = True
            if detect_in_spot(
                self._pos, x_loc=n * 3 / 4 - n / 2, y_loc=n * 3 / 4 - n / 2, radius=self.noisy_spot_radius
            ):
                gen_noise = True
            if detect_in_spot(self._pos, x_loc=n / 4 - n / 2, y_loc=n * 3 / 4 - n / 2, radius=self.noisy_spot_radius):
                gen_noise = True
            if detect_in_spot(self._pos, x_loc=n * 3 / 4 - n / 2, y_loc=n / 4 - n / 2, radius=self.noisy_spot_radius):
                gen_noise = True
            if gen_noise:
                noise = np.random.rand(1) * 2 - 1
            else:
                noise = np.zeros(1)
            obs = np.concatenate((self._pos.copy(), noise))
        else:
            obs = self._pos.copy()

        return obs, {}

    def render(self, mode: str = "human", render_data="data", input_data = None, rewards = None) -> np.ndarray | None:
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid mode in render. Valid modes are {self.metadata['render_modes']}")

        screen_dim = 500
        bound = 13
        scale = screen_dim / (bound * 2)
        offset = screen_dim // 2

        if self._screen is None:
            pygame.init()
            pygame.display.list_modes()
            self._screen = pygame.display.set_mode((screen_dim, screen_dim))

        surf = pygame.Surface((screen_dim, screen_dim))
        surf.fill(_BLACK)

        if self.noisy_spots:
            n = 24
            locs = [1 / 4, 3 / 4]
            for i in locs:
                for j in locs:
                    gfxdraw.filled_polygon(
                        surf,
                        [
                            (
                                offset + scale * (n * i - n / 2 + self.noisy_spot_radius / (self.size / 12.0)),
                                offset + scale * (n * j - n / 2 + self.noisy_spot_radius / (self.size / 12.0)),
                            ),
                            (
                                offset + scale * (n * i - n / 2 + self.noisy_spot_radius / (self.size / 12.0)),
                                offset + scale * (n * j - n / 2 - self.noisy_spot_radius / (self.size / 12.0)),
                            ),
                            (
                                offset + scale * (n * i - n / 2 - self.noisy_spot_radius / (self.size / 12.0)),
                                offset + scale * (n * j - n / 2 - self.noisy_spot_radius / (self.size / 12.0)),
                            ),
                            (
                                offset + scale * (n * i - n / 2 - self.noisy_spot_radius / (self.size / 12.0)),
                                offset + scale * (n * j - n / 2 + self.noisy_spot_radius / (self.size / 12.0)),
                            ),
                        ],
                        _DARKBLUE,
                    )

        # if input_data is not None:
        #     data = input_data
        # else:
        #     data = self._all_pos
            

        if render_data == "data":

            if input_data is not None:
                for pos in self._all_pos:  # type: ignore
                    x, y = pos * scale + offset
                    gfxdraw.filled_circle(surf, int(x), int(y), 1, _DARKRED)
                for pos in input_data:  # type: ignore
                    x, y = pos * scale + offset
                    gfxdraw.filled_circle(surf, int(x), int(y), 2, _RED)
            else:
                for pos in self._all_pos:  # type: ignore
                    x, y = pos * scale + offset
                    gfxdraw.filled_circle(surf, int(x), int(y), 1, _RED)

            for pos in self._last_pos:
                x, y = pos * scale + offset
                gfxdraw.filled_circle(surf, int(x), int(y), 2, _YELLOW)

            if self._goal is not None:
                x, y = self._goal[:2] * scale + offset
                gfxdraw.filled_circle(surf, int(x), int(y), 4, _TEAL)

            if rewards is not None:
                for pos in rewards:  # type: ignore
                    x, y = pos * scale + offset
                    gfxdraw.filled_circle(surf, int(x), int(y), 4, _GREEN)

        if render_data == "value" and self.values is not None and self.value_coords is not None:
            for pos, val in zip(self.value_coords, self.values, strict=False):
                x, y = pos * scale + offset
                value_pix = int(val.item())
                gfxdraw.filled_circle(surf, int(x), int(y), 2, (255 - value_pix, value_pix, 0))

        for wall in self.walls:
            x1, y1 = wall[0] * scale + offset
            x2, y2 = wall[1] * scale + offset
            gfxdraw.line(surf, int(x1), int(y1), int(x2), int(y2), _WHITE)

        surf = pygame.transform.flip(surf, flip_x=False, flip_y=True)
        self._screen.blit(surf, (0, 0))
        if mode == "human":
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2))

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._isopen = False