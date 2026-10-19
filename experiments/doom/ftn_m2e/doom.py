"""Synchronous single-lifetime Doom, with explicit discrete button combinations."""
import itertools
from pathlib import Path
import re
import struct

import numpy as np
import vizdoom as vzd


BUTTON_NAMES = [
    'MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_LEFT', 'MOVE_RIGHT',
    'TURN_LEFT', 'TURN_RIGHT', 'ATTACK', 'SPEED', 'USE',
] + [f'SELECT_WEAPON{i}' for i in range(1, 8)]


def action_catalog():
    rows, names = [], []
    for move, strafe, turn, fire, run in itertools.product(
        (-1, 0, 1), (-1, 0, 1), (-1, 0, 1), (0, 1), (0, 1)
    ):
        # Running while stationary is redundant, and does not alter aim speed.
        if run and move == strafe == 0:
            continue
        buttons = [move > 0, move < 0, strafe < 0, strafe > 0,
                   turn < 0, turn > 0, fire, run, 0] + [0] * 7
        rows.append(list(map(int, buttons)))
        names.append(f'move={move} strafe={strafe} turn={turn} fire={fire} run={run}')
    for move in (0, 1, -1):
        rows.append([int(move > 0), int(move < 0), 0, 0, 0, 0, 0, 0, 1] + [0] * 7)
        names.append(f'use move={move}')
    for weapon in range(7):
        row = [0] * len(BUTTON_NAMES)
        row[9 + weapon] = 1
        rows.append(row)
        names.append(f'weapon {weapon + 1}')
    return rows, names


def normalize_telemetry(health, ammo):
    """Bounded health and selected-weapon ammo; zero remains zero."""
    return np.array([np.clip(health / 200., 0., 1.),
                     np.clip(np.log1p(max(0., ammo)) / np.log1p(600.), 0., 1.)], np.float32)


class Doom:
    def __init__(self, wad, seed, map_number=1, skill=3, tics=4):
        # Read the actual IWAD map directory: Doom uses E1M1, Doom II MAP01.
        with Path(wad).open('rb') as source:
            magic, count, directory = struct.unpack('<4sii', source.read(12))
            if magic != b'IWAD':
                raise ValueError('Expected an IWAD containing a complete game')
            source.seek(directory)
            names = [struct.unpack('<ii8s', source.read(16))[2].rstrip(b'\0').decode()
                     for _ in range(count)]
        self.maps = sorted({name for name in names if re.fullmatch(r'MAP\d\d', name)})
        if not self.maps:
            self.maps = sorted({name for name in names if re.fullmatch(r'E\dM[1-8]', name)})
        if not self.maps or not 1 <= map_number <= len(self.maps):
            raise ValueError('No supported map at the requested campaign index')
        self.game = vzd.DoomGame()
        self.game.set_doom_game_path(str(Path(wad).resolve()))
        self.map_number = map_number
        self.game.set_doom_map(self.map_name)
        self.game.set_doom_skill(skill)
        self.game.set_seed(seed)
        self.game.set_window_visible(False)
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_render_hud(True)
        self.game.set_render_crosshair(True)
        self.game.set_render_messages(False)
        self.game.set_automap_buffer_enabled(False)
        self.game.set_labels_buffer_enabled(False)
        self.game.set_depth_buffer_enabled(False)
        self.game.set_episode_timeout(0)
        self.game.set_episode_start_time(1)
        self.game.set_living_reward(0.)
        self.game.set_death_penalty(0.)
        self.game.set_map_exit_reward(0.)
        self.game.set_available_buttons([getattr(vzd.Button, name) for name in BUTTON_NAMES])
        self.game.set_available_game_variables([
            vzd.GameVariable.HEALTH, vzd.GameVariable.SELECTED_WEAPON_AMMO])
        self.game.add_game_args('+screenblocks 10')
        self.actions, self.action_names = action_catalog()
        self.tics = tics
        self.game.init()

    def observation(self):
        state = self.game.get_state()
        if state is None:
            return None
        return state.screen_buffer.copy(), state.game_variables.copy()

    @property
    def map_name(self):
        return self.maps[self.map_number - 1]

    def step(self, action):
        # Deliberately discard the engine reward, including any WAD-defined rewards.
        self.game.make_action(self.actions[action], self.tics)
        ended = self.game.is_episode_finished()
        dead = self.game.is_player_dead()
        return ended, dead

    def reset(self, completed=False):
        if completed:
            self.map_number = self.map_number % len(self.maps) + 1
            self.game.set_doom_map(self.map_name)
        self.game.new_episode()

    def close(self):
        self.game.close()
