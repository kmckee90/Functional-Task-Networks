from pathlib import Path

import numpy as np
import pytest

from .doom import Doom, action_catalog, normalize_telemetry


def test_controls_and_feedback():
    actions, names = action_catalog()
    assert len(actions) == len(names) == len(set(map(tuple, actions)))
    assert any(a[0] and a[3] and a[4] and a[6] and a[7] for a in actions)
    for action in actions:
        assert not (action[0] and action[1])
        assert not (action[2] and action[3])
        assert not (action[4] and action[5])
    np.testing.assert_array_equal(normalize_telemetry(-1, -1), [0, 0])
    np.testing.assert_array_equal(normalize_telemetry(500, 10000), [1, 1])
    assert normalize_telemetry(100, 50)[0] == .5


@pytest.mark.parametrize('wad,map_name', [
    ('src/freedoom2.wad', 'MAP01'),
    ('wad/Doom + Doom II Collection WAD files/doom.wad', 'E1M1'),
])
def test_direct_game_and_zero_rewards(wad, map_name):
    env = Doom(Path(wad), 123)
    try:
        assert env.map_name == map_name
        image, telemetry = env.observation()
        assert image.shape == (480, 640, 3)
        assert telemetry.shape == (2,)
        assert image[-60:].std() > 10  # HUD rendered, not a black letterbox.
        assert env.game.get_living_reward() == env.game.get_death_penalty() == 0
        assert env.game.get_map_exit_reward() == 0
        assert env.game.get_episode_timeout() == 0
        action = next(i for i, a in enumerate(env.actions) if a[0] and a[3] and a[6])
        env.step(action)
        assert env.game.get_total_reward() == 0
        assert not np.array_equal(env.observation()[0], image)
        env.game.send_game_command('kill')
        for _ in range(10):
            if env.game.is_episode_finished():
                break
            env.step(action)
        assert env.game.is_player_dead()
        env.reset()
        assert env.observation() is not None
        assert env.map_number == 1
        env.reset(completed=True)
        assert env.map_name == ('E1M2' if map_name == 'E1M1' else 'MAP02')
        assert env.observation() is not None
    finally:
        env.close()
