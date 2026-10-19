import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import count
from math import prod

import mlflow
import numpy as np
import torch

from agents.agent import RDDPG_Dual
from agents.agent_dqn import RDQN
from PIL import Image

from envs.continuous_maze import ContinuousMaze

NUM_EPISODES = 1000
CURRENT_ALGORITHM = RDQN
SERVER_URI = "databricks"
EXPERIMENT_NAME = "RDDPG Explorer Ablations"


@dataclass
class MazeConfig:
    old_walls: bool = True
    randomized: bool = False
    size: int = 13
    max_steps: int = 100
    seed: int | None = None
    center_box_radius: float = 0.0
    noisy_spots: bool = False
    noisy_spot_radius: float = 2.0
    reset: bool = True
    obs_scale = 4.0


@dataclass
class LGE_Maze(MazeConfig):
    old_walls: bool = True
    randomized: bool = False
    size: int = 13
    max_steps: int = 100
    seed: int | None = None
    center_box_radius: float = 0.0
    noisy_spots: bool = False
    noisy_spot_radius: float = 2.0
    reset: bool = True
    obs_scale = 4.0


@dataclass
class Random_Mazes(MazeConfig):
    old_walls: bool = False
    randomized: bool = True
    size: int = 14
    max_steps: int = 100
    seed: int | None = None
    center_box_radius: float = 0.0
    noisy_spots: bool = False
    noisy_spot_radius: float = 0.0
    reset: bool = True
    obs_scale = 4.0


@dataclass
class Continual_Maze(MazeConfig):
    old_walls: bool = False
    randomized: bool = False
    size: int = 24
    max_steps: int = 100
    seed: int | None = 12345
    center_box_radius: float = 2.0
    noisy_spots: bool = False
    noisy_spot_radius: float = 0.0
    reset: bool = False
    obs_scale = 4.0


@dataclass
class Noisy_Maze(MazeConfig):
    old_walls: bool = False
    randomized: bool = False
    size: int = 17
    max_steps: int = 100
    seed: int | None = 12345
    center_box_radius: float = 2.0
    noisy_spots: bool = True
    noisy_spot_radius: float = 2.0
    reset: bool = True
    obs_scale = 4.0


@dataclass
class Config:
    model_name: str = "RDDPG Dual"
    maze_config: MazeConfig = None# LGE_Maze()
    note: str = ""

    # Training and task settings
    experience_replay_interval: int = 100
    gather_interval: int = 500
    epsilon: float = 1.0
    epsilon_decay_rate: float = 0.90
    epsilon_min: float = 0.1

    # Main ablations
    FCP: bool = True
    OCP: bool = True


@dataclass
class Hyperparameters:
    k: int = 15
    polyak_beta: float = 1.0
    lr: float = 3e-4
    gamma: float = 0.9
    memory_length: int = 15000
    goal_memory_length: int = 20000
    sample_memory_length: int = 200000
    minibatch_size: int = 200
    n_optimizer_steps: int = 10
    n_optimizer_epochs: int = 10
    num_negdensity_bins: int = 8
    memory_threshold: float = 0.0
    memory_epsilon: float = 0.25
    hidden_size: int = 256
    spectral_radius: float = 1.15
    extra_zeros: int = 2
    main_sparse_reward_scale: float = 10.0
    goal_sparse_reward_scale: float = 10.0
    goal_recall_type: str = "goal"
    prioritized_replay: bool = True
    optimizer: str = "RMSprop"


def coverage(locations: torch.Tensor):
    locations_scaled = locations * cfg.maze_config.obs_scale
    x = torch.arange(-12, 12).repeat(24, 1).to(locations.device)
    coords = torch.stack((x, x.T)).reshape(2, -1).permute(1, 0) + 0.5
    dists = torch.cdist(coords, locations_scaled) < (2 ** (0.5))
    coverage = dists.any(1).sum()
    return coverage




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", default="cpu", help="CUDA or CPU")
    parser.add_argument("--task", default="ContinuousMaze")
    parser.add_argument("-r", "--render", action="store_true", default=False, help="Render with pygame")
    parser.add_argument("-l", "--log", action="store_true", default=False, help="Enable logging.")
    parser.add_argument("--notes", default="", help="Extra labeling for the run")
    parser.add_argument("--load")
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.log:
        mlflow.set_tracking_uri(uri=SERVER_URI)
        mlflow.set_experiment(f"/Shared/{EXPERIMENT_NAME}")

    device = args.device
    cfg = Config()
    cfg.maze_config = LGE_Maze()
    cfgdict = asdict(cfg)
    cfgdict.update(asdict(Hyperparameters()))
    
    env = ContinuousMaze(
        size=cfg.maze_config.size,
        max_steps=cfg.maze_config.max_steps,
        randomized=cfg.maze_config.randomized,
        seed=cfg.maze_config.seed,
        old_walls=cfg.maze_config.old_walls,
        center_box_radius=cfg.maze_config.center_box_radius,
        noisy_spots=cfg.maze_config.noisy_spots,
        noisy_spot_radius=cfg.maze_config.noisy_spot_radius,
    )
    print(f"Running maze with params:", cfg.maze_config)
    
    env.reset()

    assert env.observation_space.shape is not None
    observation_size = prod(env.observation_space.shape)
    assert env.action_space.shape is not None
    action_size = 4

    agent = RDQN(observation_size, action_size, **cfgdict).to(device)
    agent = agent.to(device)

    prev_action = torch.zeros(action_size, device=device)
    prev_reward = torch.zeros(1, device=device)
    binned_reward = torch.zeros(cfgdict["num_negdensity_bins"], device=device)

    extra_zeros = torch.zeros(cfgdict["extra_zeros"], device=device)

    max_dens = 1e-6
    step = 0
    print(f"Starting run on device {device}")
    with mlflow.start_run() if args.log else nullcontext():
        if args.log:
            mlflow.set_tag("MLFLow test on explorer", "checking out mlflow")
            mlflow.log_params(cfgdict)

        gif_frames = []
        for ep in range(NUM_EPISODES):
            if not cfg.maze_config.reset and ep > 0:
                last_pos = env.pos
            obs, _ = env.reset(seed=0)
            obs = torch.tensor(obs, dtype=torch.float, device=device) / cfg.maze_config.obs_scale
            if not cfg.maze_config.reset and ep > 0:
                env.pos = last_pos  # type: ignore
                obs[:2] = torch.tensor(last_pos, dtype=torch.float, device=device) / cfg.maze_config.obs_scale  # type: ignore
            done = False

            feedback = torch.cat((prev_action, prev_reward, binned_reward, extra_zeros), -1)

            if step > cfgdict["gather_interval"]:
                agent.epsilon *= cfgdict["epsilon_decay_rate"]
                agent.epsilon = max(agent.epsilon, cfgdict["epsilon_min"])
            for _ in count():
                with torch.no_grad():
                    # Need to build obs feedback concat here
                    _, action = agent(obs, feedback)
                    next_obs, _, done, _, _ = env.step(action.item())
                    next_obs = torch.tensor(next_obs, dtype=torch.float, device=device) / cfg.maze_config.obs_scale

                    # Density reward
                    if len(agent.obs_samples) > 0:
                        reward = agent._distance(next_obs.unsqueeze(0), agent.obs_samples[:], cfgdict["k"])  # type: ignore
                    else:
                        reward = torch.zeros(1, device=device)

                    # Add to the memory samples if it's sufficiently novel
                    if (
                        reward > cfgdict["memory_threshold"]
                        or step < 2
                        or np.random.rand(1) < cfgdict["memory_epsilon"]
                    ):
                        agent.obs_samples.append(next_obs.clone())

                    # Normalize density by its top score
                    if reward > max_dens:
                        max_dens = reward.item()

                    binned_reward = agent._discretize_unit_intervals(reward / max_dens).squeeze()
                    
                    one_hot_action = torch.eye(action_size, device=device)[action].squeeze()
                    next_feedback = torch.cat((one_hot_action, reward, binned_reward, extra_zeros), -1)

                    agent.memory.remember(
                        obs,
                        feedback,
                        next_obs,
                        next_feedback,
                        action,
                        reward,
                        torch.tensor([done], device=device),
                    )
                    obs = next_obs
                    feedback = next_feedback
                    prev_reward = reward
                    prev_action = torch.eye(action_size, device=device)[action].squeeze()


                # if (step % cfgdict["experience_replay_interval"]) == 0 and step >= cfgdict["gather_interval"]:

                agent.step()
                step += 1

                # if step % cfgdict["experience_replay_interval"] == 0 and step > cfgdict["experience_replay_interval"]:
                if done:
                    if step > cfgdict["gather_interval"]:
                        agent.replay()

                    if args.render:
                        if hasattr(agent, "goal_location"):
                            env.goal = agent.goal_location.cpu().numpy() * cfg.maze_config.obs_scale
                        env._all_pos = list(agent.obs_samples[:, :2] * cfg.maze_config.obs_scale)  # type: ignore
                        env.render("human")

                    if args.log:
                        cover = coverage(agent.obs_samples[:]).cpu().item() / 24**2  # type: ignore
                        mlflow.log_metric("Coverage", cover, step=step)

                        if ep % 10 == 0:
                            if hasattr(agent, "goal_location"):
                                env.goal = agent.goal_location.cpu().numpy() * cfg.maze_config.obs_scale
                            env._all_pos = list(agent.obs_samples[:, :2] * cfg.maze_config.obs_scale)  # type: ignore
                            img = env.render(mode="rgb_array")
                            image_obj = mlflow.Image(img)
                            # mlflow.log_metric("Coverage_Image", image_obj, step=step)
                            mlflow.log_image(img, artifact_file="rddpg_progress.png")
                            # mlflow.log_artifact("outputs/sample.png", artifact_path="images")

                            gif_frames[0].save(
                                "explorer_output.gif",
                                save_all=True,
                                append_images=gif_frames[1:],
                                duration=200,  # milliseconds per frame
                                loop=0         # 0 = infinite loop
                            )
                        img = env.render(mode="rgb_array")
                        gif_frames.append(Image.fromarray(img))


                    break
