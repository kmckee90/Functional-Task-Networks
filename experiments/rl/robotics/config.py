from dataclasses import dataclass

# Full MT10 task list (Meta-World v3).
MT10_TASKS = (
    "reach-v3",
    # "push-v3",
    # "pick-place-v3", #Not learning
    # "door-open-v3",
    # "drawer-open-v3",
    "drawer-close-v3",  # Weird one
    # "button-press-topdown-v3",
    # "peg-insert-side-v3",
    "window-open-v3",
    # "window-close-v3",
)


@dataclass
class Config:
    # --- experiment ---
    seed: int = 0
    tasks: tuple = MT10_TASKS
    steps_per_task: int = 50_000
    warmup_steps: int = 1_000  # random-action steps before learning starts (global)
    eval_episodes: int = 20  # K eval episodes per (training-phase, task)
    max_episode_steps: int = 500  # Meta-World convention
    reset_replay_between_tasks: bool = False
    reset_agent_between_tasks: bool = False
    output_path: str = "results.npz"
    log_every: int = 5_000

    # --- algorithm (DDPG) ---
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    updates_per_step: int = 1
    noise_std: float = 0.1  # Gaussian exploration noise (action range is [-1, 1])

    # --- model ---
    model_type: str = "ftn"  # "mlp" or "ftn"
    hidden_dim: int = 256  # MLP only
    layer_norm_critic: bool = True  # MLP only

    # --- FTN backbone (ParallelMLP bank + KWTA-masked readout) ---
    # Each network (actor, critic) gets its own bank + its own mask.
    ftn_hidden_size: int = 256  # H: number of parallel neuron slots per network
    ftn_inner_size: int = 8  # width inside each parallel MLP
    ftn_layers: int = 8  # hidden layers inside each parallel MLP
    ftn_dropout_p: float = 0.0  # dropout on per-neuron output vector
    ftn_k: int = 32  # KWTA budget: number of active neurons per task

    # --- FTN mask configurer (AnnealedSTE) ---
    mask_method: str = "AnnealSTE"  # only AnnealSTE wired up
    reconfig_steps: int = 20
    reconfig_lr: float = 0.3
    reconfig_n: int = 500  # support-batch size for inner mask optimization
    # Cold-start init: m_param = mask_init_noise_scale * randn(H).
    mask_init_noise_scale: float = 0.001
    mask_warm_start: bool = False  # Use previous mask. Eval does not overwrite.
    # Periodic reconfiguration during training and inference. 0=disabled
    reconfig_every_steps: int = 50
    eval_reconfig_every_steps: int = 50

    # --- FTN mask-filtered replay (overlap-gated cross-task replay) ---
    mask_replay_n_per_step: int = 1  # # replayed transitions per gradient step
    rho_low: float = 0.2
    rho_high: float = 0.8

    # --- FTN task-boundary / head-start eval ---
    # At each new-task boundary (and for head-start eval on unseen tasks), how
    # do we collect the support batch?
    #   "random"    -- uniform random actions
    #   "prev_mask" -- run the agent using the previous task's masks (with noise)
    warmup_policy: str = "random"
    # Steps of rollout to collect for the head-start eval on unseen tasks.
    head_start_rollout_steps: int = 1_000


def smoke_config() -> Config:
    """Tiny settings for end-to-end testing. Real runs should use Config() defaults."""
    return Config(
        tasks=("reach-v3", "drawer-close-v3"),
        steps_per_task=400,
        warmup_steps=100,
        eval_episodes=2,
        max_episode_steps=50,
        replay_capacity=2_000,
        hidden_dim=32,
        batch_size=64,
        log_every=100,
        output_path="results_smoke.npz",
    )


def smoke_ftn_config() -> Config:
    """Tiny FTN end-to-end test."""
    return Config(
        tasks=("reach-v3", "drawer-close-v3"),
        steps_per_task=400,
        warmup_steps=100,
        eval_episodes=2,
        max_episode_steps=50,
        replay_capacity=2_000,
        batch_size=64,
        log_every=100,
        output_path="results_ftn_smoke.npz",
        model_type="ftn",
        ftn_hidden_size=32,
        ftn_inner_size=4,
        ftn_layers=2,
        ftn_k=8,
        reconfig_steps=10,
        reconfig_n=64,
        head_start_rollout_steps=200,
    )
