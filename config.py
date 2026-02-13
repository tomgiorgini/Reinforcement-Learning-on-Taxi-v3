from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GlobalConfig:
    #env id
    env_id: str = "Taxi-v3"
    rolling_window: int = 50


@dataclass
class QLearningConfig:
    # Training
    episodes: int = 5000
    max_steps_per_episode: int = 200

    # Q-learning update
    alpha: float = 0.3
    gamma: float = 0.9

    # Linear decay
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2000


@dataclass
class DQNConfig:
    # Training
    episodes: int = 3000
    max_steps_per_episode: int = 200

    # Discount
    gamma: float = 0.99

    # Linear decay
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 80_000  #environment steps until eps_end

    # Replay buffer
    replay_capacity: int = 50_000
    batch_size: int = 128
    learning_starts: int = 2_000   # collect steps before learning begins
    train_every_steps: int = 1     # gradient step frequency

    # Optimizer
    lr: float = 1e-3
    grad_clip_norm: float = 10.0

    # Target network update
    target_update_every_steps: int = 1000

    # Network architecture
    embedding_dim: int = 32
    hidden_dim: int = 128