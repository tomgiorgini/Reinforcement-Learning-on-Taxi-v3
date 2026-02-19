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
    episodes: int = 3000
    max_steps_per_episode: int = 200

    # Q-learning update
    alpha: float = 0.7
    gamma: float = 0.99

    # Linear decay
    eps_start: float = 1.0
    eps_end: float = 0.10
    eps_decay_episodes: int = 1500


@dataclass
class DQNConfig:
    # Training
    episodes: int = 3000
    max_steps_per_episode: int = 200

    # Discount
    gamma: float = 0.97

    # Linear decay
    eps_start: float = 1.0
    eps_end: float = 0.10
    eps_decay_episodes: int = 1500 

    # Replay buffer
    replay_capacity: int = 50_000
    batch_size: int = 256
    learning_starts: int = 2_000   # collect steps before learning begins
    train_every_episodes: int = 2     # gradient step frequency

    # Optimizer
    lr: float = 5e-4
    grad_clip_norm: float = 10.0


    # Network architecture
    embedding_dim: int = 32
    hidden_dim: int = 128