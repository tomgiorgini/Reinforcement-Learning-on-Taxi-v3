from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import gymnasium as gym
import torch

# Add project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import save_single_run_curve

from deep.DQN import DQN


# Run greedy DQN policy for given episodes and return metrics arrays.
@torch.no_grad()
def run_greedy_dqn(
    env_id: str,
    policy_net: torch.nn.Module,
    device: torch.device,
    seed: int,
    episodes: int,
    max_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Create environment
    env = gym.make(env_id)

    rewards = np.zeros(episodes, dtype=float)
    steps = np.zeros(episodes, dtype=int)
    penalties = np.zeros(episodes, dtype=int)
    success = np.zeros(episodes, dtype=float)

    # policy in evaluation mode 
    policy_net.eval()

    for ep in range(episodes):
        # reset environment with test seed
        obs, _ = env.reset(seed=seed + ep)

        ep_r, ep_steps, ep_pen = 0.0, 0, 0
        ok = 0.0

        for _ in range(max_steps):
            # convert obs to tensor 
            st = torch.tensor([obs], dtype=torch.long, device=device)
            # select action greedily from policy_net
            action = int(policy_net(st).argmax(dim=1).item())

            obs, r, terminated, truncated, _ = env.step(action)

            # update episode metrics
            ep_r += float(r)
            ep_steps += 1
            if r == -10:
                ep_pen += 1
            if terminated and r == 20:
                ok = 1.0

            if terminated or truncated:
                break

        rewards[ep] = ep_r
        steps[ep] = ep_steps
        penalties[ep] = ep_pen
        success[ep] = ok

    env.close()
    policy_net.train()
    return rewards, steps, penalties, success

# Load a DQN model from the given path and return it ready for evaluation
def load_dqn_model(
    env_id: str,
    model_path: str,
    device: torch.device,
    embedding_dim: int,
    hidden_dim: int,
) -> torch.nn.Module:
    # Create a DQN model with the same architecture used for training
    env = gym.make(env_id)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    env.close()

    model = DQN(
        n_states=n_states,
        n_actions=n_actions,
        emb_dim=embedding_dim,
        hidden=hidden_dim,
    ).to(device)
    # Load the saved state dict into the model
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model


if __name__ == "__main__":
    ENV_ID = "Taxi-v3"
    SEED = 42
    TEST_EPISODES = 100
    MAX_STEPS = 200
    ROLLING = 50

    MODEL_PATH = "results/train_dqn/dqn_seed42.pth"

    OUTDIR = "results/test_dqn"
    os.makedirs(OUTDIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EMBEDDING_DIM = 32
    HIDDEN_DIM = 128

    policy_net = load_dqn_model(
        env_id=ENV_ID,
        model_path=MODEL_PATH,
        device=device,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    rewards, steps, penalties, success = run_greedy_dqn(
        env_id=ENV_ID,
        policy_net=policy_net,
        device=device,
        seed=SEED + 50_000,   # separate test seed space (avoid overlap with training)
        episodes=TEST_EPISODES,
        max_steps=MAX_STEPS,
    )

    # Print summary
    print("\n[DQN TEST]")
    print(f"Model: {MODEL_PATH}")
    print(f"Episodes: {TEST_EPISODES} | seed base: {SEED}")
    print(f"Device: {device}")
    print(f"Mean reward: {rewards.mean():.2f}")
    print(f"Mean steps: {steps.mean():.2f}")
    print(f"Mean penalties: {penalties.mean():.2f}")
    print(f"Success rate: {success.mean():.3f}")

    # Plots
    ep_axis = np.arange(1, TEST_EPISODES + 1)

    save_single_run_curve(
        ep_axis, rewards,
        title="Greedy TEST - Reward per episode (100)",
        xlabel="Test episode", ylabel="Reward",
        outpath=os.path.join(OUTDIR, "test_reward_100.png"),
        rolling_window=ROLLING,
    )

    save_single_run_curve(
        ep_axis, steps,
        title="Greedy TEST - Steps per episode (100)",
        xlabel="Test episode", ylabel="Steps",
        outpath=os.path.join(OUTDIR, "test_steps_100.png"),
        rolling_window=ROLLING,
    )

    save_single_run_curve(
        ep_axis, penalties,
        title="Greedy TEST - Penalties per episode (100)",
        xlabel="Test episode", ylabel="Penalties",
        outpath=os.path.join(OUTDIR, "test_penalties_100.png"),
        rolling_window=ROLLING,
    )

    save_single_run_curve(
        ep_axis, success,
        title="Greedy TEST - Success per episode (100)",
        xlabel="Test episode", ylabel="Success",
        outpath=os.path.join(OUTDIR, "test_success_100.png"),
        rolling_window=ROLLING,
    )

    print(f"\nPlots saved in: {OUTDIR}\n")
