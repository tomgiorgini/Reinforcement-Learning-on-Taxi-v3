"""
deep/train_DQN.py

Training script for Taxi-v3 using:
- DQN (with target network)  [classic stabilization]
- DDQN (Double DQN target)   [improvement over classic DQN]

Structure intentionally similar to the reference project:
- set_seed()
- ReplayBuffer
- plot_curves()
- evaluate_model()
- train()
- main()

Corrections vs many "student versions":
1) Epsilon decay is computed consistently (not re-derived inside loops in a wrong way).
2) Terminal handling uses done = terminated OR truncated.
3) DDQN target is correct:
     a*  = argmax_a Q_policy(s', a)
     y   = r + gamma * Q_target(s', a*)
4) Target network update uses a clear step-based schedule.
5) Success definition is consistent with Taxi: success when reward == +20 at terminal.
"""

from __future__ import annotations

import os
import sys
import math
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict

import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Make imports work when running this file directly
# ---------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deep.DQN import DQN

# Where to save plots and best models
SAVE_DIR = "results/deep_dqn"
MODEL_DIR = os.path.join(SAVE_DIR, "models")
PLOT_DIR = os.path.join(SAVE_DIR, "plots")

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy, torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    """
    Simple uniform replay buffer.

    Why we need it:
    - Breaks correlations between consecutive transitions
    - Reuses data multiple times (more sample efficient)
    - Stabilizes deep RL training
    """
    def __init__(self, capacity: int, seed: int):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Transition:
        batch = self.rng.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))


def linear_epsilon_by_step(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """
    Linear decay of epsilon over ENVIRONMENT STEPS (not episodes).

    eps(step) = eps_start + frac*(eps_end - eps_start), frac in [0,1]
    """
    if decay_steps <= 0:
        return eps_end
    frac = min(1.0, step / decay_steps)
    return eps_start + frac * (eps_end - eps_start)


def moving_average(x: List[float], window: int) -> np.ndarray:
    """Simple moving average for smoother plots."""
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_curves(
    episode_rewards: List[float],
    episode_steps: List[int],
    episode_penalties: List[int],
    episode_successes: List[bool],
    eps_history: Optional[List[float]] = None,
    title_prefix: str = "DQN",
    window: int = 100,
) -> None:
    """
    Save plots in a way that is easy to interpret:
    - Reward: raw + moving average
    - Steps: raw + moving average
    - Penalties: raw + moving average
    - Success rate: block-based (every `window` episodes)
    - Epsilon: raw
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    episodes = np.arange(1, len(episode_rewards) + 1)

    # --- Reward (raw + MA)
    plt.figure(figsize=(9, 4))
    plt.plot(episodes, episode_rewards, alpha=0.35, label="Reward (raw)")
    ma = moving_average(episode_rewards, window)
    if len(ma) > 1:
        plt.plot(np.arange(window, window + len(ma)), ma, label=f"Reward (MA{window})")
    plt.title(f"{title_prefix} - Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{title_prefix}_reward.png"))
    plt.close()

    # --- Steps (raw + MA)
    plt.figure(figsize=(9, 4))
    plt.plot(episodes, episode_steps, alpha=0.35, label="Steps (raw)")
    ma = moving_average(episode_steps, window)
    if len(ma) > 1:
        plt.plot(np.arange(window, window + len(ma)), ma, label=f"Steps (MA{window})")
    plt.title(f"{title_prefix} - Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{title_prefix}_steps.png"))
    plt.close()

    # --- Penalties (raw + MA)
    plt.figure(figsize=(9, 4))
    plt.plot(episodes, episode_penalties, alpha=0.35, label="Penalties (raw)")
    ma = moving_average(episode_penalties, window)
    if len(ma) > 1:
        plt.plot(np.arange(window, window + len(ma)), ma, label=f"Penalties (MA{window})")
    plt.title(f"{title_prefix} - Penalties (-10 count)")
    plt.xlabel("Episode")
    plt.ylabel("Penalties")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{title_prefix}_penalties.png"))
    plt.close()

    # --- Success rate (block-based)
    succ = np.array([1.0 if s else 0.0 for s in episode_successes], dtype=float)
    n_blocks = math.ceil(len(succ) / window)
    block_rates = []
    for i in range(n_blocks):
        start = i * window
        end = min(start + window, len(succ))
        block_rates.append(float(succ[start:end].mean()))
    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(1, n_blocks + 1), block_rates, marker="o")
    plt.title(f"{title_prefix} - Success Rate (blocks of {window})")
    plt.xlabel("Block #")
    plt.ylabel("Success rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{title_prefix}_success_rate.png"))
    plt.close()

    # --- Epsilon curve (if available)
    if eps_history is not None:
        plt.figure(figsize=(9, 4))
        plt.plot(episodes, eps_history)
        plt.title(f"{title_prefix} - Epsilon (step-based, sampled per episode)")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{title_prefix}_epsilon.png"))
        plt.close()


@torch.no_grad()
def evaluate_model(
    policy_net: torch.nn.Module,
    env_name: str,
    device: torch.device,
    n_episodes: int = 20,
    seed: int = 1000,
    max_steps: int = 200,
) -> Dict[str, float]:
    """
    Evaluate greedy policy (no exploration) for n_episodes.

    Returns mean metrics:
    - eval_mean_reward
    - eval_mean_steps
    - eval_mean_penalties
    - eval_success_rate
    """
    env = gym.make(env_name)

    rewards, steps_list, penalties_list, successes = [], [], [], []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_r = 0.0
        steps = 0
        penalties = 0
        success = False

        for _ in range(max_steps):
            st = torch.tensor([state], dtype=torch.long, device=device)
            action = policy_net(st).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_r += float(reward)
            steps += 1
            if reward == -10:
                penalties += 1
            if terminated and reward == 20:
                success = True

            state = next_state
            if done:
                break

        rewards.append(total_r)
        steps_list.append(steps)
        penalties_list.append(penalties)
        successes.append(1.0 if success else 0.0)

    env.close()

    return {
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_mean_steps": float(np.mean(steps_list)),
        "eval_mean_penalties": float(np.mean(penalties_list)),
        "eval_success_rate": float(np.mean(successes)),
    }


def train(
    env_name: str,
    seed: int,
    num_episodes: int,
    lr: float,
    batch_size: int,
    gamma: float,
    replay_capacity: int,
    learning_starts: int,
    train_every_steps: int,
    target_update_every_steps: int,
    eps_start: float,
    eps_end: float,
    eps_decay_steps: int,
    ddqn: bool,
    eval_interval_episodes: int,
    eval_episodes: int,
    max_steps_per_episode: int = 200,
) -> Tuple[List[float], List[int], List[int], List[bool], List[float], str]:
    """
    Main training loop.

    If ddqn=False:
      Classic DQN target (with target network):
        y = r + gamma * max_a Q_target(s', a)

    If ddqn=True:
      Double DQN target:
        a* = argmax_a Q_policy(s', a)
        y  = r + gamma * Q_target(s', a*)
    """
    set_seed(seed)
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = gym.make(env_name)
    env.action_space.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(n_states=n_states, n_actions=n_actions).to(device)
    target_net = DQN(n_states=n_states, n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=replay_capacity, seed=seed)

    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    episode_penalties: List[int] = []
    episode_successes: List[bool] = []
    eps_history: List[float] = []

    best_eval_reward = -float("inf")
    best_model_path = os.path.join(MODEL_DIR, f"best_{'DDQN' if ddqn else 'DQN'}_seed{seed}.pth")

    global_step = 0

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed + episode)

        total_reward = 0.0
        steps_taken = 0
        penalties = 0
        success = False

        # We log epsilon once per episode (useful for plots)
        # Note: epsilon is computed step-based; we store the value at episode start.
        eps = linear_epsilon_by_step(global_step, eps_start, eps_end, eps_decay_steps)
        eps_history.append(eps)

        for _ in range(max_steps_per_episode):
            global_step += 1

            # Epsilon-greedy action
            eps = linear_epsilon_by_step(global_step, eps_start, eps_end, eps_decay_steps)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    st = torch.tensor([state], dtype=torch.long, device=device)
                    action = policy_net(st).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Metrics
            steps_taken += 1
            total_reward += float(reward)
            if reward == -10:
                penalties += 1
            if terminated and reward == 20:
                success = True

            # Store transition
            replay.push(state, action, float(reward), next_state, done)
            state = next_state

            # Learn (only after a minimum buffer size)
            if len(replay) >= learning_starts and (global_step % train_every_steps == 0):
                batch = replay.sample(batch_size)

                state_batch = torch.tensor(batch.state, dtype=torch.long, device=device)
                action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
                next_state_batch = torch.tensor(batch.next_state, dtype=torch.long, device=device)
                done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

                # Current Q(s,a)
                q_values = policy_net(state_batch).gather(1, action_batch)

                # Build targets
                with torch.no_grad():
                    if not ddqn:
                        # Classic DQN (target net chooses and evaluates)
                        next_q = target_net(next_state_batch).max(dim=1)[0].unsqueeze(1)
                    else:
                        # DDQN: policy selects, target evaluates
                        best_next_actions = policy_net(next_state_batch).argmax(dim=1, keepdim=True)
                        next_q = target_net(next_state_batch).gather(1, best_next_actions)

                    q_targets = reward_batch + gamma * next_q * (1.0 - done_batch)

                loss = F.smooth_l1_loss(q_values, q_targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                optimizer.step()

            # Target network update
            if global_step % target_update_every_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Episode end logs
        episode_rewards.append(total_reward)
        episode_steps.append(steps_taken)
        episode_penalties.append(penalties)
        episode_successes.append(success)

        # Progress print
        if episode % 100 == 0:
            print(f"Seed {seed} | Ep {episode} | Reward {total_reward:.1f} | Eps {eps:.3f} | Success {success}")

        # Periodic evaluation + best checkpoint
        if eval_interval_episodes > 0 and (episode % eval_interval_episodes == 0):
            metrics = evaluate_model(
                policy_net=policy_net,
                env_name=env_name,
                device=device,
                n_episodes=eval_episodes,
                seed=seed + 10_000 + episode,
                max_steps=max_steps_per_episode,
            )
            avg_ret = metrics["eval_mean_reward"]
            print(f"[Eval] Ep {episode}: avg_return={avg_ret:.2f} | success={metrics['eval_success_rate']:.2f}")

            if avg_ret > best_eval_reward:
                best_eval_reward = avg_ret
                torch.save(policy_net.state_dict(), best_model_path)
                print(f"  New best model saved: {best_model_path}")

    env.close()
    return episode_rewards, episode_steps, episode_penalties, episode_successes, eps_history, best_model_path


def main():
    # ============================================================
    # QUICK CONFIG (edit here)
    # ============================================================
    ENV_NAME = "Taxi-v3"
    SEED = 42

    # Training length
    NUM_EPISODES = 6000
    MAX_STEPS_PER_EPISODE = 200

    # DQN / DDQN switch
    DDQN = False   # False = classic DQN, True = DDQN

    # Optim / learning
    LR = 1e-3
    GAMMA = 0.99
    BATCH_SIZE = 128

    # Replay
    REPLAY_CAPACITY = 50_000
    LEARNING_STARTS = 2_000
    TRAIN_EVERY_STEPS = 1

    # Target update
    TARGET_UPDATE_EVERY_STEPS = 2000

    # Exploration (step-based)
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY_STEPS = 150_000  # ~ NUM_EPISODES * avg_steps; good default for Taxi

    # Evaluation
    EVAL_INTERVAL_EPISODES = 200
    EVAL_EPISODES = 50

    # Plot smoothing
    PLOT_WINDOW = 100
    # ============================================================

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Training {'DDQN' if DDQN else 'DQN'} | seed={SEED}")
    ep_rew, ep_steps, ep_pen, ep_succ, eps_hist, best_model_path = train(
        env_name=ENV_NAME,
        seed=SEED,
        num_episodes=NUM_EPISODES,
        lr=LR,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        replay_capacity=REPLAY_CAPACITY,
        learning_starts=LEARNING_STARTS,
        train_every_steps=TRAIN_EVERY_STEPS,
        target_update_every_steps=TARGET_UPDATE_EVERY_STEPS,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay_steps=EPS_DECAY_STEPS,
        ddqn=DDQN,
        eval_interval_episodes=EVAL_INTERVAL_EPISODES,
        eval_episodes=EVAL_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
    )

    title = f"{'DDQN' if DDQN else 'DQN'}_S{SEED}_lr{LR}_bs{BATCH_SIZE}_tgt{TARGET_UPDATE_EVERY_STEPS}"
    plot_curves(ep_rew, ep_steps, ep_pen, ep_succ, eps_history=eps_hist, title_prefix=title, window=PLOT_WINDOW)

    print("\nDone.")
    print(f"Best model saved at: {best_model_path}")
    print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()