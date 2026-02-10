from typing import Dict, Tuple
import os
import sys

# Add project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import gymnasium as gym

from config import QLearningConfig, GlobalConfig
from utils.logging import EpisodeLog

def linear_epsilon(episode: int, eps_start: float, eps_end: float, decay_episodes: int) -> float:
    """
    Linear epsilon decay:
    - stays at eps_start initially
    - decreases linearly for `decay_episodes`
    - bottoms out at eps_end
    """
    if decay_episodes <= 0:
        return eps_end
    frac = min(1.0, episode / decay_episodes)
    return eps_start + frac * (eps_end - eps_start)


def evaluate_q_table(env_id: str, Q: np.ndarray, seed: int, episodes: int, max_steps: int) -> Dict[str, float]:
    """
    Evaluate a greedy policy derived from the Q-table.

    We compute:
    - mean_reward
    - mean_steps
    - success_rate
    - mean_penalties

    Success definition (robust):
    - The Taxi environment terminates when a correct drop-off happens.
    - We also check the terminal reward == +20 as confirmation.
    """
    env = gym.make(env_id)
    rewards, steps_list, penalties_list, successes = [], [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        ep_reward = 0
        ep_steps = 0
        ep_penalties = 0
        success = False

        for _ in range(max_steps):
            action = int(np.argmax(Q[obs]))
            obs, reward, terminated, truncated, _ = env.step(action)

            ep_reward += reward
            ep_steps += 1
            if reward == -10:
                ep_penalties += 1

            if terminated:
                # terminated usually means successful drop-off in Taxi.
                # We also verify reward == 20 for clarity.
                if reward == 20:
                    success = True
                done = True
            if truncated:
                done = True

            if done:
                break

        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        penalties_list.append(ep_penalties)
        successes.append(1.0 if success else 0.0)

    env.close()

    return {
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_mean_steps": float(np.mean(steps_list)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_mean_penalties": float(np.mean(penalties_list)),
    }


def train_q_learning(global_cfg: GlobalConfig, q_cfg: QLearningConfig, seed: int) -> Tuple[EpisodeLog, Dict[str, np.ndarray]]:
    """
    Train tabular Q-learning for a single seed.

    Returns:
    - EpisodeLog (raw log entries)
    - dict of numpy arrays for key metrics (for plotting / aggregation)
    """
    env = gym.make(global_cfg.env_id)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    log = EpisodeLog()

    for ep in range(1, q_cfg.episodes + 1):
        eps = linear_epsilon(ep, q_cfg.eps_start, q_cfg.eps_end, q_cfg.eps_decay_episodes)

        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0
        ep_steps = 0
        ep_penalties = 0
        success = False

        for _ in range(q_cfg.max_steps_per_episode):
            # Epsilon-greedy action selection
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[obs]))

            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Standard tabular Q-learning update
            td_target = reward + q_cfg.gamma * np.max(Q[next_obs])
            td_error = td_target - Q[obs, action]
            Q[obs, action] += q_cfg.alpha * td_error

            # Metrics
            ep_reward += reward
            ep_steps += 1
            if reward == -10:
                ep_penalties += 1

            # Terminal logic
            if terminated:
                if reward == 20:
                    success = True
                break
            if truncated:
                break

            obs = next_obs

        # Periodic evaluation (greedy)
        eval_metrics = {}
        if ep % global_cfg.eval_every_episodes == 0:
            eval_metrics = evaluate_q_table(
                env_id=global_cfg.env_id,
                Q=Q,
                seed=seed,
                episodes=global_cfg.eval_episodes,
                max_steps=q_cfg.max_steps_per_episode
            )

        # Log episode
        log.add(
            algorithm="Q_LEARNING",
            seed=seed,
            episode=ep,
            episode_reward=ep_reward,
            steps=ep_steps,
            penalties=ep_penalties,
            success=1.0 if success else 0.0,
            epsilon=eps,
            **eval_metrics
        )

    env.close()

    # Convert to arrays for plotting
    d = log.as_dict_of_lists()
    out = {
        "episode": np.array(d["episode"], dtype=int),
        "episode_reward": np.array(d["episode_reward"], dtype=float),
        "steps": np.array(d["steps"], dtype=float),
        "penalties": np.array(d["penalties"], dtype=float),
        "success": np.array(d["success"], dtype=float),
        "epsilon": np.array(d["epsilon"], dtype=float),
    }

    # Optional evaluation arrays (may be missing for non-eval episodes)
    # We fill missing with NaN to preserve alignment.
    for k in ["eval_mean_reward", "eval_mean_steps", "eval_success_rate", "eval_mean_penalties"]:
        if k in d:
            arr = np.array([v if v is not None else np.nan for v in d[k]], dtype=float)
            out[k] = arr

    return log, out, Q

if __name__ == "__main__":
    # ============================================================
    # QUICK CONFIG (edit here, then press Run in VSCode)
    # ============================================================
    SEED = 42

    # If you want a fast test, lower EPISODES (e.g., 300)
    EPISODES = 2500

    # Set to True to disable periodic evaluation (faster)
    DISABLE_EVAL = False

    # Where to save quick single-run diagnostic plots
    OUTDIR = "results/q_learning_single"
    # ============================================================

    import os
    from utils.plotting import save_single_run_curve

    global_cfg = GlobalConfig()
    q_cfg = QLearningConfig()

    # Apply overrides
    q_cfg.episodes = EPISODES
    if DISABLE_EVAL:
        global_cfg.eval_every_episodes = 10**9  # effectively disables evaluation

    # Train
    log, metrics,Q = train_q_learning(global_cfg, q_cfg, seed=SEED)
    np.save(os.path.join(OUTDIR, f"Q_seed{SEED}.npy"), Q)

    # Print summary
    last = -1
    print("\n[Q-LEARNING] Finished.")
    print(f"Seed: {SEED}")
    print(f"Episodes: {q_cfg.episodes}")

    # Save a couple of diagnostic plots (single run)
    os.makedirs(OUTDIR, exist_ok=True)
    ep = metrics["episode"]

    save_single_run_curve(
        ep, metrics["episode_reward"],
        title="Q-LEARNING (single run) - Reward",
        xlabel="Episode", ylabel="Reward",
        outpath=os.path.join(OUTDIR, f"reward_seed{SEED}.png"),
        rolling_window=global_cfg.rolling_window,
    )

    save_single_run_curve(
        ep, metrics["penalties"],
        title="Q-LEARNING (single run) - Penalties (-10 count)",
        xlabel="Episode", ylabel="Penalties",
        outpath=os.path.join(OUTDIR, f"penalties_seed{SEED}.png"),
        rolling_window=global_cfg.rolling_window,
    )

    save_single_run_curve(
    ep, metrics["steps"],
    title="Q-LEARNING (single run) - Steps per episode",
    xlabel="Episode", ylabel="Steps",
    outpath=os.path.join(OUTDIR, f"steps_seed{SEED}.png"),
    rolling_window=global_cfg.rolling_window,
    )

    save_single_run_curve(
    ep, metrics["success"],
    title="Q-LEARNING (single run) - Success (rolling mean)",
    xlabel="Episode", ylabel="Success rate",
    outpath=os.path.join(OUTDIR, f"success_seed{SEED}.png"),
    rolling_window=global_cfg.rolling_window,
    )

    print(f"Plots saved in: {OUTDIR}\n")