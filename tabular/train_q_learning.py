from typing import Dict, Tuple
import os
import sys
import os
import numpy as np
import gymnasium as gym

# Add project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.plotting import save_single_run_curve
from config import QLearningConfig, GlobalConfig
from utils.logging import EpisodeLog


# epsilon decay linearly until decay_episodes, when it stops
def linear_epsilon(episode: int, eps_start: float, eps_end: float, decay_episodes: int) -> float:
    if decay_episodes <= 0:
        return eps_end
    frac = min(1.0, episode / decay_episodes)
    return eps_start + frac * (eps_end - eps_start)


# Main Training loop 
def train_q_learning(global_cfg: GlobalConfig, q_cfg: QLearningConfig, seed: int) -> Tuple[EpisodeLog, Dict[str, np.ndarray]]:

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

    return log, out, Q

if __name__ == "__main__":

    SEED = 42
    EPISODES = 3000
    DISABLE_EVAL = True
    OUTDIR = "results/q_learning_single"

    global_cfg = GlobalConfig()
    q_cfg = QLearningConfig()

    # Overrides
    q_cfg.episodes = EPISODES
    if DISABLE_EVAL:
        global_cfg.eval_every_episodes = 999999999

    # Train
    log, metrics,Q = train_q_learning(global_cfg, q_cfg, seed=SEED)
    np.save(os.path.join(OUTDIR, f"Q_seed{SEED}.npy"), Q)

    # Print summary
    print("\n[Q-LEARNING] Finished.")
    print(f"Seed: {SEED}")
    print(f"Episodes: {q_cfg.episodes}")



    # Save  plots
    os.makedirs(OUTDIR, exist_ok=True)
    ep = metrics["episode"]

    # Reward
    save_single_run_curve(
        ep, metrics["episode_reward"],
        title="Q-LEARNING (single run) - Reward",
        xlabel="Episode", ylabel="Reward",
        outpath=os.path.join(OUTDIR, f"reward_seed{SEED}.png"),
        rolling_window=global_cfg.rolling_window,
    )
    
    # Penalties
    save_single_run_curve(
        ep, metrics["penalties"],
        title="Q-LEARNING (single run) - Penalties (-10 count)",
        xlabel="Episode", ylabel="Penalties",
        outpath=os.path.join(OUTDIR, f"penalties_seed{SEED}.png"),
        rolling_window=global_cfg.rolling_window,
    )
    
    # Steps
    save_single_run_curve(
    ep, metrics["steps"],
    title="Q-LEARNING (single run) - Steps per episode",
    xlabel="Episode", ylabel="Steps",
    outpath=os.path.join(OUTDIR, f"steps_seed{SEED}.png"),
    rolling_window=global_cfg.rolling_window,
    )

    # Success
    
    save_single_run_curve(
    ep, metrics["success"],
    title="Q-LEARNING (single run) - Success (rolling mean)",
    xlabel="Episode", ylabel="Success rate",
    outpath=os.path.join(OUTDIR, f"success_seed{SEED}.png"),
    rolling_window=global_cfg.rolling_window,
    )

    print(f"Plots saved in: {OUTDIR}\n")