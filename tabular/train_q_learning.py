from __future__ import annotations

import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Add project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import QLearningConfig, GlobalConfig
from utils.logging import EpisodeLog
from utils.plotting import rolling_mean

def linear_epsilon(episode: int, eps_start: float, eps_end: float, decay_episodes: int) -> float:
    if decay_episodes <= 0:
        return eps_end
    frac = min(1.0, episode / decay_episodes)
    return eps_start + frac * (eps_end - eps_start)


def train_q_learning(global_cfg: GlobalConfig, q_cfg: QLearningConfig, seed: int):
    env = gym.make(global_cfg.env_id)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    log = EpisodeLog()

    for ep in range(1, q_cfg.episodes + 1):
        eps = linear_epsilon(ep, q_cfg.eps_start, q_cfg.eps_end, q_cfg.eps_decay_episodes)
        obs, _ = env.reset(seed=seed + ep)

        ep_reward = 0.0
        ep_steps = 0
        ep_penalties = 0
        success = False

        for _ in range(q_cfg.max_steps_per_episode):
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[obs]))

            next_obs, reward, terminated, truncated, _ = env.step(action)

            td_target = reward + q_cfg.gamma * np.max(Q[next_obs])
            td_error = td_target - Q[obs, action]
            Q[obs, action] += q_cfg.alpha * td_error

            ep_reward += float(reward)
            ep_steps += 1
            if float(reward) == -10.0:
                ep_penalties += 1

            if terminated:
                if float(reward) == 20.0:
                    success = True
                break
            if truncated:
                break

            obs = next_obs

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

    d = log.as_dict_of_lists()
    metrics = {
        "episode": np.array(d["episode"], dtype=int),
        "episode_reward": np.array(d["episode_reward"], dtype=float),
        "steps": np.array(d["steps"], dtype=float),
        "penalties": np.array(d["penalties"], dtype=float),
        "success": np.array(d["success"], dtype=float),
        "epsilon": np.array(d["epsilon"], dtype=float),
    }
    return log, metrics, Q


if __name__ == "__main__":
    SEED = 42
    EPISODES = 3000
    OUTDIR = "results/train_q_learning"
    os.makedirs(OUTDIR, exist_ok=True)

    global_cfg = GlobalConfig()
    q_cfg = QLearningConfig()
    q_cfg.episodes = EPISODES

    log, metrics, Q = train_q_learning(global_cfg, q_cfg, seed=SEED)
    np.save(os.path.join(OUTDIR, f"Q_seed{SEED}.npy"), Q)

    ep = metrics["episode"]
    w = global_cfg.rolling_window

    # raw + rolling
    reward_raw = metrics["episode_reward"]
    steps_raw  = metrics["steps"]
    pen_raw    = metrics["penalties"]
    succ_raw   = metrics["success"]

    reward_sm = rolling_mean(reward_raw, w)
    steps_sm  = rolling_mean(steps_raw,  w)
    pen_sm    = rolling_mean(pen_raw,    w)
    succ_sm   = rolling_mean(succ_raw,   w)

    # mask zoom
    m = (ep >= 1000) & (ep <= 3000)


    plt.figure(figsize=(6, 4))
    plt.plot(ep, reward_raw, linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep, reward_sm,  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("Q-LEARNING - Reward per episode (0–3000)")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.xlim(0, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, f"reward_full.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep, steps_raw, linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep, steps_sm,  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("Q-LEARNING - Steps per episode (0–3000)")
    plt.xlabel("Episode"); plt.ylabel("Steps")
    plt.xlim(0, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, f"steps_full.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep, pen_raw, linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep, pen_sm,  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("Q-LEARNING - Penalties per episode (0–3000)")
    plt.xlabel("Episode"); plt.ylabel("Penalties")
    plt.xlim(0, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, f"penalties_full.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep, succ_raw, linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep, succ_sm,  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("Q-LEARNING - Success per episode (0–3000)")
    plt.xlabel("Episode"); plt.ylabel("Success")
    plt.xlim(0, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, f"success_full.png"), dpi=150)
    plt.close()

    # -------------------------
    # ZOOM (1000–3000) — 4 plot separati
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(ep[m], reward_raw[m], linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep[m], reward_sm[m],  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("Q-LEARNING - Reward per episode (1000–3000)")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.xlim(1000, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, f"reward_zoom.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep[m], steps_raw[m], linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep[m], steps_sm[m],  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("Q-LEARNING - Steps per episode (1000–3000)")
    plt.xlabel("Episode"); plt.ylabel("Steps")
    plt.xlim(1000, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, f"steps_zoom.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep[m], pen_raw[m], linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep[m], pen_sm[m],  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("Q-LEARNING - Penalties per episode (1000–3000)")
    plt.xlabel("Episode"); plt.ylabel("Penalties")
    plt.xlim(1000, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, f"penalties_zoom.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep[m], succ_raw[m], linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep[m], succ_sm[m],  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("Q-LEARNING - Success per episode (1000–3000)")
    plt.xlabel("Episode"); plt.ylabel("Success")
    plt.xlim(1000, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, f"success_zoom.png"), dpi=150)
    plt.close()

    print(f"[Q-LEARNING] Done. Plots saved in: {OUTDIR}")