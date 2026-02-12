from __future__ import annotations
from typing import Dict, Tuple
import os
import sys
import random
from collections import deque, namedtuple

import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Add project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.plotting import rolling_mean
from utils.logging import EpisodeLog
from config import GlobalConfig, DQNConfig

# DQN module
from deep.DQN import DQN

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_epsilon_by_step(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return eps_end
    frac = min(1.0, step / decay_steps)
    return eps_start + frac * (eps_end - eps_start)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        self.buffer.append(Transition(s, a, r, s2, done))

    def sample(self, batch_size: int) -> Transition:
        batch = self.rng.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))


def train_dqn(
    global_cfg: GlobalConfig,
    dqn_cfg: DQNConfig,
    seed: int,
    outdir: str | None = None,
) -> Tuple[EpisodeLog, Dict[str, np.ndarray], torch.nn.Module]:
    set_seed(seed)

    env = gym.make(global_cfg.env_id)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    env.action_space.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(
        n_states=n_states,
        n_actions=n_actions,
        emb_dim=dqn_cfg.embedding_dim,
        hidden=dqn_cfg.hidden_dim,
    ).to(device)

    target_net = DQN(
        n_states=n_states,
        n_actions=n_actions,
        emb_dim=dqn_cfg.embedding_dim,
        hidden=dqn_cfg.hidden_dim,
    ).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=dqn_cfg.lr)
    replay = ReplayBuffer(capacity=dqn_cfg.replay_capacity, seed=seed)

    log = EpisodeLog()
    global_step = 0

    # ---- progress settings ----
    progress_every = 100  # print training rolling averages every 100 episodes

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    for ep in range(1, dqn_cfg.episodes + 1):
        obs, _ = env.reset(seed=seed + ep)

        ep_reward = 0.0
        ep_steps = 0
        ep_penalties = 0
        success = False

        # epsilon logged as the FINAL epsilon reached in the episode
        eps = linear_epsilon_by_step(global_step, dqn_cfg.eps_start, dqn_cfg.eps_end, dqn_cfg.eps_decay_steps)

        for _ in range(dqn_cfg.max_steps_per_episode):
            global_step += 1

            # epsilon-greedy (step-based schedule)
            eps = linear_epsilon_by_step(global_step, dqn_cfg.eps_start, dqn_cfg.eps_end, dqn_cfg.eps_decay_steps)
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    st = torch.tensor([obs], dtype=torch.long, device=device)
                    action = int(policy_net(st).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store transition
            replay.push(obs, action, float(reward), next_obs, bool(done))

            # metrics
            ep_reward += float(reward)
            ep_steps += 1
            if float(reward) == -10.0:
                ep_penalties += 1
            if terminated and float(reward) == 20.0:
                success = True

            # learn
            if len(replay) >= dqn_cfg.learning_starts and (global_step % dqn_cfg.train_every_steps == 0):
                batch = replay.sample(dqn_cfg.batch_size)

                s = torch.tensor(batch.state, dtype=torch.long, device=device)
                a = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
                r = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
                s2 = torch.tensor(batch.next_state, dtype=torch.long, device=device)
                d = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

                q_sa = policy_net(s).gather(1, a)

                with torch.no_grad():
                    max_next_q = target_net(s2).max(dim=1)[0].unsqueeze(1)
                    target = r + dqn_cfg.gamma * max_next_q * (1.0 - d)

                loss = F.smooth_l1_loss(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=dqn_cfg.grad_clip_norm)
                optimizer.step()

            # target network update
            if global_step % dqn_cfg.target_update_every_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

            obs = next_obs

        # log episode
        log.add(
            algorithm="DQN",
            seed=seed,
            episode=ep,
            episode_reward=ep_reward,
            steps=ep_steps,
            penalties=ep_penalties,
            success=1.0 if success else 0.0,
            epsilon=eps,
        )

        # ---- Progress print every 100 episodes: print ONLY training rolling averages of the 4 metrics ----
        if ep % progress_every == 0:
            d = log.as_dict_of_lists()
            w = progress_every

            avg_reward = float(np.mean(d["episode_reward"][-w:]))
            avg_steps = float(np.mean(d["steps"][-w:]))
            avg_penalties = float(np.mean(d["penalties"][-w:]))
            avg_success = float(np.mean(d["success"][-w:])) * 100.0

            print(
                f"[Train] ep={ep:5d}/{dqn_cfg.episodes} | step={global_step:8d} | eps={eps:6.3f} | "
                f"avgReward({w})={avg_reward:7.2f} | avgSteps({w})={avg_steps:6.1f} | "
                f"avgPen({w})={avg_penalties:6.2f} | avgSucc({w})={avg_success:5.1f}%"
            )

    env.close()

    # arrays for plotting (same schema as Q-learning)
    d = log.as_dict_of_lists()
    out = {
        "episode": np.array(d["episode"], dtype=int),
        "episode_reward": np.array(d["episode_reward"], dtype=float),
        "steps": np.array(d["steps"], dtype=float),
        "penalties": np.array(d["penalties"], dtype=float),
        "success": np.array(d["success"], dtype=float),
        "epsilon": np.array(d["epsilon"], dtype=float),
    }

    # Save ONLY the final model here (last episode)
    if outdir is not None:
        final_path = os.path.join(outdir, f"dqn_seed{seed}.pth")
        torch.save(policy_net.state_dict(), final_path)
        print(f"[Checkpoint] Final model saved: {final_path}")

    return log, out, policy_net


if __name__ == "__main__":
    SEED = 42
    EPISODES = 3000
    OUTDIR = "results/train_dqn"

    global_cfg = GlobalConfig()
    dqn_cfg = DQNConfig()

    # overrides
    dqn_cfg.episodes = EPISODES

    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("device =", "cuda" if torch.cuda.is_available() else "cpu")

    # train (this will save the final model at the end)
    log, metrics, policy_net = train_dqn(global_cfg, dqn_cfg, seed=SEED, outdir=OUTDIR)

    print("\n[DQN] Finished.")
    print(f"Seed: {SEED}")
    print(f"Episodes: {dqn_cfg.episodes}")
    print(f"Artifacts dir: {OUTDIR}")

    # -------------------------
# Setup for plots (raw + rolling) + zoom mask
# -------------------------
    ep = metrics["episode"]
    w = global_cfg.rolling_window

    # raw
    reward_raw = metrics["episode_reward"]
    steps_raw  = metrics["steps"]
    pen_raw    = metrics["penalties"]
    succ_raw   = metrics["success"]

    # rolling
    reward_sm = rolling_mean(reward_raw, w)
    steps_sm  = rolling_mean(steps_raw,  w)
    pen_sm    = rolling_mean(pen_raw,    w)
    succ_sm   = rolling_mean(succ_raw,   w)

    # zoom mask (1000–3000)
    m = (ep >= 1000) & (ep <= 3000)


    # -------------------------
    # FULL (0–3000) — raw + rolling mean
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(ep, reward_raw, linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep, reward_sm,  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("DQN - Reward per episode (0–3000)")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.xlim(0, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, "reward_full.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep, pen_raw, linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep, pen_sm,  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("DQN - Penalties per episode (0–3000)")
    plt.xlabel("Episode"); plt.ylabel("Penalties")
    plt.xlim(0, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, "penalties_full.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep, steps_raw, linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep, steps_sm,  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("DQN - Steps per episode (0–3000)")
    plt.xlabel("Episode"); plt.ylabel("Steps")
    plt.xlim(0, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, "steps_full.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep, succ_raw, linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep, succ_sm,  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("DQN - Success per episode (0–3000)")
    plt.xlabel("Episode"); plt.ylabel("Success rate")
    plt.xlim(0, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, "success_full.png"), dpi=150)
    plt.close()


    # -------------------------
    # ZOOM (1000–3000) — raw + rolling mean (4 plot separati)
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(ep[m], reward_raw[m], linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep[m], reward_sm[m],  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("DQN - Reward per episode (1000–3000)")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.xlim(1000, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, "reward_zoom.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep[m], pen_raw[m], linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep[m], pen_sm[m],  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("DQN - Penalties per episode (1000–3000)")
    plt.xlabel("Episode"); plt.ylabel("Penalties")
    plt.xlim(1000, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, "penalties_zoom.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep[m], steps_raw[m], linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep[m], steps_sm[m],  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("DQN - Steps per episode (1000–3000)")
    plt.xlabel("Episode"); plt.ylabel("Steps")
    plt.xlim(1000, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, "steps_zoom.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep[m], succ_raw[m], linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(ep[m], succ_sm[m],  linewidth=2.0, label=f"rolling mean (w={w})")
    plt.title("DQN - Success per episode (1000–3000)")
    plt.xlabel("Episode"); plt.ylabel("Success rate")
    plt.xlim(1000, 3000); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTDIR, "success_zoom.png"), dpi=150)
    plt.close()
    print(f"Plots saved in: {OUTDIR}\n")
