from __future__ import annotations
from typing import Dict, Tuple
import os
import sys
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

from utils import rolling_mean
from utils import EpisodeLog
from config import GlobalConfig, DQNConfig
from utils import save_rolling_means
from utils import set_global_seeds
from utils import linear_epsilon
# DQN module
from deep.DQN import DQN
from deep.DQN import ReplayBuffer


# Main training loop for DQN
def train_dqn(
    global_cfg: GlobalConfig,
    dqn_cfg: DQNConfig,
    seed: int,
    outdir: str | None = None,
) -> Tuple[EpisodeLog, Dict[str, np.ndarray], torch.nn.Module]:
    
    # seeding
    set_global_seeds(seed)

    # create environment and get dimensions
    env = gym.make(global_cfg.env_id)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    env.action_space.seed(seed)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize policy network
    policy_net = DQN(
        n_states=n_states,
        n_actions=n_actions,
        emb_dim=dqn_cfg.embedding_dim,
        hidden=dqn_cfg.hidden_dim,
    ).to(device)

    # optimizer and replay buffer 
    optimizer = optim.Adam(policy_net.parameters(), lr=dqn_cfg.lr)
    replay = ReplayBuffer(capacity=dqn_cfg.replay_capacity, seed=seed)

    log = EpisodeLog()
    global_step = 0

    progress_every = 500 # print training rolling averages every n episodes

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    # Main training loop
    for ep in range(1, dqn_cfg.episodes + 1):

        # compute epsilon for this episode
        eps = linear_epsilon(ep, dqn_cfg.eps_start, dqn_cfg.eps_end, dqn_cfg.eps_decay_episodes)

        # reset environment at the start of each episode
        obs, _ = env.reset(seed=seed + ep)

        ep_reward = 0.0
        ep_steps = 0
        ep_penalties = 0
        success = False

        # episode loop
        for _ in range(dqn_cfg.max_steps_per_episode):
            global_step += 1

            # epsilon-greedy action selection
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

            # metrics and termination conditions
            ep_reward += float(reward)
            ep_steps += 1
            if float(reward) == -10.0:
                ep_penalties += 1
            if terminated and float(reward) == 20.0:
                success = True

            # learning step (if enough data (>= 2000) and every 2 episodes)
            if len(replay) >= dqn_cfg.learning_starts and (global_step % dqn_cfg.train_every_episodes == 0):

                # sample a batch of transitions from the replay buffer
                batch = replay.sample(dqn_cfg.batch_size)

                # convert batch to tensors
                s = torch.tensor(batch.state, dtype=torch.long, device=device)
                a = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
                r = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
                s2 = torch.tensor(batch.next_state, dtype=torch.long, device=device)
                d = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

                # policy_net(s) returns Q-values for all actions: shape (B, A).
                # gather(1, a) selects for each transition i the Q-value of the action taken a[i]
                # producing q_sa with shape (B, 1) = Q(s_i, a_i)
                q_sa = policy_net(s).gather(1, a)
                

                # Build the TD target without tracking gradients.
                # Compute max_a' Q(s', a') from the network outputs,
                # then compute the target as r + gamma * max_a' Q(s', a') * (1 - done)
                with torch.no_grad():
                    max_next_q = policy_net(s2).max(dim=1)[0].unsqueeze(1)
                    target = r + dqn_cfg.gamma * max_next_q * (1.0 - d)

                # Huber loss 
                loss = F.smooth_l1_loss(q_sa, target)

                # backpropagation step
                optimizer.zero_grad()
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=dqn_cfg.grad_clip_norm)
                optimizer.step()

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

        # print progress every n episodes 
        if ep % progress_every == 0:
            d = log.as_dict_of_lists()
            w = progress_every

            avg_reward = float(np.mean(d["episode_reward"][-w:]))
            avg_steps = float(np.mean(d["steps"][-w:]))
            avg_penalties = float(np.mean(d["penalties"][-w:]))
            avg_success = float(np.mean(d["success"][-w:])) * 100.0

            print(f"[Train] ep={ep}/{dqn_cfg.episodes} eps={eps:.3f} | R={avg_reward:.2f} S={avg_steps:.1f} P={avg_penalties:.2f} Succ={avg_success:.1f}%")

    env.close()

    # arrays for plotting 
    d = log.as_dict_of_lists()
    out = {
        "episode": np.array(d["episode"], dtype=int),
        "episode_reward": np.array(d["episode_reward"], dtype=float),
        "steps": np.array(d["steps"], dtype=float),
        "penalties": np.array(d["penalties"], dtype=float),
        "success": np.array(d["success"], dtype=float),
        "epsilon": np.array(d["epsilon"], dtype=float),
    }

    # Save the final model (at the last episode)
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

    # Train DQN
    log, metrics, policy_net = train_dqn(global_cfg, dqn_cfg, seed=SEED, outdir=OUTDIR)

    print("\n[DQN] Finished.")

    # Plotting
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


    # full range plots
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


   # zoomed-in plots (1000–3000)
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
    
    save_rolling_means(
    outdir=OUTDIR,
    episode=ep,
    reward=metrics["episode_reward"],
    steps=metrics["steps"],
    penalties=metrics["penalties"],
    success=metrics["success"],
    w=w,
    tag="dqn_train",
    )
    
    print(f"Plots saved in: {OUTDIR}\n")
