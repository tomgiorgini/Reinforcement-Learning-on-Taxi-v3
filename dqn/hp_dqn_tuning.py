from __future__ import annotations

import os
import sys
import time
import copy
from itertools import product

import numpy as np
import torch

# Add project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import GlobalConfig, DQNConfig
from train_dqn import train_dqn
from test_dqn import run_greedy_dqn
from utils import score


def main() -> None:
    SEED = 42
    TRAIN_EPISODES = 1500
    EVAL_EPISODES = 100
    OUTDIR = "results/dqn_tuning"
    os.makedirs(OUTDIR, exist_ok=True)

    global_cfg = GlobalConfig()
    base_cfg = DQNConfig()
    base_cfg.episodes = TRAIN_EPISODES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # Search space
    lrs = [5e-4, 1e-4, 5e-3] # learning rates 
    batch_sizes = [128,256] # batch sizes
    gammas = [0.97, 0.99] # discount factors
    train_every_steps = [1, 2, 4] # train every n steps

    total = len(lrs) * len(batch_sizes) * len(gammas) * len(train_every_steps)
    print(f"Total runs: {total}")

    csv_path = os.path.join(OUTDIR, "tuning_results.csv")
    best_score = float("-inf")
    best_run = None

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "run_id,lr,batch_size,train_every_steps,gamma,"
            "eval_success,eval_reward,eval_steps,eval_penalties,score,train_seconds\n"
        )

        run_id = 0
        # Iterate over all combinations of hyperparameters
        for lr, bs, tst, gamma in product(lrs, batch_sizes, train_every_steps, gammas):
            run_id += 1
            cfg = copy.deepcopy(base_cfg)

            # Update config with current hyperparameters
            cfg.lr = lr
            cfg.batch_size = bs
            cfg.train_every_steps= tst
            cfg.gamma = gamma

            max_steps = getattr(cfg, "max_steps_per_episode", 200)
            total_steps = int(TRAIN_EPISODES * max_steps)

            # linear decay over the whole run
            cfg.eps_start = 1.0
            cfg.eps_end = 0.10
            cfg.eps_decay_steps = total_steps

            t0 = time.time()

            # Train
            _, _, policy_net = train_dqn(global_cfg, cfg, seed=SEED, outdir=None)

            # Greedy eval
            rewards, steps, penalties, success = run_greedy_dqn(
                env_id=global_cfg.env_id,
                policy_net=policy_net,
                device=device,
                seed=SEED + 50_000,
                episodes=EVAL_EPISODES,
                max_steps=max_steps,
            )

            eval_reward = float(np.mean(rewards))
            eval_steps = float(np.mean(steps))
            eval_penalties = float(np.mean(penalties))
            eval_success = float(np.mean(success))

            # Compute score for this run
            s = score(eval_success, eval_reward, eval_steps, eval_penalties)
            dt = time.time() - t0

            print(
                f"[{run_id:03d}/{total}] lr={lr:g} bs={bs} tst={tst} gamma={gamma} | "
                f"succ={eval_success:.3f} r={eval_reward:.2f} steps={eval_steps:.1f} pen={eval_penalties:.2f} | "
                f"score={s:.2f} | {dt:.1f}s"
            )

            f.write(
                f"{run_id},{lr},{bs},{tst},{gamma},"
                f"{eval_success},{eval_reward},{eval_steps},{eval_penalties},{s},{dt}\n"
            )
            f.flush()

            # Check if this is the best run so far
            if s > best_score:
                best_score = s
                best_run = {
                    "lr": lr,
                    "batch_size": bs,
                    "train_every_steps": tst,
                    "gamma": gamma,
                    "eval_success": eval_success,
                    "eval_reward": eval_reward,
                    "eval_steps": eval_steps,
                    "eval_penalties": eval_penalties,
                }
                with open(os.path.join(OUTDIR, "best_run.txt"), "w", encoding="utf-8") as g:
                    g.write(str(best_run) + "\n")

    print("\n BEST ")
    print("Best score:", best_score)
    print("Best run:", best_run)
    print("CSV saved to:", csv_path)


if __name__ == "__main__":
    main()
