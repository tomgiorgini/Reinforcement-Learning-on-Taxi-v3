from __future__ import annotations

import os
import sys
import time
from itertools import product
from typing import Any, Dict, Tuple

import numpy as np
import torch

# Add project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import GlobalConfig, DQNConfig
from train_dqn import train_dqn
from test_dqn import run_greedy_dqn


def score(success: float, reward: float, steps: float, penalties: float) -> float:
    return 200.0 * success + 1.0 * reward - 0.25 * steps - 2.0 * penalties


def clone_cfg(base: DQNConfig) -> DQNConfig:
    cfg = DQNConfig()
    if hasattr(base, "__dict__") and hasattr(cfg, "__dict__"):
        cfg.__dict__.update(base.__dict__)
    return cfg


def set_attr_if_exists(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def main() -> None:
    SEED = 42
    TRAIN_EPISODES = 1000
    EVAL_EPISODES = 100
    OUTDIR = "results/dqn_tuning_1000"
    os.makedirs(OUTDIR, exist_ok=True)

    global_cfg = GlobalConfig()
    base_cfg = DQNConfig()
    set_attr_if_exists(base_cfg, "episodes", TRAIN_EPISODES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("device =", device)

    # Hyperparameter search space
    lrs = [1e-3, 1e-4, 1e-5]
    batch_sizes = [64, 128]
    target_updates = [1000, 2000, 3000]
    decay_modes = ["linear_full", "linear_half"]

    total = len(lrs) * len(batch_sizes) * len(target_updates) * len(decay_modes)
    print(f"Total runs: {total}")

    csv_path = os.path.join(OUTDIR, "tuning_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "run_id,lr,batch_size,target_update_every_steps,decay_mode,"
            "eval_success,eval_reward,eval_steps,eval_penalties,score,train_seconds\n"
        )

    best: Tuple[float, Dict[str, Any]] | None = None

    run_id = 0
    for lr, bs, tgt, decay_mode in product(lrs, batch_sizes, target_updates, decay_modes):
        run_id += 1
        cfg = clone_cfg(base_cfg)

        # set core params
        set_attr_if_exists(cfg, "lr", lr)
        set_attr_if_exists(cfg, "batch_size", bs)
        set_attr_if_exists(cfg, "target_update_every_steps", tgt)

        # compute decay steps based on total training steps
        max_steps = getattr(cfg, "max_steps_per_episode", 200)
        total_steps = int(TRAIN_EPISODES * max_steps)

        if decay_mode == "fixed_0.1":
            # make epsilon constant 0.1
            set_attr_if_exists(cfg, "eps_start", 0.1)
            set_attr_if_exists(cfg, "eps_end", 0.1)
            set_attr_if_exists(cfg, "eps_decay_steps", 1)
        elif decay_mode == "linear_full":
            # linear decay over the whole run
            set_attr_if_exists(cfg, "eps_start", 1.0)
            set_attr_if_exists(cfg, "eps_end", 0.05)
            set_attr_if_exists(cfg, "eps_decay_steps", total_steps)
        elif decay_mode == "linear_half":
            # reaches eps_end earlier
            set_attr_if_exists(cfg, "eps_start", 1.0)
            set_attr_if_exists(cfg, "eps_end", 0.05)
            set_attr_if_exists(cfg, "eps_decay_steps", max(1, total_steps // 2))

        t0 = time.time()

        # train
        _, _, policy_net = train_dqn(global_cfg, cfg, seed=SEED, outdir=None)

        # greedy eval
        rewards, steps, penalties, success = run_greedy_dqn(
            env_id=global_cfg.env_id,
            policy_net=policy_net,
            device=device,
            seed=SEED,
            episodes=EVAL_EPISODES,
            max_steps=getattr(cfg, "max_steps_per_episode", 200),
        )

        eval_reward = float(np.mean(rewards))
        eval_steps = float(np.mean(steps))
        eval_penalties = float(np.mean(penalties))
        eval_success = float(np.mean(success))

        s = score(eval_success, eval_reward, eval_steps, eval_penalties)
        dt = time.time() - t0

        print(
            f"[{run_id:03d}/{total}] "
            f"lr={lr:g} bs={bs} tgt={tgt} decay={decay_mode} | "
            f"succ={eval_success:.3f} r={eval_reward:.2f} steps={eval_steps:.1f} pen={eval_penalties:.2f} | "
            f"score={s:.2f} | {dt:.1f}s"
        )

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{run_id},{lr},{bs},{tgt},{decay_mode},"
                f"{eval_success},{eval_reward},{eval_steps},{eval_penalties},{s},{dt}\n"
            )

        if best is None or s > best[0]:
            best = (
                s,
                {
                    "lr": lr,
                    "batch_size": bs,
                    "target_update_every_steps": tgt,
                    "decay_mode": decay_mode,
                    "eval_success": eval_success,
                    "eval_reward": eval_reward,
                    "eval_steps": eval_steps,
                    "eval_penalties": eval_penalties,
                },
            )
            torch.save(policy_net.state_dict(), os.path.join(OUTDIR, "best_model.pth"))
            with open(os.path.join(OUTDIR, "best_run.txt"), "w", encoding="utf-8") as f:
                f.write(str(best[1]) + "\n")

    print("\n================ BEST ================")
    assert best is not None
    print("Best score:", best[0])
    print("Best run:", best[1])
    print("CSV saved to:", csv_path)
    print("Best model:", os.path.join(OUTDIR, "best_model.pth"))


if __name__ == "__main__":
    main()