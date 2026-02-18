from __future__ import annotations

from dataclasses import replace
from itertools import product
import os
import sys

# Make project root importable when running directly
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import GlobalConfig, QLearningConfig
from utils import set_global_seeds
from utils import score
from tabular.train_q_learning import train_q_learning
from test_q_learning import run_greedy_q_table


def evaluate_q_table(env_id: str, Q, seed: int, episodes: int, max_steps: int) -> dict:
    rewards, steps, penalties, success = run_greedy_q_table(
        env_id=env_id,
        Q=Q,
        seed=seed,
        episodes=episodes,
        max_steps=max_steps,
    )

    return {
        "eval_mean_reward": float(rewards.mean()),
        "eval_mean_steps": float(steps.mean()),
        "eval_mean_penalties": float(penalties.mean()),
        "eval_success_rate": float(success.mean()),
    }


def main():
    SEED = 42
    TRAIN_EPISODES = 1500
    EVAL_EPISODES = 100  # greedy eval episodes after training

    set_global_seeds(SEED)

    global_cfg = GlobalConfig()

    base_cfg = QLearningConfig()
    base_cfg.episodes = TRAIN_EPISODES

    # grid search space
    alphas = [0.10, 0.30, 0.50, 0.70]      # learning rate
    gammas = [0.90, 0.95, 0.99]            # discount
    eps_ends = [0.01, 0.05, 0.10]          # final exploration
    decays = [500, 1000, 1500]             # decay length (episodes)

    total = len(alphas) * len(gammas) * len(eps_ends) * len(decays)
    run_id = 0
    best = None

    for alpha, gamma, eps_end, decay in product(alphas, gammas, eps_ends, decays):
        run_id += 1

        cfg = replace(
            base_cfg,
            alpha=alpha,
            gamma=gamma,
            eps_end=eps_end,
            eps_decay_episodes=decay,
        )

        # Train
        _, out, Q = train_q_learning(global_cfg, cfg, seed=SEED)

        # Greedy evaluation after training 
        max_steps = getattr(cfg, "max_steps_per_episode", 200)
        eval_metrics = evaluate_q_table(
            env_id=global_cfg.env_id,
            Q=Q,
            seed=SEED + 50_000,     # separate eval seed stream
            episodes=EVAL_EPISODES,
            max_steps=max_steps,
        )

        eval_reward = float(eval_metrics["eval_mean_reward"])
        eval_steps = float(eval_metrics["eval_mean_steps"])
        eval_success = float(eval_metrics["eval_success_rate"])
        eval_penalties = float(eval_metrics["eval_mean_penalties"])

        s = score(eval_success, eval_reward, eval_steps, eval_penalties)

        overrides = {"alpha": alpha, "gamma": gamma, "eps_end": eps_end, "eps_decay_episodes": decay}

        if best is None or s > best[0]:
            best = (s, overrides, eval_metrics)

        print(
            f"[{run_id}/{total}] alpha={alpha} gamma={gamma} eps_end={eps_end} decay={decay} "
            f"=> success={eval_success:.2f} reward={eval_reward:.1f} steps={eval_steps:.1f} pens={eval_penalties:.1f} "
            f"score={s:.2f}"
        )

    best_score, best_overrides, best_eval = best

    print("\nBEST CONFIG")
    print(f"Seed: {SEED} | Train episodes: {TRAIN_EPISODES} | Eval episodes: {EVAL_EPISODES}")
    print("Overrides:", best_overrides)
    print("Eval:", best_eval)
    print("Score:", best_score)

if __name__ == "__main__":
    main()
