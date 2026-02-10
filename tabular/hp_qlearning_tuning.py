"""
tabular/tune_q_learning_simple.py

Very simple hyperparameter tuning for tabular Q-learning on Taxi-v3.

Constraints:
- single seed = 42
- training episodes = 1000  (you can change it here)
- minimal search (small grid)
- prints the best configuration + its evaluation metrics

Adds:
- saves results to CSV + JSON
- saves plots (score ranking, success vs reward)

Run:
  python tabular/tune_q_learning_simple.py
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Make project root importable when running directly
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import GlobalConfig, QLearningConfig
from utils.seeding import set_global_seeds
from tabular.train_q_learning import train_q_learning


def score(eval_success: float, eval_reward: float, eval_steps: float, eval_penalties: float) -> float:
    """
    Single scalar score used to rank configs.

    In Taxi, success_rate is the most important.
    Then reward, then (light) penalties for steps and penalties.
    """
    return 200.0 * eval_success + 1.0 * eval_reward - 0.25 * eval_steps - 2.0 * eval_penalties


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(path: str, header: list[str], rows: list[list[object]]) -> None:
    """Minimal CSV writer (no pandas dependency)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def main():
    # =============================
    # Fixed tuning settings
    # =============================
    SEED = 42
    TRAIN_EPISODES = 1000
    EVAL_EPISODES = 50  # greedy eval episodes at the end

    OUTDIR = "results/tuning_q_learning_simple"
    # =============================

    _ensure_dir(OUTDIR)

    set_global_seeds(SEED)

    global_cfg = GlobalConfig()
    global_cfg.eval_every_episodes = TRAIN_EPISODES   # evaluate only at the end
    global_cfg.eval_episodes = EVAL_EPISODES

    base_cfg = QLearningConfig()
    base_cfg.episodes = TRAIN_EPISODES

    # =============================
    # Small grid (Taxi-v3 sensible)
    # =============================
    alphas = [0.1, 0.3, 0.5]
    gammas = [0.90, 0.95, 0.99]
    eps_ends = [0.01, 0.05, 0.10]

    # You decided to NOT tune decay; we keep it fixed.
    # NOTE: if TRAIN_EPISODES=1000, decay=2000 means epsilon may still be > eps_end at the end.
    decays = [2000]
    # =============================

    best = None  # (best_score, overrides_dict, eval_metrics_dict)

    total = len(alphas) * len(gammas) * len(eps_ends) * len(decays)
    i = 0

    # We store every run here
    rows = []
    results_json = []

    for alpha, gamma, eps_end, decay in product(alphas, gammas, eps_ends, decays):
        i += 1
        cfg = replace(
            base_cfg,
            alpha=alpha,
            gamma=gamma,
            eps_end=eps_end,
            eps_decay_episodes=decay,
        )

        _, out,Q = train_q_learning(global_cfg, cfg, seed=SEED)

        # Because eval happens only at the end, last eval_* is the one we want
        eval_reward = float(out["eval_mean_reward"][-1])
        eval_steps = float(out["eval_mean_steps"][-1])
        eval_success = float(out["eval_success_rate"][-1])
        eval_penalties = float(out["eval_mean_penalties"][-1])

        s = score(eval_success, eval_reward, eval_steps, eval_penalties)

        overrides = {"alpha": alpha, "gamma": gamma, "eps_end": eps_end, "eps_decay_episodes": decay}
        eval_metrics = {
            "eval_mean_reward": eval_reward,
            "eval_mean_steps": eval_steps,
            "eval_success_rate": eval_success,
            "eval_mean_penalties": eval_penalties
        }

        # save row for CSV
        rows.append([
            i, alpha, gamma, eps_end, decay,
            eval_reward, eval_steps, eval_success, eval_penalties, s
        ])

        # save record for JSON
        results_json.append({
            "run_id": i,
            "seed": SEED,
            "train_episodes": TRAIN_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "overrides": overrides,
            "eval": eval_metrics,
            "score": s
        })

        if best is None or s > best[0]:
            best = (s, overrides, eval_metrics)

        print(f"[{i}/{total}] alpha={alpha} gamma={gamma} eps_end={eps_end} decay={decay} "
              f"=> success={eval_success:.2f} reward={eval_reward:.1f} steps={eval_steps:.1f} pens={eval_penalties:.1f} "
              f"score={s:.2f}")

    best_score, best_overrides, best_eval = best

    # -----------------------------
    # Save outputs (CSV + JSON)
    # -----------------------------
    csv_path = os.path.join(OUTDIR, "tuning_results.csv")
    json_path = os.path.join(OUTDIR, "tuning_results.json")

    header = [
        "run_id", "alpha", "gamma", "eps_end", "eps_decay_episodes",
        "eval_mean_reward", "eval_mean_steps", "eval_success_rate", "eval_mean_penalties",
        "score"
    ]
    save_csv(csv_path, header, rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best": {"overrides": best_overrides, "eval": best_eval, "score": best_score},
                "all_runs": results_json
            },
            f,
            indent=2
        )

    # -----------------------------
    # Create plots
    # -----------------------------
    # Convert rows to numpy for easier plotting
    arr = np.array(rows, dtype=float)
    run_ids = arr[:, 0]
    eval_rewards = arr[:, 5]
    eval_steps = arr[:, 6]
    eval_success = arr[:, 7]
    eval_penalties = arr[:, 8]
    scores = arr[:, 9]

    # 1) Score ranking plot (sorted)
    order = np.argsort(scores)[::-1]
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, len(scores) + 1), scores[order], marker="o")
    plt.title("Q-learning tuning — Score ranking (best to worst)")
    plt.xlabel("Rank (1 = best)")
    plt.ylabel("Score")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "plot_score_ranking.png"))
    plt.close()

    # 2) Success vs Reward scatter (colored by score)
    plt.figure(figsize=(6.5, 5))
    sc = plt.scatter(eval_rewards, eval_success, c=scores, s=70)
    plt.title("Q-learning tuning — Success vs Reward")
    plt.xlabel("Eval mean reward")
    plt.ylabel("Eval success rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.colorbar(sc, label="Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "plot_success_vs_reward.png"))
    plt.close()

    # 3) Steps vs Penalties scatter (colored by success)
    plt.figure(figsize=(6.5, 5))
    sc2 = plt.scatter(eval_steps, eval_penalties, c=eval_success, s=70)
    plt.title("Q-learning tuning — Steps vs Penalties")
    plt.xlabel("Eval mean steps")
    plt.ylabel("Eval mean penalties")
    plt.grid(alpha=0.25)
    plt.colorbar(sc2, label="Eval success rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "plot_steps_vs_penalties.png"))
    plt.close()

    # Print best summary
    print("\n================ BEST CONFIG ================")
    print(f"Seed: {SEED} | Train episodes: {TRAIN_EPISODES} | Eval episodes: {EVAL_EPISODES}")
    print("Overrides:", best_overrides)
    print("Eval:", best_eval)
    print("Score:", best_score)

    print("\nSaved:")
    print(f"- CSV:  {csv_path}")
    print(f"- JSON: {json_path}")
    print(f"- Plots in: {OUTDIR}")


if __name__ == "__main__":
    main()