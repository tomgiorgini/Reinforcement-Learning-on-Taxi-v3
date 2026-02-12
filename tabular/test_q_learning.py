import os
import sys
import numpy as np
import gymnasium as gym

# Add project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.plotting import save_single_run_curve


def run_greedy_q_table(env_id: str, Q: np.ndarray, seed: int, episodes: int, max_steps: int):
    #Run greedy policy from Q-table and return per-episode metrics arrays
    env = gym.make(env_id)

    rewards = np.zeros(episodes, dtype=float)
    steps = np.zeros(episodes, dtype=int)
    penalties = np.zeros(episodes, dtype=int)
    success = np.zeros(episodes, dtype=float)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_r, ep_steps, ep_pen = 0.0, 0, 0
        ok = 0.0

        for _ in range(max_steps):
            action = int(np.argmax(Q[obs]))
            obs, r, terminated, truncated, _ = env.step(action)

            ep_r += float(r)
            ep_steps += 1
            if r == -10:
                ep_pen += 1
            if terminated and r == 20:
                ok = 1.0

            if terminated or truncated:
                break

        rewards[ep] = ep_r
        steps[ep] = ep_steps
        penalties[ep] = ep_pen
        success[ep] = ok

    env.close()
    return rewards, steps, penalties, success


if __name__ == "__main__":
    ENV_ID = "Taxi-v3"
    SEED = 42
    TEST_EPISODES = 1000
    MAX_STEPS = 200

    Q_PATH = "results/train_q_learning/Q_seed42.npy"
    OUTDIR = "results/test_q_learning_seed42"
    ROLLING = 200

    os.makedirs(OUTDIR, exist_ok=True)

    Q = np.load(Q_PATH)

    rewards, steps, penalties, success = run_greedy_q_table(
        env_id=ENV_ID,
        Q=Q,
        seed=SEED + 50_000,   # separate test seed space
        episodes=TEST_EPISODES,
        max_steps=MAX_STEPS
    )

    # Print summary
    print("\n[Q-LEARNING TEST]")
    print(f"Episodes: {TEST_EPISODES} | seed base: {SEED}")
    print(f"Mean reward: {rewards.mean():.2f}")
    print(f"Mean steps: {steps.mean():.2f}")
    print(f"Mean penalties: {penalties.mean():.2f}")
    print(f"Success rate: {success.mean():.3f}")

    # Plots (4)
    ep_axis = np.arange(1, TEST_EPISODES + 1)

    save_single_run_curve(
        ep_axis, rewards,
        title="TEST (greedy) - Reward per episode",
        xlabel="Test episode", ylabel="Reward",
        outpath=os.path.join(OUTDIR, "test_reward_1000.png"),
        rolling_window=ROLLING
    )

    save_single_run_curve(
        ep_axis, steps,
        title="TEST (greedy) - Steps per episode",
        xlabel="Test episode", ylabel="Steps",
        outpath=os.path.join(OUTDIR, "test_steps_1000.png"),
        rolling_window=ROLLING
    )

    save_single_run_curve(
        ep_axis, penalties,
        title="TEST (greedy) - Penalties per episode (-10 count)",
        xlabel="Test episode", ylabel="Penalties",
        outpath=os.path.join(OUTDIR, "test_penalties_1000.png"),
        rolling_window=ROLLING
    )

    save_single_run_curve(
        ep_axis, success,
        title="TEST (greedy) - Success (rolling mean)",
        xlabel="Test episode", ylabel="Success",
        outpath=os.path.join(OUTDIR, "test_success_1000.png"),
        rolling_window=ROLLING
    )

    print(f"\nPlots saved in: {OUTDIR}\n")