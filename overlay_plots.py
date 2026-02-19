"""
overlay_training_plots.py

Overlay plots (rolling mean) Q-learning vs DQN for:
- Reward
- Steps
- Penalties
- Success

Creates:
- overlay_<metric>_full.png
- overlay_<metric>_zoom.png
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLS = ["episode", "reward_rm", "steps_rm", "penalties_rm", "success_rm"]


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} missing columns: {missing}\n"
            f"Found: {list(df.columns)}\nExpected: {REQUIRED_COLS}"
        )
    df = df[REQUIRED_COLS].copy()
    df = df.sort_values("episode").reset_index(drop=True)
    return df


def _plot_overlay(
    q_df: pd.DataFrame,
    dqn_df: pd.DataFrame,
    col: str,
    ylabel: str,
    title_suffix: str,
    outpath: Path,
) -> None:
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    ax.plot(q_df["episode"], q_df[col], label="Q-learning (rolling mean)")
    ax.plot(dqn_df["episode"], dqn_df[col], label="DQN (rolling mean)")

    ax.set_title(f"Training comparison - {ylabel} (rolling mean, {title_suffix})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main() -> None:
    # =========================
    # CONFIG (edit here only)
    # =========================
    Q_CSV = "results/train_q_learning/q_learn_train_rolling_means.csv"
    DQN_CSV = "results/train_dqn/dqn_train_rolling_means.csv"

    OUTDIR = Path("results/overlays")      # change if you want (e.g., Path("figures/compare"))
    ZOOM_START = 1000                      # e.g. 1000
    ZOOM_END = None                        # None = last episode available

    # =========================
    # LOAD
    # =========================
    q_df = _load_csv(Q_CSV)
    dqn_df = _load_csv(DQN_CSV)

    ep_min = float(min(q_df["episode"].min(), dqn_df["episode"].min()))
    ep_max = float(max(q_df["episode"].max(), dqn_df["episode"].max()))
    zoom_end = float(ZOOM_END) if ZOOM_END is not None else ep_max

    # Metrics mapping: CSV column -> (label, filename stem)
    metrics = {
        "reward_rm": ("Reward", "reward"),
        "steps_rm": ("Steps", "steps"),
        "penalties_rm": ("Penalties", "penalties"),
        "success_rm": ("Success", "success"),
    }

    # =========================
    # FULL RANGE
    # =========================
    full_suffix = f"{int(ep_min)}-{int(ep_max)}"
    for col, (ylabel, stem) in metrics.items():
        _plot_overlay(
            q_df=q_df,
            dqn_df=dqn_df,
            col=col,
            ylabel=ylabel,
            title_suffix=full_suffix,
            outpath=OUTDIR / f"overlay_{stem}_full.png",
        )

    # =========================
    # ZOOMED (autoscale Y because we filter the window)
    # =========================
    q_zoom = q_df[(q_df["episode"] >= ZOOM_START) & (q_df["episode"] <= zoom_end)]
    dqn_zoom = dqn_df[(dqn_df["episode"] >= ZOOM_START) & (dqn_df["episode"] <= zoom_end)]

    zoom_suffix = f"{int(ZOOM_START)}-{int(zoom_end)}"
    for col, (ylabel, stem) in metrics.items():
        _plot_overlay(
            q_df=q_zoom,
            dqn_df=dqn_zoom,
            col=col,
            ylabel=ylabel,
            title_suffix=zoom_suffix,
            outpath=OUTDIR / f"overlay_{stem}_zoom.png",
        )

    print(f"Saved overlay plots to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()