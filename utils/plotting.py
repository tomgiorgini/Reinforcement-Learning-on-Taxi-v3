from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os


def rolling_mean(x, window):
    # for t < window: average over x[0:t]
    #for t >= window: average over last `window` points
    x = np.array(x, dtype=float)
    out = np.empty_like(x, dtype=float)

    cumsum = np.cumsum(x)
    for i in range(len(x)):
        if i < window:
            out[i] = cumsum[i] / (i + 1)
        else:
            out[i] = (cumsum[i] - cumsum[i - window]) / window
    return out


def save_single_run_curve(
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: str,
    rolling_window: int = 50,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)

    plt.figure()
    plt.plot(x, y, linewidth=1.0, alpha=0.35)  # raw
    y_rm = rolling_mean(y, rolling_window)
    plt.plot(x, y_rm, linewidth=2.0)           # smoothed

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    
    
def save_rolling_means(outdir: str, episode, reward, steps, penalties, success, w: int, tag: str):
    os.makedirs(outdir, exist_ok=True)

    ep = np.asarray(episode, dtype=int)
    reward = np.asarray(reward, dtype=float)
    steps = np.asarray(steps, dtype=float)
    penalties = np.asarray(penalties, dtype=float)
    success = np.asarray(success, dtype=float)

    reward_rm = rolling_mean(reward, w)
    steps_rm = rolling_mean(steps, w)
    penalties_rm = rolling_mean(penalties, w)
    success_rm = rolling_mean(success, w)

    csv_path = os.path.join(outdir, f"{tag}_rolling_means.csv")
    header = "episode,reward_rm,steps_rm,penalties_rm,success_rm"
    data = np.column_stack([ep, reward_rm, steps_rm, penalties_rm, success_rm])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")