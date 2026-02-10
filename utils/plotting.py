"""
utils/plotting.py

Matplotlib-only plotting utilities (no seaborn).

Goal: provide very readable diagnostic plots:
- raw curve (thin)
- rolling mean (thicker)
- optional evaluation points
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt


def rolling_mean(x, window):
    """
    Rolling mean that starts from episode 1:
    - for t < window: average over x[0:t]
    - for t >= window: average over last `window` points
    Returns an array of the SAME length as x.
    """
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
    y_eval: Optional[Sequence[float]] = None,
    eval_every: Optional[int] = None,
) -> None:
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)

    plt.figure()
    plt.plot(x, y, linewidth=1.0, alpha=0.35)  # raw
    y_rm = rolling_mean(y, rolling_window)
    plt.plot(x, y_rm, linewidth=2.0)           # smoothed

    # Optional: show evaluation markers (if you pass them already aligned)
    if y_eval is not None and eval_every is not None:
        y_eval = np.asarray(y_eval, dtype=float)
        idx = np.arange(len(x))
        eval_mask = (idx + 1) % eval_every == 0
        plt.scatter(x[eval_mask], y_eval[eval_mask], s=18)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()