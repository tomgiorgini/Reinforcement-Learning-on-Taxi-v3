from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import torch
import random
import numpy as np
from typing import Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import os


#seeding
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# rolling mean for smoothing curves in plots
def rolling_mean(x, window):
    # for t < window: average over x[0:t]
    #for t >= window: average over last window values
    x = np.array(x, dtype=float)
    out = np.empty_like(x, dtype=float)

    cumsum = np.cumsum(x)
    for i in range(len(x)):
        if i < window:
            out[i] = cumsum[i] / (i + 1)
        else:
            out[i] = (cumsum[i] - cumsum[i - window]) / window
    return out


# plot test results
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
    

# Saving rolling means into csvs for the overlay of Q-learning vs DQN training curves    
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
    
    
# logging episodes
@dataclass
class EpisodeLog:
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, **kwargs: Any) -> None:
        self.entries.append(dict(kwargs))

    def as_dict_of_lists(self) -> Dict[str, List[Any]]:
        # Convert list-of-dicts to dict-of-lists for plotting.
        out: Dict[str, List[Any]] = {}
        for row in self.entries:
            for k, v in row.items():
                out.setdefault(k, []).append(v)
        return out
    
    
# Linear epsilon decay by episode
def linear_epsilon(episode: int, eps_start: float, eps_end: float, decay_episodes: int) -> float:
    if decay_episodes <= 0:
        return eps_end
    frac = min(1.0, episode / decay_episodes) # fraction of decay completed
    return eps_start + frac * (eps_end - eps_start) # linearly interpolate between start and end


# scoring function for hyperparameter tuning
def score(eval_success: float, eval_reward: float, eval_steps: float, eval_penalties: float) -> float:
    return 20 * eval_success + 1.0 * eval_reward - 1 * eval_steps - 10 * eval_penalties
