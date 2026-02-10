"""
utils/logging.py

Lightweight structured logging for RL experiments.

This module exposes `EpisodeLog`, used by both the tabular and deep RL trainers.
We store episode metrics as a list of dictionaries and provide a helper to
convert to a dict-of-lists for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EpisodeLog:
    """
    Episode-by-episode metrics container.

    Each entry is a dict. Typical keys:
      - algorithm, seed, episode
      - episode_reward, steps, penalties, success, epsilon
      - optional evaluation keys: eval_mean_reward, eval_success_rate, ...
      - optional deep keys: loss_mean, q_mean, grad_norm, etc.
    """
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, **kwargs: Any) -> None:
        """Append a new episode entry."""
        self.entries.append(dict(kwargs))

    def as_dict_of_lists(self) -> Dict[str, List[Any]]:
        """
        Convert list-of-dicts to dict-of-lists for plotting.

        If some keys appear only in some episodes (e.g., evaluation metrics),
        they will have shorter lists. In plotting we usually align by episode
        and fill missing values with NaN.
        """
        out: Dict[str, List[Any]] = {}
        for row in self.entries:
            for k, v in row.items():
                out.setdefault(k, []).append(v)
        return out