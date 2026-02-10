"""
utils/seeding.py

Reproducibility utilities.

Taxi-v3 is deterministic given a seed, but we also need to control:
- numpy random
- python random
- torch (if used)

We keep this in one place and call it at the start of each run.
"""

from __future__ import annotations

import random
import numpy as np

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch not installed or not used -> ignore
        pass