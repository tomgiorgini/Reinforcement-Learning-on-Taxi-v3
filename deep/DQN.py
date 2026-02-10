"""
deep/DQN.py

Neural network for Taxi-v3.

Taxi-v3 has:
- 500 discrete states (0..499)
- 6 discrete actions

We use an Embedding for states instead of one-hot:
- More compact and usually faster than a 500-dim one-hot.
- Common approach for discrete-state DQN on Taxi.

Architecture (same spirit as the reference code):
Embedding(500 -> 32) -> Linear(32->128) -> ReLU -> Linear(128->128) -> ReLU -> Linear(128->n_actions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_states: int, n_actions: int, emb_dim: int = 32, hidden: int = 128):
        super().__init__()

        # In Taxi-v3 n_states is 500, but we keep it generic.
        self.emb = nn.Embedding(num_embeddings=n_states, embedding_dim=emb_dim)

        self.l1 = nn.Linear(emb_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, n_actions)

    def forward(self, state_idx: torch.Tensor) -> torch.Tensor:
        """
        state_idx: LongTensor of shape [B] (batch of discrete state ids).
        returns: Q-values tensor of shape [B, n_actions].
        """
        x = self.emb(state_idx)          # [B, emb_dim]
        x = F.relu(self.l1(x))           # [B, hidden]
        x = F.relu(self.l2(x))           # [B, hidden]
        x = self.l3(x)                   # [B, n_actions]
        return x