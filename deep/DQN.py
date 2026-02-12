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
        x = self.emb(state_idx)          # [B, emb_dim]
        x = F.relu(self.l1(x))           # [B, hidden]
        x = F.relu(self.l2(x))           # [B, hidden]
        x = self.l3(x)                   # [B, n_actions]
        return x