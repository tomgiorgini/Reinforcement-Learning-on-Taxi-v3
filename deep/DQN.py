import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random


class DQN(nn.Module):
    def __init__(self, n_states: int, n_actions: int, emb_dim: int = 32, hidden: int = 128):
        super().__init__()

        #MLP with state embedding
        self.emb = nn.Embedding(num_embeddings=n_states, embedding_dim=emb_dim) # 500 x 32

        self.l1 = nn.Linear(emb_dim, hidden)                                    # 32 x 128
        self.l2 = nn.Linear(hidden, hidden)                                     # 128 x 128            
        self.l3 = nn.Linear(hidden, n_actions)                                  # 128 x 6   

    # forward pass 
    def forward(self, state_idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(state_idx)          
        x = F.relu(self.l1(x))           
        x = F.relu(self.l2(x))           
        x = self.l3(x)                   
        return x
    
    

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))
    
# Replay buffer for DQN (stores transitions and samples batches for learning)
class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        self.buffer.append(Transition(s, a, r, s2, done))

    def sample(self, batch_size: int) -> Transition:
        batch = self.rng.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))