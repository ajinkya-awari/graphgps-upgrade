from __future__ import annotations

from .base import require_torch
from .gcn import _mean_pool


torch = require_torch()
nn = torch.nn


class GINClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        del edge_index, edge_attr
        h = self.mlp(x.float())
        return self.head(_mean_pool(h, batch))
