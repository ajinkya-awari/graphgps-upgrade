from __future__ import annotations

from .base import require_torch


torch = require_torch()
nn = torch.nn


class GCNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        del edge_index, edge_attr
        h = torch.relu(self.encoder(x.float()))
        pooled = _mean_pool(h, batch)
        return self.head(pooled)


def _mean_pool(h, batch):
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    pooled = h.new_zeros((graph_count, h.size(-1)))
    counts = h.new_zeros((graph_count, 1))
    pooled.index_add_(0, batch, h)
    counts.index_add_(0, batch, h.new_ones((h.size(0), 1)))
    return pooled / counts.clamp_min(1.0)
