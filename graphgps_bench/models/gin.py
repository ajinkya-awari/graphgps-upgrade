from __future__ import annotations

from .base import require_torch


torch = require_torch()
nn = torch.nn


class GINClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        layers: int = 2,
        dropout: float = 0.0,
        use_ogb_encoder: bool = False,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GINConv, global_mean_pool
        from .encoders import build_node_encoder

        if layers <= 0:
            raise ValueError("layers must be positive")
        self.encoder = build_node_encoder(input_dim, hidden_dim, use_ogb_encoder)
        self.convs = nn.ModuleList(
            GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                train_eps=True,
            )
            for _ in range(layers)
        )
        self.dropout = float(dropout)
        self.pool = global_mean_pool
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        del edge_attr
        x = x.long() if x.dtype not in (torch.float16, torch.float32, torch.float64) else x.float()
        h = self.encoder(x)
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
            h = torch.nn.functional.dropout(h, p=self.dropout, training=self.training)
        return self.head(self.pool(h, batch))
