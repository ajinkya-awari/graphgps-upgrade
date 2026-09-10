from __future__ import annotations

from dataclasses import asdict

from graphgps_bench.config import GPSRecipe
from .base import require_torch


torch = require_torch()
nn = torch.nn


def _require_pyg():
    try:
        from torch_geometric.nn import GINConv, GPSConv, global_mean_pool
    except ImportError as exc:
        raise RuntimeError(
            "torch-geometric is required for the GPSConv adapter; install the documented runtime dependencies"
        ) from exc
    return GINConv, GPSConv, global_mean_pool


class GPSClassifier(nn.Module):
    """Graph-level classifier built from the selected PyG GPSConv API."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        recipe: GPSRecipe,
        *,
        use_ogb_encoder: bool = False,
    ) -> None:
        super().__init__()
        errors = recipe.validate()
        if errors:
            raise ValueError("; ".join(errors))
        if recipe.local_layer != "GINConv":
            raise ValueError("the frozen GPS adapter currently requires gps.local_layer=GINConv")
        if recipe.global_attention != "MultiheadAttention":
            raise ValueError(
                "the frozen GPS adapter currently requires gps.global_attention=MultiheadAttention"
            )

        GINConv, GPSConv, _ = _require_pyg()
        from .encoders import build_node_encoder

        self.recipe = recipe
        self.node_encoder = build_node_encoder(input_dim, hidden_dim, use_ogb_encoder)
        self.encoder = nn.Linear(hidden_dim + recipe.positional_dim, hidden_dim)
        self.use_ogb_encoder = use_ogb_encoder
        self.convs = nn.ModuleList()
        for _ in range(recipe.layers):
            local_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            local_conv = GINConv(local_mlp, train_eps=True)
            self.convs.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=local_conv,
                    heads=1,
                    dropout=recipe.dropout,
                    attn_type="multihead",
                )
            )
        self.head = nn.Linear(hidden_dim, 1)

    def manifest(self) -> dict[str, object]:
        payload = asdict(self.recipe)
        payload["interface"] = "forward(x, edge_index, edge_attr, batch) -> [B, 1]"
        payload["implementation"] = "torch_geometric.nn.GPSConv"
        payload["positional_encoding_scope"] = "graph-local Laplacian eigenvectors"
        payload["node_encoder"] = "ogb.AtomEncoder" if self.use_ogb_encoder else "linear"
        return payload

    def forward(self, x, edge_index, edge_attr, batch):
        del edge_attr
        _, _, global_mean_pool = _require_pyg()
        x = self.node_encoder(x.long() if self.use_ogb_encoder else x.float())
        pe = _laplacian_positional_features(
            edge_index=edge_index,
            batch=batch,
            dim=self.recipe.positional_dim,
            dtype=x.dtype,
        )
        h = torch.relu(self.encoder(torch.cat([x, pe], dim=-1)))
        for conv in self.convs:
            h = conv(h, edge_index, batch=batch)
        return self.head(global_mean_pool(h, batch))


def _laplacian_positional_features(edge_index, batch, dim: int, dtype):
    """Compute deterministic, graph-local Laplacian eigenvector features."""
    node_count = int(batch.numel())
    features = torch.zeros((node_count, dim), device=batch.device, dtype=dtype)
    if node_count == 0:
        return features

    for graph_id in torch.unique(batch, sorted=True).tolist():
        nodes = torch.where(batch == graph_id)[0]
        local_index = torch.full((node_count,), -1, device=batch.device, dtype=torch.long)
        local_index[nodes] = torch.arange(nodes.numel(), device=batch.device)
        edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
        local_edges = local_index[edge_index[:, edge_mask]]

        adjacency = torch.zeros(
            (nodes.numel(), nodes.numel()), device=batch.device, dtype=dtype
        )
        if local_edges.numel():
            adjacency[local_edges[0], local_edges[1]] = 1.0
        adjacency = torch.maximum(adjacency, adjacency.transpose(0, 1))
        degree = adjacency.sum(dim=1)
        laplacian = torch.diag(degree) - adjacency
        _, eigenvectors = torch.linalg.eigh(laplacian)
        available = min(dim, max(0, nodes.numel() - 1))
        if available:
            features[nodes, :available] = eigenvectors[:, 1 : available + 1]
    return features
