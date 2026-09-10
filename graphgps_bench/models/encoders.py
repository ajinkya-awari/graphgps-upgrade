from __future__ import annotations


def build_node_encoder(input_dim: int, hidden_dim: int, use_ogb_encoder: bool):
    import torch

    if not use_ogb_encoder:
        return torch.nn.Linear(input_dim, hidden_dim)
    try:
        from ogb.graphproppred.mol_encoder import AtomEncoder
    except ImportError as exc:
        raise RuntimeError("ogb is required for the molecular categorical encoder") from exc
    return AtomEncoder(emb_dim=hidden_dim)
