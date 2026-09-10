from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Subset

from .ogb_safe import register_pyg_safe_globals


@dataclass(frozen=True)
class OGBBundle:
    dataset: object
    split_idx: dict[str, object]


def load_ogb_molhiv(root: str | Path) -> OGBBundle:
    from ogb.graphproppred import PygGraphPropPredDataset

    register_pyg_safe_globals()
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    split_idx = dataset.get_idx_split()
    required = {"train", "valid", "test"}
    if set(split_idx) != required:
        raise RuntimeError(f"unexpected OGB split keys: {sorted(split_idx)}")
    return OGBBundle(dataset=dataset, split_idx=split_idx)


def make_ogb_loader(bundle: OGBBundle, split: str, batch_size: int, *, shuffle: bool, seed: int):
    import torch
    from torch_geometric.loader import DataLoader

    if split not in bundle.split_idx:
        raise ValueError(f"unknown OGB split: {split}")
    indices = bundle.split_idx[split]
    subset = Subset(bundle.dataset, [int(index) for index in indices.tolist()])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def infer_feature_dimensions(bundle: OGBBundle) -> tuple[int, int]:
    sample = bundle.dataset[0]
    input_dim = int(sample.x.size(-1))
    edge_dim = int(sample.edge_attr.size(-1)) if getattr(sample, "edge_attr", None) is not None else 0
    return input_dim, edge_dim
