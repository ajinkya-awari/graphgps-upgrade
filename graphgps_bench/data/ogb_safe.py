from __future__ import annotations


PYG_SAFE_GLOBAL_NAMES = (
    "torch_geometric.data.data.Data",
    "torch_geometric.data.data.DataEdgeAttr",
    "torch_geometric.data.data.DataTensorAttr",
    "torch_geometric.data.storage.GlobalStorage",
)


def pyg_safe_global_names() -> tuple[str, ...]:
    return PYG_SAFE_GLOBAL_NAMES


def register_pyg_safe_globals() -> tuple[str, ...]:
    import torch
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
    return PYG_SAFE_GLOBAL_NAMES
