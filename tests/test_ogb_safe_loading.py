from graphgps_bench.data.ogb_safe import pyg_safe_global_names


def test_pyg_safe_global_names_are_explicit_and_minimal():
    assert pyg_safe_global_names() == (
        "torch_geometric.data.data.Data",
        "torch_geometric.data.data.DataEdgeAttr",
        "torch_geometric.data.data.DataTensorAttr",
        "torch_geometric.data.storage.GlobalStorage",
    )
