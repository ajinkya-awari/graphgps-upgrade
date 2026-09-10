import pytest

from graphgps_bench.config import load_config
from graphgps_bench.data import make_fixture_batch
from graphgps_bench.models import ModelSpec, build_model


torch = pytest.importorskip("torch")


def test_all_models_return_graph_logits_with_b_1_shape_and_finite_values():
    fixture = make_fixture_batch(seed=0).to_torch()
    config = load_config("configs/fixture.toml")

    for model_name in ("gcn", "gin", "gps"):
        model = build_model(
            ModelSpec(
                name=model_name,
                input_dim=fixture["x"].size(-1),
                edge_dim=fixture["edge_attr"].size(-1),
                hidden_dim=16,
                gps_recipe=config.gps if model_name == "gps" else None,
            )
        )
        output = model(fixture["x"], fixture["edge_index"], fixture["edge_attr"], fixture["batch"])

        assert tuple(output.shape) == (fixture["y"].size(0), 1)
        assert torch.isfinite(output).all()


def test_gps_output_is_deterministic_for_a_fixed_seed_and_fixture():
    fixture = make_fixture_batch(seed=3).to_torch()
    config = load_config("configs/fixture.toml")

    torch.manual_seed(123)
    first_model = build_model(
        ModelSpec(
            name="gps",
            input_dim=fixture["x"].size(-1),
            edge_dim=fixture["edge_attr"].size(-1),
            hidden_dim=16,
            gps_recipe=config.gps,
        )
    ).eval()
    torch.manual_seed(123)
    second_model = build_model(
        ModelSpec(
            name="gps",
            input_dim=fixture["x"].size(-1),
            edge_dim=fixture["edge_attr"].size(-1),
            hidden_dim=16,
            gps_recipe=config.gps,
        )
    ).eval()

    with torch.no_grad():
        first_output = first_model(
            fixture["x"], fixture["edge_index"], fixture["edge_attr"], fixture["batch"]
        )
        second_output = second_model(
            fixture["x"], fixture["edge_index"], fixture["edge_attr"], fixture["batch"]
        )

    torch.testing.assert_close(first_output, second_output)


def test_gps_model_exposes_explicit_recipe_metadata():
    fixture = make_fixture_batch(seed=0).to_torch()
    config = load_config("configs/fixture.toml")
    model = build_model(
        ModelSpec(
            name="gps",
            input_dim=fixture["x"].size(-1),
            edge_dim=fixture["edge_attr"].size(-1),
            hidden_dim=16,
            gps_recipe=config.gps,
        )
    )

    manifest = model.manifest()

    assert manifest["positional_encoding"] == "laplacian_eigenvectors_fixture_contract"
    assert manifest["local_layer"] == "GINConv"
    assert manifest["global_attention"] == "MultiheadAttention"
    assert manifest["interface"] == "forward(x, edge_index, edge_attr, batch) -> [B, 1]"
    assert manifest["implementation"] == "torch_geometric.nn.GPSConv"
    assert manifest["positional_encoding_scope"] == "graph-local Laplacian eigenvectors"


def test_gps_model_uses_real_pyg_gpsconv_adapter():
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import GPSConv

    fixture = make_fixture_batch(seed=0).to_torch()
    config = load_config("configs/fixture.toml")
    model = build_model(
        ModelSpec(
            name="gps",
            input_dim=fixture["x"].size(-1),
            edge_dim=fixture["edge_attr"].size(-1),
            hidden_dim=16,
            gps_recipe=config.gps,
        )
    )

    assert any(isinstance(module, GPSConv) for module in model.modules())

    with torch.no_grad():
        output = model(fixture["x"], fixture["edge_index"], fixture["edge_attr"], fixture["batch"])

    assert tuple(output.shape) == (fixture["y"].size(0), 1)
    assert torch.isfinite(output).all()


def test_gps_batch_attention_does_not_cross_graph_boundaries():
    fixture = make_fixture_batch(seed=0).to_torch()
    config = load_config("configs/fixture.toml")
    model = build_model(
        ModelSpec(
            name="gps",
            input_dim=fixture["x"].size(-1),
            edge_dim=fixture["edge_attr"].size(-1),
            hidden_dim=16,
            gps_recipe=config.gps,
        )
    )
    model.eval()

    changed = {key: value.clone() for key, value in fixture.items()}
    changed["x"][changed["batch"] == 1] += 100.0

    with torch.no_grad():
        original_output = model(
            fixture["x"], fixture["edge_index"], fixture["edge_attr"], fixture["batch"]
        )
        changed_output = model(
            changed["x"], changed["edge_index"], changed["edge_attr"], changed["batch"]
        )

    torch.testing.assert_close(original_output[0], changed_output[0])
