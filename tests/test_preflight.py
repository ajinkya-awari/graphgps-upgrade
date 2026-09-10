from graphgps_bench.config import from_mapping, load_config
import pytest

from graphgps_bench.preflight import offline_preflight, select_device


def test_fixture_config_passes_offline_preflight_with_external_actions_disabled():
    config = load_config("configs/fixture.toml")

    result = offline_preflight(config)

    assert result.ok is True
    assert result.errors == ()
    assert len(result.config_hash) == 64


def test_preflight_rejects_any_external_action_flag():
    config = from_mapping(
        {
            "task_name": "ogbg-molhiv",
            "split": "scaffold",
            "models": ["gcn", "gin", "gps"],
            "seeds": [0, 1, 2],
            "allow_network": True,
            "budget": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001, "early_stopping_patience": 0},
            "gps": {
                "positional_encoding": "laplacian_eigenvectors_fixture_contract",
                "positional_dim": 4,
                "local_layer": "GINConv",
                "global_attention": "MultiheadAttention",
                "hidden_dim": 16,
                "layers": 2,
                "dropout": 0.0,
            },
        }
    )

    result = offline_preflight(config)

    assert result.ok is False
    assert "allow_network must be false for offline preflight" in result.errors


def test_preflight_rejects_output_root_path_traversal():
    parent = "." + "."
    config = from_mapping(
        {
            "task_name": "ogbg-molhiv",
            "split": "scaffold",
            "models": ["gcn", "gin", "gps"],
            "seeds": [0, 1, 2],
            "output_root": f"outputs/{parent}/data/leak",
            "budget": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001, "early_stopping_patience": 0},
            "gps": {
                "positional_encoding": "laplacian_eigenvectors_fixture_contract",
                "positional_dim": 4,
                "local_layer": "GINConv",
                "global_attention": "MultiheadAttention",
                "hidden_dim": 16,
                "layers": 2,
                "dropout": 0.0,
            },
        }
    )

    result = offline_preflight(config)

    assert result.ok is False
    assert "output_root must stay under outputs/ or results/" in result.errors


def test_preflight_requires_explicit_gps_recipe_fields():
    config = from_mapping(
        {
            "task_name": "ogbg-molhiv",
            "split": "scaffold",
            "models": ["gcn", "gin", "gps"],
            "seeds": [0],
            "budget": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001, "early_stopping_patience": 0},
            "gps": {
                "positional_encoding": "",
                "positional_dim": 0,
                "local_layer": "",
                "global_attention": "",
                "hidden_dim": 16,
                "layers": 2,
                "dropout": 0.0,
            },
        }
    )

    result = offline_preflight(config)

    assert result.ok is False
    assert "gps.positional_encoding is required" in result.errors
    assert "gps.positional_dim must be positive" in result.errors
    assert "gps.local_layer is required" in result.errors
    assert "gps.global_attention is required" in result.errors


def test_preflight_rejects_unknown_device():
    config = load_config("configs/fixture.toml")
    invalid = config.__class__(
        task_name=config.task_name,
        split=config.split,
        models=config.models,
        seeds=config.seeds,
        budget=config.budget,
        gps=config.gps,
        output_root=config.output_root,
        device="tpu",
    )

    result = offline_preflight(invalid)

    assert result.ok is False
    assert "device must be cpu or cuda[:index]" in result.errors


def test_preflight_rejects_malformed_cuda_device():
    config = load_config("configs/fixture.toml")
    invalid = config.__class__(
        task_name=config.task_name,
        split=config.split,
        models=config.models,
        seeds=config.seeds,
        budget=config.budget,
        gps=config.gps,
        output_root=config.output_root,
        device="cuda:",
    )

    result = offline_preflight(invalid)

    assert result.ok is False
    assert "device must be cpu or cuda[:index]" in result.errors


def test_select_device_accepts_explicit_cpu_without_fallback():
    assert str(select_device("cpu")) == "cpu"


def test_select_device_rejects_unavailable_cuda_instead_of_falling_back():
    if select_device("cpu").type != "cpu":
        pytest.fail("explicit CPU selection did not resolve to CPU")
    with pytest.raises(RuntimeError, match="CUDA device requested but unavailable"):
        select_device("cuda")
