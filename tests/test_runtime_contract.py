from pathlib import Path

import pytest

from graphgps_bench.config import from_mapping, load_config
from graphgps_bench.reporting import RunRecord


ROOT = Path(__file__).resolve().parents[1]


def _runtime_config(**overrides):
    payload = {
        "task_name": "ogbg-molhiv",
        "split": "scaffold",
        "models": ["gcn", "gin", "gps"],
        "seeds": [0, 1, 2],
        "output_root": "results/ogbg_molhiv",
        "device": "cuda",
        "allow_network": True,
        "allow_ogb_download": True,
        "allow_gpu": True,
        "budget": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001, "early_stopping_patience": 0},
        "gps": {
            "positional_encoding": "laplacian_eigenvectors",
            "positional_dim": 4,
            "local_layer": "GINConv",
            "global_attention": "MultiheadAttention",
            "hidden_dim": 16,
            "layers": 2,
            "dropout": 0.0,
        },
    }
    payload.update(overrides)
    return from_mapping(payload)


def test_official_run_record_preserves_non_fixture_provenance():
    record = RunRecord(
        model="gps",
        seed=0,
        metric_name="rocauc",
        metric_value=0.75,
        config_hash="a" * 64,
        split="scaffold",
        task_name="ogbg-molhiv",
        fixture_only=False,
        checkpoint_path="results/ogbg_molhiv/gps/seed-0/best.pt",
        hardware="cuda",
        versions_json='{"torch":"2.10.0"}',
        parameter_count=1234,
        duration_seconds=2.5,
        best_epoch=1,
    )

    record.validate()


def test_runtime_config_requires_all_external_approval_flags():
    from graphgps_bench.runtime import validate_runtime_config

    config = _runtime_config(allow_gpu=False)

    with pytest.raises(ValueError, match="allow_gpu"):
        validate_runtime_config(config)


def test_full_ablation_config_matches_approved_frozen_budget():
    config = load_config(ROOT / "configs" / "kaggle_ablation.toml")

    assert config.models == ("gcn", "gin", "gps")
    assert config.seeds == (0, 1, 2)
    assert config.budget.epochs == 30
    assert config.budget.batch_size == 32
    assert config.budget.learning_rate == 0.001
    assert config.budget.early_stopping_patience == 5
    assert config.allow_network is True
    assert config.allow_ogb_download is True
    assert config.allow_gpu is True
    assert config.allow_wandb is False
