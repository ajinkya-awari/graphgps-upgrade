import pytest

from graphgps_bench.reporting import RunRecord, aggregate_records


def _record(model: str, seed: int, value: float = 0.5) -> RunRecord:
    return RunRecord(
        model=model,
        seed=seed,
        metric_name="rocauc",
        metric_value=value,
        config_hash="a" * 64,
        split="scaffold",
        task_name="ogbg-molhiv",
        fixture_only=True,
        checkpoint_path="",
        hardware="local-fixture",
    )


def test_aggregate_requires_complete_model_seed_grid():
    records = [_record("gcn", 0), _record("gin", 0), _record("gps", 0)]

    with pytest.raises(ValueError, match="required seeds"):
        aggregate_records(records, ("gcn", "gin", "gps"), (0, 1))


def test_aggregate_rejects_duplicate_model_seed_records():
    records = [_record(model, seed) for model in ("gcn", "gin", "gps") for seed in (0, 1)]
    records.append(_record("gcn", 0, 0.9))

    with pytest.raises(ValueError, match="duplicate model/seed"):
        aggregate_records(records, ("gcn", "gin", "gps"), (0, 1))


def test_aggregate_rejects_cross_model_config_hash_mismatch():
    records = [
        _record("gcn", 0),
        _record("gin", 0).__class__(
            model="gin",
            seed=0,
            metric_name="rocauc",
            metric_value=0.5,
            config_hash="b" * 64,
            split="scaffold",
            task_name="ogbg-molhiv",
            fixture_only=True,
            checkpoint_path="",
            hardware="local-fixture",
        ),
        _record("gps", 0),
    ]

    with pytest.raises(ValueError, match="mismatched config hashes"):
        aggregate_records(records, ("gcn", "gin", "gps"), (0,))


def test_aggregate_rejects_unexpected_model_records():
    records = [_record(model, 0) for model in ("gcn", "gin", "gps")]
    records.append(_record("gat", 0))

    with pytest.raises(ValueError, match="unexpected records for models"):
        aggregate_records(records, ("gcn", "gin", "gps"), (0,))


def test_aggregate_derives_values_from_fixture_records_without_winner_claims():
    records = [_record(model, seed, 0.5 + (seed * 0.1)) for model in ("gcn", "gin", "gps") for seed in (0, 1)]

    aggregate = aggregate_records(records, ("gcn", "gin", "gps"), (0, 1))

    assert set(aggregate) == {"gcn", "gin", "gps"}
    assert aggregate["gcn"]["mean"] == pytest.approx(0.55)
    assert aggregate["gps"]["n"] == 2.0


def test_fixture_record_rejects_checkpoint_and_non_fixture_metrics():
    with pytest.raises(ValueError, match="checkpoints"):
        _record("gcn", 0).__class__(
            model="gcn",
            seed=0,
            metric_name="rocauc",
            metric_value=0.5,
            config_hash="a" * 64,
            split="scaffold",
            task_name="ogbg-molhiv",
            fixture_only=True,
            checkpoint_path="checkpoints/model.pt",
            hardware="local-fixture",
        ).validate()
