from graphgps_bench.training import seed_manifest


def test_seed_manifest_is_deterministic_and_hash_backed():
    first = seed_manifest(42)
    second = seed_manifest(42)

    assert first == second
    assert first.seed == 42
    assert len(first.manifest_hash) == 64

