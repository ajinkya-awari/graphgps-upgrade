from graphgps_bench.data import make_fixture_batch


def test_synthetic_fixture_is_deterministic_for_same_seed():
    first = make_fixture_batch(seed=7)
    second = make_fixture_batch(seed=7)

    assert first == second
    assert first.validate() == []


def test_synthetic_fixture_changes_with_seed_without_changing_shapes():
    first = make_fixture_batch(seed=7)
    second = make_fixture_batch(seed=8)

    assert first != second
    assert first.num_graphs == second.num_graphs == 2
    assert first.num_nodes == second.num_nodes == 7
    assert all(len(label) == 1 for label in first.y)
