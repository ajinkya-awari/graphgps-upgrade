from graphgps_bench.cli import main


def test_cli_preflight_accepts_fixture_config():
    assert main(["--config", "configs/fixture.toml", "--mode", "preflight"]) == 0


def test_cli_blocks_kaggle_modes_locally_even_with_fixture_safe_config():
    assert main(["--config", "configs/fixture.toml", "--mode", "kaggle-smoke"]) == 2
    assert main(["--config", "configs/fixture.toml", "--mode", "ablation"]) == 2
