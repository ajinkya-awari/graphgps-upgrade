from graphgps_bench.preflight import require_runtime_dependencies


def test_runtime_dependency_check_reports_missing_optional_packages_without_versions():
    missing = require_runtime_dependencies()

    assert isinstance(missing, tuple)
    assert all(isinstance(item, str) for item in missing)
