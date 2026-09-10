from pathlib import Path

import pytest

from graphgps_bench.static_validation import (
    StaticValidationResult,
    validate_forbidden_artifacts,
    validate_no_cross_project_references,
    validate_notebook_approval_gates,
    validate_report_language,
)


ROOT = Path(__file__).resolve().parents[1]


def test_forbidden_artifact_scan_allows_source_and_blocks_runtime_outputs(tmp_path):
    (tmp_path / "graphgps_bench.py").write_text("print('fixture')\n", encoding="utf-8")
    (tmp_path / "model.pt").write_text("checkpoint", encoding="utf-8")

    result = validate_forbidden_artifacts(tmp_path)

    assert result.ok is False
    assert any("model.pt" in finding for finding in result.findings)
    assert not any("graphgps_bench.py" in finding for finding in result.findings)


def test_forbidden_artifact_scan_allows_approved_ignored_ogb_cache(tmp_path):
    cache = tmp_path / "data" / "ogb" / "ogbg_molhiv" / "processed"
    cache.mkdir(parents=True)
    (cache / "geometric_data_processed.pt").write_text("approved public dataset cache", encoding="utf-8")

    result = validate_forbidden_artifacts(tmp_path)

    assert result.ok is True


def test_cross_project_scan_flags_parent_path_references(tmp_path):
    doc = tmp_path / "HANDOVER.md"
    doc.write_text("See ../PORTFOLIO_STATUS.md for historical context.\n", encoding="utf-8")

    result = validate_no_cross_project_references(tmp_path, allowed_files=())

    assert result.ok is False
    assert "HANDOVER.md" in result.findings[0]


def test_cross_project_scan_allows_documented_historical_handover_reference():
    result = validate_no_cross_project_references(ROOT, allowed_files=("HANDOVER.md",))

    assert isinstance(result, StaticValidationResult)
    assert result.ok is True


def test_report_language_scan_rejects_unsupported_claim_phrases(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("This proves SOTA and GraphGPS advantage.\n", encoding="utf-8")

    result = validate_report_language(tmp_path)

    assert result.ok is False
    assert any("SOTA" in finding for finding in result.findings)


def test_report_language_scan_rejects_case_variant_of_blocked_claim(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("This benchmark is sota.\n", encoding="utf-8")

    result = validate_report_language(tmp_path)

    assert result.ok is False
    assert any("SOTA" in finding for finding in result.findings)


def test_report_language_scan_checks_later_unguarded_claim_occurrences(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        "Do not claim SOTA without measured evidence.\n"
        "This benchmark is SOTA.\n",
        encoding="utf-8",
    )

    result = validate_report_language(tmp_path)

    assert result.ok is False
    assert any("SOTA" in finding for finding in result.findings)


def test_report_language_scan_rejects_unrelated_negation_before_claim(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("This is not preliminary; it is SOTA.\n", encoding="utf-8")

    result = validate_report_language(tmp_path)

    assert result.ok is False
    assert any("SOTA" in finding for finding in result.findings)


def test_report_language_scan_allows_comma_separated_no_claim_guardrail(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        "No clinical efficacy, SOTA, upload, publication, outreach, or deployment claim is automatic.\n",
        encoding="utf-8",
    )

    result = validate_report_language(tmp_path)

    assert result.ok is True


def test_notebook_gate_scan_requires_self_blocking_approval_cells():
    result = validate_notebook_approval_gates(ROOT / "notebooks" / "kaggle_graphgps_benchmark.ipynb")

    assert result.ok is True


def test_requested_kaggle_upgrade_notebook_and_runbook_exist_with_manual_cells():
    notebook_path = ROOT / "notebooks" / "kaggle_graphgps_upgrade.ipynb"
    runbook_path = ROOT / "notebooks" / "KAGGLE_RUNBOOK_graphgps_upgrade.md"

    assert notebook_path.exists()
    assert runbook_path.exists()

    result = validate_notebook_approval_gates(notebook_path)

    assert result.ok is True


def test_kaggle_dependency_verification_cell_is_self_blocking():
    import json

    notebook_path = ROOT / "notebooks" / "kaggle_graphgps_upgrade.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source_texts = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    dependency_cells = [
        source
        for source in source_texts
        if "GPSConv" in source and "torch_geometric" in source and "inspect.signature" in source
    ]

    assert dependency_cells
    assert all("Approval required before dependency installation or verification" in source for source in dependency_cells)
    assert all("raise RuntimeError" in source for source in dependency_cells)
