from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FORBIDDEN_ARTIFACT_SUFFIXES = (".pt", ".pth", ".safetensors", ".parquet", ".jsonl")
APPROVED_ARTIFACT_ROOTS = ("data", "datasets", "outputs", "results", "checkpoints", "wandb")
FORBIDDEN_CLAIM_PHRASES = (
    "SOTA",
    "state-of-the-art",
    "leaderboard reproduction",
    "clinical efficacy",
    "deployment ready",
    "GraphGPS advantage",
)
CROSS_PROJECT_MARKERS = ("../", "..\\", "portfolio-projects")
NEGATION_MARKERS = (
    "no ",
    "never ",
    "without ",
    "before measured evidence",
    "do not ",
    "must not ",
    "cannot ",
    "reject ",
    "avoid ",
    "unsupported ",
    "blocked ",
    "block ",
    "claiming ",
    "claims ",
    "unless ",
    "avoid ",
    "avoids ",
    "requires ",
    "require ",
)
SELF_VALIDATION_FILES = {
    "static_validation.py",
    "test_static_validation.py",
    "graphgps_bench/static_validation.py",
    "tests/test_static_validation.py",
}


@dataclass(frozen=True)
class StaticValidationResult:
    name: str
    ok: bool
    findings: tuple[str, ...]


def validate_forbidden_artifacts(root: str | Path) -> StaticValidationResult:
    root = Path(root)
    findings = [
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES
        and not _is_under_approved_artifact_root(path.relative_to(root))
    ]
    return StaticValidationResult("forbidden_artifacts", not findings, tuple(findings))


def validate_kaggle_kernel_metadata(path: str | Path) -> StaticValidationResult:
    path = Path(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    kernel_id = str(metadata.get("id", ""))
    title = str(metadata.get("title", ""))
    expected_slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
    actual_slug = kernel_id.rsplit("/", maxsplit=1)[-1]
    findings: list[str] = []
    if not kernel_id or "/" not in kernel_id:
        findings.append("kernel metadata id must use owner/slug format")
    if not title:
        findings.append("kernel metadata title is required")
    if title and actual_slug != expected_slug:
        findings.append(
            f"kernel id slug '{actual_slug}' does not match title-derived slug '{expected_slug}'"
        )
    return StaticValidationResult("kaggle_kernel_metadata", not findings, tuple(findings))


def validate_no_cross_project_references(
    root: str | Path,
    allowed_files: Iterable[str] = ("HANDOVER.md",),
) -> StaticValidationResult:
    root = Path(root)
    allowed = {item.replace("\\", "/") for item in allowed_files}
    findings: list[str] = []
    for path in _text_files(root):
        relative = str(path.relative_to(root)).replace("\\", "/")
        if relative in allowed or relative in SELF_VALIDATION_FILES:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if any(marker in line for marker in CROSS_PROJECT_MARKERS):
                findings.append(f"{relative}:{line_number}")
    return StaticValidationResult("cross_project_references", not findings, tuple(findings))


def validate_report_language(root: str | Path) -> StaticValidationResult:
    root = Path(root)
    findings: list[str] = []
    for path in _text_files(root):
        relative = str(path.relative_to(root)).replace("\\", "/")
        if relative in SELF_VALIDATION_FILES:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            for phrase in FORBIDDEN_CLAIM_PHRASES:
                if phrase.casefold() in line.casefold() and not _is_guardrail_line(line, phrase):
                    findings.append(f"{relative}:{line_number}: unsupported claim phrase '{phrase}'")
    return StaticValidationResult("report_language", not findings, tuple(findings))


def validate_notebook_approval_gates(path: str | Path) -> StaticValidationResult:
    path = Path(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    findings: list[str] = []
    required_messages = (
        "Approval required before dependency installation or verification",
        "Approval required before OGB download/access",
        "Approval required before GPU execution/training",
    )
    source_text = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    for message in required_messages:
        if message not in source_text:
            findings.append(f"missing notebook approval gate: {message}")
    return StaticValidationResult("notebook_approval_gates", not findings, tuple(findings))


def run_static_validation(root: str | Path = ".") -> tuple[StaticValidationResult, ...]:
    root = Path(root)
    return (
        validate_forbidden_artifacts(root),
        validate_no_cross_project_references(root),
        validate_report_language(root),
        validate_all_notebook_approval_gates(root / "notebooks"),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run local-safe static validation for Project 12.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    results = run_static_validation(args.root)
    for result in results:
        status = "ok" if result.ok else "fail"
        print(f"{result.name}={status}")
        for finding in result.findings:
            print(f"  {finding}")
    return 0 if all(result.ok for result in results) else 2


def _text_files(root: Path) -> list[Path]:
    suffixes = {".md", ".py", ".toml", ".json", ".ps1", ".sh", ".ipynb"}
    ignored_parts = {
        ".pytest_cache",
        "__pycache__",
        "kaggle_validation",
        "kaggle_gpu_smoke",
        "kaggle_gpu_smoke_v2",
        "kaggle_gpu_smoke_v3",
        "kaggle_official_benchmark",
        "kaggle_official_benchmark_run",
        "kaggle_full_ablation",
    }
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and not (set(path.parts) & ignored_parts)
    ]


def _is_under_approved_artifact_root(relative_path: Path) -> bool:
    return bool(relative_path.parts) and relative_path.parts[0] in APPROVED_ARTIFACT_ROOTS


def validate_all_notebook_approval_gates(path: str | Path) -> StaticValidationResult:
    path = Path(path)
    findings: list[str] = []
    notebooks = sorted(path.glob("kaggle*.ipynb"))
    if not notebooks:
        findings.append("missing Kaggle notebook")
    for notebook in notebooks:
        result = validate_notebook_approval_gates(notebook)
        findings.extend(f"{notebook.name}: {finding}" for finding in result.findings)
    return StaticValidationResult("notebook_approval_gates", not findings, tuple(findings))


def _is_guardrail_line(line: str, phrase: str) -> bool:
    lower = line.lower()
    phrase_index = lower.find(phrase.lower())
    if phrase_index < 0:
        return False
    prefix = lower[:phrase_index]
    stripped_prefix = prefix.strip().lstrip("-0123456789. )").strip()
    explicit_guardrail_prefixes = (
        "no ",
        "never ",
        "without ",
        "before measured evidence",
        "do not ",
        "must not ",
        "cannot ",
        "reject ",
        "avoid ",
        "unsupported ",
        "blocked ",
        "block ",
    )
    explicit_guardrail_markers = (
        " do not ",
        " must not ",
        " never ",
        " without ",
        " before measured evidence",
        " unless ",
        " unsupported ",
        " avoid",
        " reject",
        " block",
        " claim",
        " require",
    )
    padded_prefix = f" {stripped_prefix} "
    return stripped_prefix == "no" or stripped_prefix.startswith(explicit_guardrail_prefixes) or any(
        marker in padded_prefix for marker in explicit_guardrail_markers
    )


if __name__ == "__main__":
    raise SystemExit(main())
