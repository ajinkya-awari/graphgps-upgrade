from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import dataclass
from typing import Iterable

from .config import BenchmarkConfig, load_config


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    config_hash: str


def offline_preflight(config: BenchmarkConfig) -> PreflightResult:
    errors = list(config.validate_offline())
    warnings = compatibility_warnings(config.selected_versions.keys())
    return PreflightResult(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        config_hash=config.stable_hash(),
    )


def compatibility_warnings(packages: Iterable[str]) -> list[str]:
    warnings: list[str] = []
    for package in packages:
        if importlib.util.find_spec(package.replace("-", "_")) is None:
            warnings.append(f"{package} is not importable in this local environment")
            continue
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            warnings.append(f"{package} importable but package metadata is unavailable")
    return warnings


def require_runtime_dependencies() -> tuple[str, ...]:
    missing: list[str] = []
    for module_name in ("torch", "torch_geometric", "ogb"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return tuple(missing)


def select_device(requested: str):
    """Resolve an explicit device and fail instead of silently falling back."""
    if requested == "cpu":
        import torch

        return torch.device("cpu")
    if requested == "cuda" or (requested.startswith("cuda:") and requested[5:].isdigit()):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but unavailable")
        if requested.startswith("cuda:") and int(requested[5:]) >= torch.cuda.device_count():
            raise RuntimeError("CUDA device index out of range")
        return torch.device(requested)
    raise ValueError("device must be cpu or cuda[:index]")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run offline Project 12 preflight.")
    parser.add_argument("--config", default="configs/fixture.toml")
    args = parser.parse_args(argv)
    result = offline_preflight(load_config(args.config))
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}")
    print(f"config_hash={result.config_hash}")
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
