from __future__ import annotations

from graphgps_bench.config import BenchmarkConfig


def validate_runtime_config(config: BenchmarkConfig) -> None:
    errors = [
        error
        for error in config.validate_offline()
        if not error.endswith("must be false for offline preflight")
    ]
    if not config.allow_network:
        errors.append("allow_network must be true for official runtime")
    if not config.allow_ogb_download:
        errors.append("allow_ogb_download must be true for official runtime")
    if config.device.startswith("cuda") and not config.allow_gpu:
        errors.append("allow_gpu must be true for CUDA runtime")
    if config.allow_wandb:
        errors.append("allow_wandb must remain false for this benchmark")
    if errors:
        raise ValueError("; ".join(errors))
