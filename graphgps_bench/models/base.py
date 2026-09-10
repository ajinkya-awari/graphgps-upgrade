from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphgps_bench.config import GPSRecipe


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_dim: int
    edge_dim: int
    hidden_dim: int
    gps_recipe: GPSRecipe | None = None


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to instantiate benchmark models") from exc
    return torch


def build_model(spec: ModelSpec) -> Any:
    if spec.name == "gcn":
        from .gcn import GCNClassifier

        return GCNClassifier(spec.input_dim, spec.hidden_dim)
    if spec.name == "gin":
        from .gin import GINClassifier

        return GINClassifier(spec.input_dim, spec.hidden_dim)
    if spec.name == "gps":
        from .gps import GPSClassifier

        if spec.gps_recipe is None:
            raise ValueError("gps model requires an explicit GPSRecipe")
        return GPSClassifier(spec.input_dim, spec.hidden_dim, spec.gps_recipe)
    raise ValueError(f"unsupported model name: {spec.name}")
