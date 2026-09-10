from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ALLOWED_MODELS = ("gcn", "gin", "gps")
EXTERNAL_FLAGS = ("allow_network", "allow_ogb_download", "allow_gpu", "allow_wandb")
DEFAULT_OUTPUT_ROOT = "outputs/fixture"


@dataclass(frozen=True)
class GPSRecipe:
    positional_encoding: str
    positional_dim: int
    local_layer: str
    global_attention: str
    hidden_dim: int
    layers: int
    dropout: float

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.positional_encoding:
            errors.append("gps.positional_encoding is required")
        if self.positional_dim <= 0:
            errors.append("gps.positional_dim must be positive")
        if not self.local_layer:
            errors.append("gps.local_layer is required")
        if not self.global_attention:
            errors.append("gps.global_attention is required")
        if self.hidden_dim <= 0:
            errors.append("gps.hidden_dim must be positive")
        if self.layers <= 0:
            errors.append("gps.layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            errors.append("gps.dropout must be in [0, 1)")
        return errors


@dataclass(frozen=True)
class Budget:
    epochs: int
    batch_size: int
    learning_rate: float
    early_stopping_patience: int

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.epochs <= 0:
            errors.append("budget.epochs must be positive")
        if self.batch_size <= 0:
            errors.append("budget.batch_size must be positive")
        if self.learning_rate <= 0:
            errors.append("budget.learning_rate must be positive")
        if self.early_stopping_patience < 0:
            errors.append("budget.early_stopping_patience must be non-negative")
        return errors


@dataclass(frozen=True)
class BenchmarkConfig:
    task_name: str
    split: str
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    budget: Budget
    gps: GPSRecipe
    output_root: str = DEFAULT_OUTPUT_ROOT
    device: str = "cpu"
    allow_network: bool = False
    allow_ogb_download: bool = False
    allow_gpu: bool = False
    allow_wandb: bool = False
    selected_versions: Mapping[str, str] = field(default_factory=dict)

    def validate_offline(self) -> list[str]:
        errors: list[str] = []
        if self.task_name != "ogbg-molhiv":
            errors.append("task_name must be ogbg-molhiv")
        if self.split != "scaffold":
            errors.append("split must be scaffold")
        if tuple(self.models) != ALLOWED_MODELS:
            errors.append("models must be exactly ['gcn', 'gin', 'gps'] in that order")
        if not self.seeds:
            errors.append("seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            errors.append("seeds must be unique")
        if any(seed < 0 for seed in self.seeds):
            errors.append("seeds must be non-negative")
        valid_cuda_device = self.device == "cuda" or (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        )
        if self.device != "cpu" and not valid_cuda_device:
            errors.append("device must be cpu or cuda[:index]")
        for flag in EXTERNAL_FLAGS:
            if getattr(self, flag):
                errors.append(f"{flag} must be false for offline preflight")
        normalized = self.output_root.replace("\\", "/")
        output_parts = PurePosixPath(normalized).parts
        if (
            not output_parts
            or PurePosixPath(normalized).is_absolute()
            or output_parts[0] not in {"outputs", "results"}
            or ".." in output_parts
        ):
            errors.append("output_root must stay under outputs/ or results/")
        errors.extend(self.budget.validate())
        errors.extend(self.gps.validate())
        return errors

    def stable_hash(self) -> str:
        payload = to_manifest_dict(self)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)
    if path.suffix.lower() != ".toml":
        raise ValueError("Only TOML configs are supported locally without optional YAML dependencies")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return from_mapping(data)


def from_mapping(data: Mapping[str, Any]) -> BenchmarkConfig:
    budget_data = data.get("budget", {})
    gps_data = data.get("gps", {})
    return BenchmarkConfig(
        task_name=str(data.get("task_name", "")),
        split=str(data.get("split", "")),
        models=tuple(str(item) for item in data.get("models", ())),
        seeds=tuple(int(item) for item in data.get("seeds", ())),
        budget=Budget(
            epochs=int(budget_data.get("epochs", 0)),
            batch_size=int(budget_data.get("batch_size", 0)),
            learning_rate=float(budget_data.get("learning_rate", 0.0)),
            early_stopping_patience=int(budget_data.get("early_stopping_patience", 0)),
        ),
        gps=GPSRecipe(
            positional_encoding=str(gps_data.get("positional_encoding", "")),
            positional_dim=int(gps_data.get("positional_dim", 0)),
            local_layer=str(gps_data.get("local_layer", "")),
            global_attention=str(gps_data.get("global_attention", "")),
            hidden_dim=int(gps_data.get("hidden_dim", 0)),
            layers=int(gps_data.get("layers", 0)),
            dropout=float(gps_data.get("dropout", 0.0)),
        ),
        output_root=str(data.get("output_root", DEFAULT_OUTPUT_ROOT)),
        device=str(data.get("device", "cpu")),
        allow_network=bool(data.get("allow_network", False)),
        allow_ogb_download=bool(data.get("allow_ogb_download", False)),
        allow_gpu=bool(data.get("allow_gpu", False)),
        allow_wandb=bool(data.get("allow_wandb", False)),
        selected_versions=dict(data.get("selected_versions", {})),
    )


def to_manifest_dict(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "task_name": config.task_name,
        "split": config.split,
        "models": list(config.models),
        "seeds": list(config.seeds),
        "budget": {
            "epochs": config.budget.epochs,
            "batch_size": config.budget.batch_size,
            "learning_rate": config.budget.learning_rate,
            "early_stopping_patience": config.budget.early_stopping_patience,
        },
        "gps": {
            "positional_encoding": config.gps.positional_encoding,
            "positional_dim": config.gps.positional_dim,
            "local_layer": config.gps.local_layer,
            "global_attention": config.gps.global_attention,
            "hidden_dim": config.gps.hidden_dim,
            "layers": config.gps.layers,
            "dropout": config.gps.dropout,
        },
        "output_root": config.output_root,
        "device": config.device,
        "allow_network": config.allow_network,
        "allow_ogb_download": config.allow_ogb_download,
        "allow_gpu": config.allow_gpu,
        "allow_wandb": config.allow_wandb,
        "selected_versions": dict(config.selected_versions),
    }
