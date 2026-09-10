from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from .config import BenchmarkConfig
from .data.ogb_runtime import infer_feature_dimensions, load_ogb_molhiv, make_ogb_loader
from .models import ModelSpec, build_model
from .preflight import select_device
from .reporting import (
    RunRecord,
    aggregate_records,
    write_aggregate_figure,
    write_csv_records,
    write_json_manifest,
)
from .runtime import validate_runtime_config
from .training.loop import fit_model, seed_everything


def _versions() -> dict[str, str]:
    import importlib.metadata

    names = ("torch", "torch-geometric", "ogb")
    return {name: importlib.metadata.version(name) for name in names}


def run_official_benchmark(config: BenchmarkConfig, *, dataset_root: str | Path) -> dict[str, object]:
    import torch

    validate_runtime_config(config)
    device = select_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("official benchmark runtime requires the explicitly approved CUDA device")
    versions = _versions()
    bundle = load_ogb_molhiv(dataset_root)
    input_dim, edge_dim = infer_feature_dimensions(bundle)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[RunRecord] = []
    run_started = time.time()

    for model_name in config.models:
        for seed in config.seeds:
            seed_state = seed_everything(seed)
            loaders = {
                "train": make_ogb_loader(bundle, "train", config.budget.batch_size, shuffle=True, seed=seed),
                "valid": make_ogb_loader(bundle, "valid", config.budget.batch_size, shuffle=False, seed=seed),
                "test": make_ogb_loader(bundle, "test", config.budget.batch_size, shuffle=False, seed=seed),
            }
            spec = ModelSpec(
                name=model_name,
                input_dim=input_dim,
                edge_dim=edge_dim,
                hidden_dim=config.gps.hidden_dim,
                gps_recipe=config.gps if model_name == "gps" else None,
                layers=config.gps.layers,
                dropout=config.gps.dropout,
                use_ogb_encoder=True,
            )
            model = build_model(spec).to(device)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr=config.budget.learning_rate)
            checkpoint_path = output_root / model_name / f"seed-{seed}" / "best.pt"
            fit = fit_model(
                model,
                loaders["train"],
                loaders["valid"],
                loaders["test"],
                optimizer,
                device=device,
                epochs=config.budget.epochs,
                early_stopping_patience=config.budget.early_stopping_patience,
                checkpoint_path=checkpoint_path,
                task_name=config.task_name,
            )
            records.append(
                RunRecord(
                    model=model_name,
                    seed=seed,
                    metric_name="rocauc",
                    metric_value=fit.test_metric,
                    config_hash=config.stable_hash(),
                    split=config.split,
                    task_name=config.task_name,
                    fixture_only=False,
                    checkpoint_path=str(checkpoint_path),
                    hardware=torch.cuda.get_device_name(device),
                    versions_json=json.dumps(versions, sort_keys=True),
                    parameter_count=parameter_count,
                    duration_seconds=fit.duration_seconds,
                    best_epoch=fit.best_epoch,
                    seed_manifest_hash=seed_state_manifest(seed_state),
                )
            )

    manifest_path = output_root / "raw_records.json"
    csv_path = output_root / "raw_records.csv"
    write_json_manifest(manifest_path, records)
    write_csv_records(csv_path, records)
    aggregate = aggregate_records(records, config.models, config.seeds)
    aggregate_path = output_root / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_path = output_root / "aggregate.png"
    write_aggregate_figure(figure_path, aggregate)
    run_manifest = {
        "config_hash": config.stable_hash(),
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device),
        "platform": platform.platform(),
        "versions": versions,
        "dataset": "ogbg-molhiv",
        "split": config.split,
        "dataset_root": str(dataset_root),
        "source_revision": "provided-by-kaggle-package",
        "duration_seconds": time.time() - run_started,
        "raw_records": str(manifest_path),
        "raw_records_csv": str(csv_path),
        "aggregate": str(aggregate_path),
        "figure": str(figure_path),
    }
    run_manifest_path = output_root / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"records": records, "aggregate": aggregate, "manifest": run_manifest}


def seed_state_manifest(state) -> str:
    import hashlib

    payload = f"{state.seed}:{state.random_probe}:{state.torch_seed}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description="Run the explicitly approved OGB Kaggle benchmark.")
    parser.add_argument("--config", default="configs/kaggle_benchmark.toml")
    parser.add_argument("--dataset-root", default="data/ogb")
    args = parser.parse_args(argv)
    result = run_official_benchmark(load_config(args.config), dataset_root=args.dataset_root)
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
