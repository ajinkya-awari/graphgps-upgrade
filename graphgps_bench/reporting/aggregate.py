from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Iterable

from .records import RunRecord


def aggregate_records(records: Iterable[RunRecord], required_models: tuple[str, ...], required_seeds: tuple[int, ...]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    allowed_models = set(required_models)
    for record in records:
        record.validate()
        if record.model not in allowed_models:
            raise ValueError(f"unexpected records for models: {record.model}")
        identity = (record.model, record.seed)
        if identity in seen:
            raise ValueError(f"duplicate model/seed record: {record.model}/{record.seed}")
        seen.add(identity)
        grouped[record.model].append(record)
    missing_models = set(required_models) - set(grouped)
    if missing_models:
        raise ValueError(f"missing records for models: {sorted(missing_models)}")
    output: dict[str, dict[str, float]] = {}
    for model in required_models:
        model_records = grouped[model]
        seeds = {record.seed for record in model_records}
        if seeds != set(required_seeds):
            raise ValueError(f"{model} records do not match required seeds")
        hashes = {record.config_hash for record in model_records}
        if len(hashes) != 1:
            raise ValueError(f"{model} records have mismatched config hashes")
        values = [float(record.metric_value) for record in sorted(model_records, key=lambda item: item.seed)]
        output[model] = {
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": float(len(values)),
        }
    all_hashes = {record.config_hash for records_for_model in grouped.values() for record in records_for_model}
    if len(all_hashes) != 1:
        raise ValueError("records have mismatched config hashes across models")
    return output
