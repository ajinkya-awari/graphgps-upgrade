from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RunRecord:
    model: str
    seed: int
    metric_name: str
    metric_value: float
    config_hash: str
    split: str
    task_name: str
    fixture_only: bool
    checkpoint_path: str
    hardware: str

    def validate(self) -> None:
        if self.metric_name != "rocauc":
            raise ValueError("metric_name must be rocauc")
        if self.fixture_only is not True:
            raise ValueError("local report contracts may only use fixture metrics")
        if self.checkpoint_path:
            raise ValueError("fixture records must not reference checkpoints")
        if self.split != "scaffold":
            raise ValueError("split must be scaffold")
        if self.task_name != "ogbg-molhiv":
            raise ValueError("task_name must be ogbg-molhiv")
        if not 0.0 <= float(self.metric_value) <= 1.0:
            raise ValueError("metric_value must be in [0, 1]")


def write_json_manifest(path: str | Path, records: Iterable[RunRecord]) -> None:
    payload = []
    for record in records:
        record.validate()
        payload.append(asdict(record))
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_records(path: str | Path, records: Iterable[RunRecord]) -> None:
    rows = []
    for record in records:
        record.validate()
        rows.append(asdict(record))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RunRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(rows)
