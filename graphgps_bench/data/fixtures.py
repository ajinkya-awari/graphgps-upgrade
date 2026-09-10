from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class FixtureBatch:
    x: tuple[tuple[float, ...], ...]
    edge_index: tuple[tuple[int, int], ...]
    edge_attr: tuple[tuple[float, ...], ...]
    batch: tuple[int, ...]
    y: tuple[tuple[float], ...]

    @property
    def num_graphs(self) -> int:
        return len(self.y)

    @property
    def num_nodes(self) -> int:
        return len(self.x)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if len(self.batch) != self.num_nodes:
            errors.append("batch length must equal number of nodes")
        if any(len(label) != 1 for label in self.y):
            errors.append("labels must have shape [B, 1]")
        if any(src < 0 or dst < 0 or src >= self.num_nodes or dst >= self.num_nodes for src, dst in self.edge_index):
            errors.append("edge_index contains an out-of-range node id")
        if len(self.edge_attr) != len(self.edge_index):
            errors.append("edge_attr length must equal edge_index length")
        if any(value != value or value in (float("inf"), float("-inf")) for row in self.x for value in row):
            errors.append("x contains non-finite values")
        return errors

    def to_torch(self):
        import torch

        return {
            "x": torch.tensor(self.x, dtype=torch.float32),
            "edge_index": torch.tensor(self.edge_index, dtype=torch.long).t().contiguous(),
            "edge_attr": torch.tensor(self.edge_attr, dtype=torch.float32),
            "batch": torch.tensor(self.batch, dtype=torch.long),
            "y": torch.tensor(self.y, dtype=torch.float32),
        }


def make_fixture_batch(seed: int = 0) -> FixtureBatch:
    rng = random.Random(seed)
    x: list[tuple[float, ...]] = []
    edge_index: list[tuple[int, int]] = []
    edge_attr: list[tuple[float, ...]] = []
    batch: list[int] = []
    labels: list[tuple[float]] = []
    cursor = 0
    for graph_id, node_count in enumerate((3, 4)):
        labels.append((float(graph_id % 2),))
        for node_id in range(node_count):
            atom_type = float((node_id + graph_id + seed) % 5)
            degree_hint = float((node_count - 1) % 4)
            x.append((atom_type, degree_hint, rng.random()))
            batch.append(graph_id)
        for node_id in range(node_count):
            src = cursor + node_id
            dst = cursor + ((node_id + 1) % node_count)
            edge_index.extend(((src, dst), (dst, src)))
            bond_type = float((node_id + seed) % 3)
            edge_attr.extend(((bond_type, 1.0), (bond_type, 1.0)))
        cursor += node_count
    return FixtureBatch(tuple(x), tuple(edge_index), tuple(edge_attr), tuple(batch), tuple(labels))
