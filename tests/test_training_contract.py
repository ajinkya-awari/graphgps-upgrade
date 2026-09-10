from types import SimpleNamespace

import pytest

from graphgps_bench.training.loop import seed_everything, train_one_epoch


torch = pytest.importorskip("torch")


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        del edge_index, edge_attr, batch
        return self.head(x[:, :1].view(-1, 1)).view(-1, 1)


class _Batch:
    def __init__(self, values, labels):
        self.x = torch.tensor(values, dtype=torch.float32)
        self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.edge_attr = torch.empty((0, 1), dtype=torch.float32)
        self.batch = torch.arange(len(values), dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    def to(self, device):
        for name in ("x", "edge_index", "edge_attr", "batch", "y"):
            setattr(self, name, getattr(self, name).to(device))
        return self


def test_seed_everything_controls_python_and_torch_streams():
    first = seed_everything(12)
    first_values = (first.random_probe, torch.rand(1).item())
    second = seed_everything(12)
    second_values = (second.random_probe, torch.rand(1).item())

    assert first_values == second_values
    assert first.torch_seed == 12


def test_train_one_epoch_returns_finite_loss_and_step_count():
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batches = [_Batch([[0.0], [1.0]], [0.0, 1.0])]

    result = train_one_epoch(model, batches, optimizer, torch.device("cpu"))

    assert result.steps == 1
    assert result.examples == 2
    assert torch.isfinite(torch.tensor(result.loss))
