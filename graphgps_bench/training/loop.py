from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SeedState:
    seed: int
    random_probe: float
    torch_seed: int


@dataclass(frozen=True)
class EpochResult:
    loss: float
    steps: int
    examples: int


@dataclass(frozen=True)
class EvaluationResult:
    metric_name: str
    metric_value: float
    y_true: object
    y_pred: object


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_validation_metric: float
    test_metric: float
    duration_seconds: float


def seed_everything(seed: int) -> SeedState:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    probe = random.random()
    return SeedState(seed=seed, random_probe=probe, torch_seed=seed)


def _valid_targets(batch):
    import torch

    target = batch.y.float().view(-1, 1)
    return target, torch.isfinite(target) & (target >= 0.0)


def train_one_epoch(model, loader: Iterable, optimizer, device) -> EpochResult:
    import torch

    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    steps = 0
    examples = 0
    for batch in loader:
        batch = batch.to(device)
        target, mask = _valid_targets(batch)
        if not bool(mask.any()):
            continue
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1, 1)
        loss = criterion(logits[mask], target[mask])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
        examples += int(mask.sum().item())
    if not steps:
        raise RuntimeError("training loader produced no valid labels")
    return EpochResult(loss=total_loss / steps, steps=steps, examples=examples)


def predict(model, loader: Iterable, device):
    import torch

    model.eval()
    labels = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target, mask = _valid_targets(batch)
            if not bool(mask.any()):
                continue
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1, 1)
            labels.append(target[mask].detach().cpu())
            predictions.append(torch.sigmoid(logits[mask]).detach().cpu())
    if not labels:
        raise RuntimeError("evaluation loader produced no valid labels")
    return torch.cat(labels, dim=0), torch.cat(predictions, dim=0)


def evaluate_rocauc(model, loader: Iterable, device, task_name: str = "ogbg-molhiv") -> EvaluationResult:
    from graphgps_bench.evaluation.ogb import evaluate_with_ogb

    y_true, y_pred = predict(model, loader, device)
    scores = evaluate_with_ogb(task_name, y_true.numpy(), y_pred.numpy())
    return EvaluationResult(
        metric_name="rocauc",
        metric_value=float(scores["rocauc"]),
        y_true=y_true,
        y_pred=y_pred,
    )


def fit_model(
    model,
    train_loader: Iterable,
    valid_loader: Iterable,
    test_loader: Iterable,
    optimizer,
    *,
    device,
    epochs: int,
    early_stopping_patience: int,
    checkpoint_path: str | Path,
    task_name: str = "ogbg-molhiv",
) -> FitResult:
    import time
    import torch

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    start = time.perf_counter()
    best_metric = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device)
        validation = evaluate_rocauc(model, valid_loader, device, task_name)
        if validation.metric_value > best_metric:
            best_metric = validation.metric_value
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, checkpoint_path)
        else:
            stale_epochs += 1
        if stale_epochs > early_stopping_patience:
            break
    if best_epoch == 0:
        raise RuntimeError("no valid validation checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate_rocauc(model, test_loader, device, task_name)
    return FitResult(
        best_epoch=best_epoch,
        best_validation_metric=best_metric,
        test_metric=test.metric_value,
        duration_seconds=time.perf_counter() - start,
    )
