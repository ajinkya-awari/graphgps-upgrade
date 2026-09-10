from __future__ import annotations

import math
from typing import Sequence


def validate_evaluator_inputs(y_true: Sequence[Sequence[float]], y_pred: Sequence[Sequence[float]]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same row count")
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        for row in values:
            if len(row) != 1:
                raise ValueError(f"{name} must have shape [N, 1]")
            if not math.isfinite(float(row[0])):
                raise ValueError(f"{name} contains non-finite values")


def fixture_auc_roc(y_true: Sequence[Sequence[float]], y_pred: Sequence[Sequence[float]]) -> float:
    validate_evaluator_inputs(y_true, y_pred)
    positives = [float(pred[0]) for label, pred in zip(y_true, y_pred) if float(label[0]) == 1.0]
    negatives = [float(pred[0]) for label, pred in zip(y_true, y_pred) if float(label[0]) == 0.0]
    if not positives or not negatives:
        raise ValueError("fixture ROC-AUC requires at least one positive and one negative label")
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total
