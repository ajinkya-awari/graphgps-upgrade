from __future__ import annotations

from .contracts import validate_evaluator_inputs


def evaluate_with_ogb(task_name: str, y_true, y_pred) -> dict[str, float]:
    validate_evaluator_inputs(y_true, y_pred)
    if task_name != "ogbg-molhiv":
        raise ValueError("only ogbg-molhiv is supported")
    try:
        from ogb.graphproppred import Evaluator
    except ImportError as exc:
        raise RuntimeError("ogb is required for official evaluator execution") from exc
    evaluator = Evaluator(name=task_name)
    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})
