import math

import pytest

from graphgps_bench.evaluation import fixture_auc_roc, validate_evaluator_inputs


def test_evaluator_contract_accepts_n_1_fixture_arrays():
    y_true = [[0.0], [1.0]]
    y_pred = [[0.2], [0.8]]

    validate_evaluator_inputs(y_true, y_pred)

    assert fixture_auc_roc(y_true, y_pred) == 1.0


def test_evaluator_contract_rejects_squeezed_arrays():
    with pytest.raises(ValueError, match="shape"):
        validate_evaluator_inputs([[0.0], [1.0]], [[0.2, 0.8], [0.1, 0.9]])


def test_evaluator_contract_rejects_non_finite_predictions():
    with pytest.raises(ValueError, match="non-finite"):
        validate_evaluator_inputs([[0.0], [1.0]], [[0.2], [math.inf]])
