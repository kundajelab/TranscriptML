from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error for flattened arrays.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values with the same flattened shape as
            ``y_true``.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean((y_true - y_pred) ** 2)) if y_true.size else float("nan")


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation, returning NaN when undefined.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values to compare with ``y_true``.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 2 or y_pred.size < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])
