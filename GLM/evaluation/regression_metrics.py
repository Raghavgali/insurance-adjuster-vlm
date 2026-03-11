from __future__ import annotations

import math
from typing import Any

from GLM.evaluation.prediction_schema import validate_prediction_record


def _extract_cost_pairs(records: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """Extract numeric prediction/reference cost pairs from records."""
    if not isinstance(records, list):
        raise TypeError(f"`records` must be list got {type(records).__name__}")
    if not records:
        raise ValueError("`records` must be a non-empty list")

    predicted_costs: list[float] = []
    reference_costs: list[float] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Prediction record at index {idx} must be dict got {type(record).__name__}")
        validate_prediction_record(record)

        predicted_cost = record.get("predicted_cost")
        reference_cost = record.get("reference_cost")
        if predicted_cost is None or reference_cost is None:
            raise ValueError(
                f"Record at index {idx} is missing regression targets; "
                "both `predicted_cost` and `reference_cost` are required"
            )

        predicted_costs.append(float(predicted_cost))
        reference_costs.append(float(reference_cost))

    return predicted_costs, reference_costs


def compute_mae(records: list[dict[str, Any]]) -> float:
    """
    Compute mean absolute error over prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing numeric prediction/reference costs.

    Returns
    -------
    float
        Mean absolute error.

    Notes
    -----
    - Requires `predicted_cost` and `reference_cost` in every record.
    - Uses straightforward absolute-error averaging.
    """
    predicted_costs, reference_costs = _extract_cost_pairs(records)
    abs_errors = [abs(pred - ref) for pred, ref in zip(predicted_costs, reference_costs)]
    return sum(abs_errors) / len(abs_errors)


def compute_rmse(records: list[dict[str, Any]]) -> float:
    """
    Compute root mean squared error over prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing numeric prediction/reference costs.

    Returns
    -------
    float
        Root mean squared error.

    Notes
    -----
    - Penalizes large regression errors more heavily than MAE.
    - Uses straightforward squared-error averaging.
    """
    predicted_costs, reference_costs = _extract_cost_pairs(records)
    squared_errors = [(pred - ref) ** 2 for pred, ref in zip(predicted_costs, reference_costs)]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def compute_mape(records: list[dict[str, Any]]) -> float:
    """
    Compute mean absolute percentage error over prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing numeric prediction/reference costs.

    Returns
    -------
    float
        Mean absolute percentage error in percentage units.

    Notes
    -----
    - Reference values equal to zero are skipped to avoid division-by-zero distortion.
    - Returns 0.0 when no valid reference denominators are available.
    """
    predicted_costs, reference_costs = _extract_cost_pairs(records)
    percentage_errors = [
        abs((pred - ref) / ref)
        for pred, ref in zip(predicted_costs, reference_costs)
        if ref != 0.0
    ]
    if not percentage_errors:
        return 0.0
    return 100.0 * sum(percentage_errors) / len(percentage_errors)


def compute_r2(records: list[dict[str, Any]]) -> float:
    """
    Compute coefficient of determination (R-squared) over prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing numeric prediction/reference costs.

    Returns
    -------
    float
        R-squared score.

    Notes
    -----
    - Returns 0.0 when the reference variance is zero.
    - Uses the standard 1 - SS_res / SS_tot formulation.
    """
    predicted_costs, reference_costs = _extract_cost_pairs(records)
    mean_reference = sum(reference_costs) / len(reference_costs)

    ss_res = sum((pred - ref) ** 2 for pred, ref in zip(predicted_costs, reference_costs))
    ss_tot = sum((ref - mean_reference) ** 2 for ref in reference_costs)
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def compute_median_absolute_error(records: list[dict[str, Any]]) -> float:
    """
    Compute median absolute error over prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing numeric prediction/reference costs.

    Returns
    -------
    float
        Median absolute error.

    Notes
    -----
    - More robust than MAE to extreme outliers.
    - Uses the midpoint average for even-length samples.
    """
    predicted_costs, reference_costs = _extract_cost_pairs(records)
    abs_errors = sorted(abs(pred - ref) for pred, ref in zip(predicted_costs, reference_costs))
    n = len(abs_errors)
    mid = n // 2
    if n % 2 == 1:
        return abs_errors[mid]
    return (abs_errors[mid - 1] + abs_errors[mid]) / 2.0


def compute_prediction_bias(records: list[dict[str, Any]]) -> float:
    """
    Compute mean signed prediction error over prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing numeric prediction/reference costs.

    Returns
    -------
    float
        Mean signed error, where positive values indicate overestimation.

    Notes
    -----
    - Useful for determining systematic over- or under-prediction.
    - Uses simple signed-error averaging.
    """
    predicted_costs, reference_costs = _extract_cost_pairs(records)
    signed_errors = [pred - ref for pred, ref in zip(predicted_costs, reference_costs)]
    return sum(signed_errors) / len(signed_errors)


def compute_max_absolute_error(records: list[dict[str, Any]]) -> float:
    """
    Compute maximum absolute error over prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing numeric prediction/reference costs.

    Returns
    -------
    float
        Maximum absolute error.

    Notes
    -----
    - Useful for worst-case error reporting.
    - Sensitive to individual outliers.
    """
    predicted_costs, reference_costs = _extract_cost_pairs(records)
    return max(abs(pred - ref) for pred, ref in zip(predicted_costs, reference_costs))


def compute_regression_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute the full regression-metric suite for prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing numeric prediction/reference costs.

    Returns
    -------
    dict[str, float]
        Combined regression metrics.

    Notes
    -----
    - Includes core accuracy metrics and simple bias/worst-case diagnostics.
    - Assumes costs were already parsed into numeric form before metric computation.
    """
    return {
        "mae": compute_mae(records),
        "rmse": compute_rmse(records),
        "mape": compute_mape(records),
        "r2": compute_r2(records),
        "median_absolute_error": compute_median_absolute_error(records),
        "prediction_bias": compute_prediction_bias(records),
        "max_absolute_error": compute_max_absolute_error(records),
    }
