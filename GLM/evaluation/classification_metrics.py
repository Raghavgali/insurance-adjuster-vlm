from __future__ import annotations

from typing import Any

from GLM.evaluation.prediction_schema import validate_prediction_record


def _extract_label_pairs(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Extract predicted/reference label pairs from records."""
    if not isinstance(records, list):
        raise TypeError(f"`records` must be list got {type(records).__name__}")
    if not records:
        raise ValueError("`records` must be a non-empty list")

    predicted_labels: list[str] = []
    reference_labels: list[str] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Prediction record at index {idx} must be dict got {type(record).__name__}")
        validate_prediction_record(record)

        predicted_label = record.get("predicted_label")
        reference_label = record.get("reference_label")
        if predicted_label is None or reference_label is None:
            raise ValueError(
                f"Record at index {idx} is missing classification labels; "
                "both `predicted_label` and `reference_label` are required"
            )

        predicted_labels.append(predicted_label.strip())
        reference_labels.append(reference_label.strip())

    return predicted_labels, reference_labels


def compute_accuracy(records: list[dict[str, Any]]) -> float:
    """
    Compute classification accuracy over prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing predicted and reference labels.

    Returns
    -------
    float
        Accuracy score in the range [0.0, 1.0].

    Notes
    -----
    - Requires `predicted_label` and `reference_label` in every record.
    - Uses exact string equality on normalized labels.
    """
    predicted_labels, reference_labels = _extract_label_pairs(records)
    matches = sum(int(pred == ref) for pred, ref in zip(predicted_labels, reference_labels))
    return matches / len(predicted_labels)


def compute_precision_recall_f1(records: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute macro/micro/weighted precision, recall, and F1 scores.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing predicted and reference labels.

    Returns
    -------
    dict[str, float]
        Precision, recall, and F1 metrics across averaging strategies.

    Notes
    -----
    - Relies on `sklearn.metrics`.
    - Uses `zero_division=0` to avoid runtime errors on sparse classes.
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError as exc:
        raise RuntimeError("Classification precision/recall/F1 requires scikit-learn.") from exc

    predicted_labels, reference_labels = _extract_label_pairs(records)
    metrics: dict[str, float] = {}

    for average in ("macro", "micro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            reference_labels,
            predicted_labels,
            average=average,
            zero_division=0,
        )
        metrics[f"{average}_precision"] = float(precision)
        metrics[f"{average}_recall"] = float(recall)
        metrics[f"{average}_f1"] = float(f1)

    return metrics


def compute_per_class_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """
    Compute per-class precision, recall, F1, and support.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing predicted and reference labels.

    Returns
    -------
    dict[str, dict[str, float]]
        Per-class metrics keyed by reference label name.

    Notes
    -----
    - Relies on `sklearn.metrics`.
    - Output is JSON-safe and suitable for metrics/error-analysis reports.
    """
    try:
        from sklearn.metrics import classification_report
    except ImportError as exc:
        raise RuntimeError("Per-class classification metrics require scikit-learn.") from exc

    predicted_labels, reference_labels = _extract_label_pairs(records)
    report = classification_report(
        reference_labels,
        predicted_labels,
        output_dict=True,
        zero_division=0,
    )

    excluded_keys = {"accuracy", "macro avg", "weighted avg"}
    per_class: dict[str, dict[str, float]] = {}
    for label, values in report.items():
        if label in excluded_keys or not isinstance(values, dict):
            continue
        per_class[label] = {
            "precision": float(values.get("precision", 0.0)),
            "recall": float(values.get("recall", 0.0)),
            "f1": float(values.get("f1-score", 0.0)),
            "support": float(values.get("support", 0.0)),
        }

    return per_class


def compute_confusion_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute a JSON-safe confusion matrix from prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing predicted and reference labels.

    Returns
    -------
    dict[str, Any]
        Confusion matrix payload including label order and matrix values.

    Notes
    -----
    - Relies on `sklearn.metrics`.
    - Matrix is returned as nested lists for straightforward JSON serialization.
    """
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:
        raise RuntimeError("Confusion matrix computation requires scikit-learn.") from exc

    predicted_labels, reference_labels = _extract_label_pairs(records)
    labels = sorted(set(reference_labels) | set(predicted_labels))
    matrix = confusion_matrix(reference_labels, predicted_labels, labels=labels)
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
    }


def compute_classification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute the full classification-metric suite for prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing predicted and reference labels.

    Returns
    -------
    dict[str, Any]
        Combined classification metrics and diagnostics.

    Notes
    -----
    - Includes overall accuracy, aggregated precision/recall/F1, per-class metrics,
      and confusion-matrix output.
    """
    metrics: dict[str, Any] = {
        "accuracy": compute_accuracy(records),
    }
    metrics.update(compute_precision_recall_f1(records))
    metrics["per_class_metrics"] = compute_per_class_metrics(records)
    metrics["confusion_matrix"] = compute_confusion_matrix(records)
    return metrics
