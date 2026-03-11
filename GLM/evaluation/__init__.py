"""Evaluation schemas, metrics, and IO helpers for GLM runs."""

from .classification_metrics import compute_classification_metrics
from .generation_metrics import compute_generation_metrics
from .io import (
    build_rank_predictions_path,
    collect_rank_prediction_files,
    load_prediction_records,
    merge_prediction_records,
    save_error_analysis,
    save_metrics_report,
    write_prediction_records,
)
from .prediction_schema import (
    build_metadata_view,
    build_prediction_record,
    extract_reference_text,
    normalize_text,
    serialize_prediction_record,
    validate_prediction_record,
)
from .regression_metrics import compute_regression_metrics

__all__ = [
    "build_rank_predictions_path",
    "collect_rank_prediction_files",
    "load_prediction_records",
    "merge_prediction_records",
    "save_error_analysis",
    "save_metrics_report",
    "write_prediction_records",
    "build_metadata_view",
    "build_prediction_record",
    "extract_reference_text",
    "normalize_text",
    "serialize_prediction_record",
    "validate_prediction_record",
    "compute_generation_metrics",
    "compute_regression_metrics",
    "compute_classification_metrics",
]
