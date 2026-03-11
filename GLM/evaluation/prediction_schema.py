from __future__ import annotations
import re
from typing import Any


def build_prediction_record(
        *,
        sample_id: str,
        prediction_text: str,
        reference_text: str,
        metadata: dict[str, Any] | None = None,
        predicted_cost: float | None = None,
        reference_cost: float | None = None,
        predicted_label: str | None = None,
        reference_label: str | None = None,
) -> dict[str, Any]:
    """
    Build one normalized evaluation prediction record.

    Parameters
    ----------
    sample_id: str
        Unique sample identifier.
    prediction_text: str
        Model-generated output text.
    reference_text: str
        Ground-truth reference text for the sample.
    metadata: dict[str, Any] | None
        Optional structured metadata attached for evaluation and error analysis.
    predicted_cost: float | None
        Optional parsed numeric cost prediction.
    reference_cost: float | None
        Optional parsed numeric ground-truth cost target.
    predicted_label: str | None
        Optional predicted class label derived from model output.
    reference_label: str | None
        Optional ground-truth class label.

    Returns
    -------
    dict[str, Any]
        Normalized evaluation record with a stable schema.

    Notes
    -----
    - Keep this function side-effect free.
    - Only include optional fields when they are supported by the task.
    - Output should be safe to pass into metric and IO helpers.
    """
    if not isinstance(sample_id, str):
        raise TypeError(f"`sample_id` must be string got {type(sample_id).__name__}")
    if not isinstance(prediction_text, str):
        raise TypeError(f"`prediction_text` must be string got {type(prediction_text).__name__}")
    if not isinstance(reference_text, str):
        raise TypeError(f"`reference_text` must be string got {type(reference_text).__name__}")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError(f"`metadata` must be dict or None got {type(metadata).__name__}")
    if predicted_cost is not None and not isinstance(predicted_cost, (int, float)):
        raise TypeError(f"`predicted_cost` must be numeric or None got {type(predicted_cost).__name__}")
    if reference_cost is not None and not isinstance(reference_cost, (int, float)):
        raise TypeError(f"`reference_cost` must be numeric or None got {type(reference_cost).__name__}")
    if predicted_label is not None and not isinstance(predicted_label, str):
        raise TypeError(f"`predicted_label` must be string or None got {type(predicted_label).__name__}")
    if reference_label is not None and not isinstance(reference_label, str):
        raise TypeError(f"`reference_label` must be string or None got {type(reference_label).__name__}")

    sample_id = sample_id.strip()
    if not sample_id:
        raise ValueError("`sample_id` must be a non-empty string")
    
    prediction_text = prediction_text.strip()
    if not prediction_text:
        raise ValueError("`prediction_text` must be a non-empty string")
    
    reference_text = reference_text.strip()
    if not reference_text:
        raise ValueError("`reference_text` must be a non-empty string")

    if predicted_label is not None:
        predicted_label = predicted_label.strip()
        if not predicted_label:
            raise ValueError("`predicted_label` must be non-empty when provided")
    if reference_label is not None:
        reference_label = reference_label.strip()
        if not reference_label:
            raise ValueError("`reference_label` must be non-empty when provided")

    prediction_record: dict[str, Any] = {
        "id": sample_id,
        "prediction_text": prediction_text,
        "reference_text": reference_text,
        "metadata": metadata or {},
        "predicted_cost": float(predicted_cost) if predicted_cost is not None else None,
        "reference_cost": float(reference_cost) if reference_cost is not None else None,
        "predicted_label": predicted_label,
        "reference_label": reference_label,
    }

    validate_prediction_record(prediction_record)
    return prediction_record


def validate_prediction_record(record: dict[str, Any]) -> None:
    """
    Validate that a prediction record follows the expected schema.

    Parameters
    ----------
    record: dict[str, Any]
        One normalized prediction record.

    Returns
    -------
    None

    Notes
    -----
    - Raise clear errors for missing required keys, invalid types, or empty values.
    - Validation should be strict enough to catch bad records before metric computation.
    """
    if not isinstance(record, dict):
        raise TypeError(f"`record` must be dict got {type(record).__name__}")
    if not record:
        raise ValueError("`record` must be a non-empty dict")

    required_keys = {"id", "prediction_text", "reference_text", "metadata"}
    missing_keys = [key for key in required_keys if key not in record]
    if missing_keys:
        raise KeyError(f"Prediction record missing required keys: {missing_keys}")

    record_id = record.get("id")
    prediction_text = record.get("prediction_text")
    reference_text = record.get("reference_text")
    metadata = record.get("metadata")
    predicted_cost = record.get("predicted_cost")
    reference_cost = record.get("reference_cost")
    predicted_label = record.get("predicted_label")
    reference_label = record.get("reference_label")

    if not isinstance(record_id, str):
        raise TypeError(f"`id` must be string got {type(record_id).__name__}")
    if not isinstance(prediction_text, str):
        raise TypeError(f"`prediction_text` must be string got {type(prediction_text).__name__}")
    if not isinstance(reference_text, str):
        raise TypeError(f"`reference_text` must be string got {type(reference_text).__name__}")
    if not isinstance(metadata, dict):
        raise TypeError(f"`metadata` must be dict got {type(metadata).__name__}")
    if predicted_cost is not None and not isinstance(predicted_cost, (int, float)):
        raise TypeError(f"`predicted_cost` must be numeric or None got {type(predicted_cost).__name__}")
    if reference_cost is not None and not isinstance(reference_cost, (int, float)):
        raise TypeError(f"`reference_cost` must be numeric or None got {type(reference_cost).__name__}")
    if predicted_label is not None and not isinstance(predicted_label, str):
        raise TypeError(f"`predicted_label` must be string or None got {type(predicted_label).__name__}")
    if reference_label is not None and not isinstance(reference_label, str):
        raise TypeError(f"`reference_label` must be string or None got {type(reference_label).__name__}")

    record_id = record_id.strip()
    if not record_id:
        raise ValueError("`id` must be a non-empty string")

    prediction_text = prediction_text.strip()
    if not prediction_text:
        raise ValueError("`prediction_text` must be a non-empty string")

    reference_text = reference_text.strip()
    if not reference_text:
        raise ValueError("`reference_text` must be a non-empty string")

    if predicted_label is not None and not predicted_label.strip():
        raise ValueError("`predicted_label` must be non-empty when provided")
    if reference_label is not None and not reference_label.strip():
        raise ValueError("`reference_label` must be non-empty when provided")


def normalize_text(text: str) -> str:
    """
    Normalize raw text into a canonical form for downstream evaluation.

    Parameters
    ----------
    text: str
        Raw text string to normalize.

    Returns
    -------
    str
        Normalized text string.

    Notes
    -----
    - Keep normalization deterministic and lightweight.
    - Use this for exact-match style metrics, not for destructive semantic rewriting.
    """
    if not isinstance(text, str):
        raise TypeError(f"`text` must be string got {type(text).__name__}")

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_reference_text(sample: dict[str, Any]) -> str:
    """
    Extract the assistant/reference text from one normalized dataset sample.

    Parameters
    ----------
    sample: dict[str, Any]
        One normalized dataset sample.

    Returns
    -------
    str
        Reference target text for evaluation.

    Notes
    -----
    - Assumes dataset samples follow the schema produced by dataset.py.
    - Raise a clear error if the assistant/reference turn cannot be found.
    """
    if not isinstance(sample, dict):
        raise TypeError(f"`sample` must be dict got {type(sample).__name__}")

    conversations = sample.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("`sample['conversations']` must be a non-empty list")

    for turn in reversed(conversations):
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role == "assistant" and isinstance(content, str):
            content = content.strip()
            if content:
                return content

    raise ValueError("Could not find non-empty assistant reference text in sample")


def build_metadata_view(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Build a compact metadata dictionary for evaluation and error analysis.

    Parameters
    ----------
    sample: dict[str, Any]
        One normalized dataset sample, optionally enriched with metadata.

    Returns
    -------
    dict[str, Any]
        Metadata subset attached to the evaluation record.

    Notes
    -----
    - Include only fields needed for slicing, analysis, or auxiliary metrics.
    - Avoid copying large raw structures into the record unnecessarily.
    """
    if not isinstance(sample, dict):
        raise TypeError(f"`sample` must be dict got {type(sample).__name__}")

    metadata = sample.get("metadata")
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError(f"`sample['metadata']` must be dict got {type(metadata).__name__}")

    allowed_keys = {
        "shooting_angle",
        "view",
        "color",
        "damage_category",
        "area",
        "bbox",
        "iscrowd",
        "file_name",
    }
    return {key: metadata[key] for key in allowed_keys if key in metadata}


def serialize_prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a prediction record into a JSON-safe structure for persistence.

    Parameters
    ----------
    record: dict[str, Any]
        One validated prediction record.

    Returns
    -------
    dict[str, Any]
        JSON-serializable prediction record.

    Notes
    -----
    - Ensure values are safe for JSONL output.
    - Preserve schema consistency between in-memory and persisted records.
    """
    validate_prediction_record(record)

    serialized: dict[str, Any] = {
        "id": str(record["id"]),
        "prediction_text": str(record["prediction_text"]),
        "reference_text": str(record["reference_text"]),
        "metadata": dict(record["metadata"]),
        "predicted_cost": float(record["predicted_cost"]) if record.get("predicted_cost") is not None else None,
        "reference_cost": float(record["reference_cost"]) if record.get("reference_cost") is not None else None,
        "predicted_label": str(record["predicted_label"]) if record.get("predicted_label") is not None else None,
        "reference_label": str(record["reference_label"]) if record.get("reference_label") is not None else None,
    }
    return serialized
