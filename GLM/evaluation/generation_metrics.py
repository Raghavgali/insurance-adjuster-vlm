from __future__ import annotations

from typing import Any

from GLM.evaluation.prediction_schema import normalize_text, validate_prediction_record


def _extract_text_pairs(records: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Extract raw and normalized prediction/reference text pairs from records."""
    if not isinstance(records, list):
        raise TypeError(f"`records` must be list got {type(records).__name__}")
    if not records:
        raise ValueError("`records` must be a non-empty list")

    predictions: list[str] = []
    references: list[str] = []
    normalized_predictions: list[str] = []
    normalized_references: list[str] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Prediction record at index {idx} must be dict got {type(record).__name__}")
        validate_prediction_record(record)

        prediction_text = record["prediction_text"]
        reference_text = record["reference_text"]

        predictions.append(prediction_text)
        references.append(reference_text)
        normalized_predictions.append(normalize_text(prediction_text))
        normalized_references.append(normalize_text(reference_text))

    return predictions, references, normalized_predictions, normalized_references


def compute_exact_match(records: list[dict[str, Any]]) -> float:
    """
    Compute raw exact-match score over prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing raw prediction and reference text.

    Returns
    -------
    float
        Exact-match score in the range [0.0, 1.0].

    Notes
    -----
    - Uses raw text without normalization.
    - Intended as a strict string-match metric.
    """
    predictions, references, _, _ = _extract_text_pairs(records)
    matches = sum(int(pred == ref) for pred, ref in zip(predictions, references))
    return matches / len(predictions)


def compute_normalized_exact_match(records: list[dict[str, Any]]) -> float:
    """
    Compute normalized exact-match score over prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing raw prediction and reference text.

    Returns
    -------
    float
        Normalized exact-match score in the range [0.0, 1.0].

    Notes
    -----
    - Applies canonical text normalization before comparison.
    - More appropriate than raw exact match for free-form generation.
    """
    _, _, normalized_predictions, normalized_references = _extract_text_pairs(records)
    matches = sum(int(pred == ref) for pred, ref in zip(normalized_predictions, normalized_references))
    return matches / len(normalized_predictions)


def compute_bleu(records: list[dict[str, Any]]) -> float:
    """
    Compute corpus BLEU score from prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing raw prediction and reference text.

    Returns
    -------
    float
        Corpus BLEU score in the range [0.0, 1.0].

    Notes
    -----
    - Uses normalized whitespace/casing before scoring.
    - Relies on the `evaluate` package.
    """
    try:
        import evaluate as hf_evaluate
    except ImportError as exc:
        raise RuntimeError("BLEU computation requires the `evaluate` package.") from exc

    _, _, normalized_predictions, normalized_references = _extract_text_pairs(records)
    bleu_metric = hf_evaluate.load("bleu")
    result = bleu_metric.compute(
        predictions=normalized_predictions,
        references=[[ref] for ref in normalized_references],
    )
    return float(result["bleu"])


def compute_rouge(records: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute ROUGE scores from prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing raw prediction and reference text.

    Returns
    -------
    dict[str, float]
        ROUGE scores including rouge1, rouge2, rougeL, and rougeLsum when available.

    Notes
    -----
    - Uses normalized whitespace/casing before scoring.
    - Relies on the `evaluate` package.
    """
    try:
        import evaluate as hf_evaluate
    except ImportError as exc:
        raise RuntimeError("ROUGE computation requires the `evaluate` package.") from exc

    _, _, normalized_predictions, normalized_references = _extract_text_pairs(records)
    rouge_metric = hf_evaluate.load("rouge")
    result = rouge_metric.compute(
        predictions=normalized_predictions,
        references=normalized_references,
        use_stemmer=True,
    )
    return {key: float(value) for key, value in result.items()}


def compute_meteor(records: list[dict[str, Any]]) -> float:
    """
    Compute METEOR score from prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing raw prediction and reference text.

    Returns
    -------
    float
        Corpus METEOR score in the range [0.0, 1.0].

    Notes
    -----
    - Uses normalized whitespace/casing before scoring.
    - Relies on the `evaluate` package.
    """
    try:
        import evaluate as hf_evaluate
    except ImportError as exc:
        raise RuntimeError("METEOR computation requires the `evaluate` package.") from exc

    _, _, normalized_predictions, normalized_references = _extract_text_pairs(records)
    meteor_metric = hf_evaluate.load("meteor")
    result = meteor_metric.compute(
        predictions=normalized_predictions,
        references=normalized_references,
    )
    return float(result["meteor"])


def compute_generation_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute the full generation-metric suite for prediction records.

    Parameters
    ----------
    records: list[dict[str, Any]]
        Prediction records containing prediction/reference text.

    Returns
    -------
    dict[str, float]
        Combined generation metrics.

    Notes
    -----
    - Runs exact-match and normalized exact-match locally.
    - Computes BLEU, ROUGE, and METEOR through metric backends.
    """
    metrics: dict[str, float] = {
        "exact_match": compute_exact_match(records),
        "normalized_exact_match": compute_normalized_exact_match(records),
        "bleu": compute_bleu(records),
        "meteor": compute_meteor(records),
    }
    metrics.update(compute_rouge(records))
    return metrics
