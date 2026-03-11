from __future__ import annotations
import json
from typing import Any
from pathlib import Path

from GLM.evaluation.prediction_schema import (
    serialize_prediction_record,
    validate_prediction_record,
)


def build_rank_predictions_path(
        output_dir: str | Path,
        *,
        split: str,
        rank: int,
) -> Path:
    """
    Build the output path for one rank-local prediction file.

    Parameters
    ----------
    output_dir: str | Path
        Base directory where evaluation artifacts are written.
    split: str
        Evaluated dataset split name (for example val or test).
    rank: int
        Distributed process rank.

    Returns
    -------
    Path
        Absolute path for the rank-local predictions JSONL file.

    Notes
    -----
    - This path should be deterministic and stable across repeated runs.
    - Parent directories should be created by the caller or within this helper.
    """
    base_dir = Path(output_dir).expanduser().resolve()
    if not isinstance(split, str) or not split.strip():
        raise ValueError("`split` must be a non-empty string")
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("`rank` must be a non-negative integer")

    predictions_dir = base_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return predictions_dir / f"{split.strip().lower()}_rank_{rank}.jsonl"


def write_prediction_records(
        records: list[dict[str, Any]],
        output_path: str | Path,
) -> Path:
    """
    Write prediction records to a JSONL file.

    Parameters
    ----------
    records: list[dict[str, Any]]
        List of validated and serializable prediction records.
    output_path: str | Path
        Destination JSONL file path.

    Returns
    -------
    Path
        Resolved path to the written JSONL file.

    Notes
    -----
    - Write one record per line for streaming-friendly downstream processing.
    - Validate or serialize records before writing to avoid malformed output.
    """
    if not isinstance(records, list):
        raise TypeError(f"`records` must be list got {type(records).__name__}")

    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if not isinstance(record, dict):
                raise TypeError(f"Each prediction record must be dict got {type(record).__name__}")
            validate_prediction_record(record)
            serialized = serialize_prediction_record(record)
            handle.write(json.dumps(serialized, ensure_ascii=True) + "\n")

    return resolved_path


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    """
    Load prediction records from a JSONL file.

    Parameters
    ----------
    path: str | Path
        Path to a JSONL file containing serialized prediction records.

    Returns
    -------
    list[dict[str, Any]]
        List of loaded prediction records.

    Notes
    -----
    - Raise a clear error for missing files or malformed JSON lines.
    - Use prediction-schema validation after loading when strictness is required.
    """
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Prediction record file not found: {resolved_path}")

    records: list[dict[str, Any]] = []
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in prediction record file '{resolved_path}' at line {line_no}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"Prediction record at line {line_no} in '{resolved_path}' must be dict, "
                    f"got {type(record).__name__}"
                )
            validate_prediction_record(record)
            records.append(record)

    return records


def collect_rank_prediction_files(
        output_dir : str | Path,
        *,
        split: str,
) -> list[Path]:
    """
    Collect all rank-local prediction files for one evaluation split.

    Parameters
    ----------
    output_dir: str | Path
        Base directory where evaluation artifacts are written.
    split: str
        Evaluated dataset split name.

    Returns
    -------
    list[Path]
        Sorted list of rank-local prediction file paths.

    Notes
    -----
    - Sorting should be deterministic to keep merged outputs reproducible.
    - This helper is typically used on rank 0 after distributed evaluation completes.
    """
    base_dir = Path(output_dir).expanduser().resolve()
    if not isinstance(split, str) or not split.strip():
        raise ValueError("`split` must be a non-empty string")

    predictions_dir = base_dir / "predictions"
    if not predictions_dir.exists():
        return []

    pattern = f"{split.strip().lower()}_rank_*.jsonl"
    return sorted(predictions_dir.glob(pattern))


def merge_prediction_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    """
    Merge prediction records from multiple rank-local JSONL files.

    Parameters
    ----------
    paths: list[str | Path]
        Rank-local prediction file paths.

    Returns
    -------
    list[dict[str, Any]]
        Combined list of prediction records across all ranks.

    Notes
    -----
    - Preserve deterministic ordering where possible.
    - Validate records before returning merged output.
    """
    if not isinstance(paths, list):
        raise TypeError(f"`paths` must be list got {type(paths).__name__}")

    merged_records: list[dict[str, Any]] = []
    for path in sorted(Path(p).expanduser().resolve() for p in paths):
        merged_records.extend(load_prediction_records(path))

    return merged_records


def save_metrics_report(
        metrics: dict[str, Any],
        output_dir: str | Path,
        *,
        split: str,
        run_name: str | None = None,
) -> Path:
    """
    Save final evaluation metrics to a JSON report file.

    Parameters
    ----------
    metrics: dict[str, Any]
        Final aggregated evaluation metrics.
    output_dir: str | Path
        Base directory where evaluation artifacts are written.
    split: str
        Evaluated dataset split name.
    run_name: str | None
        Optional run name appended to the output filename.

    Returns
    -------
    Path
        Resolved path to the saved metrics report.

    Notes
    -----
    - Metrics output should be stable, human-readable, and machine-consumable.
    - Write only JSON-safe values in the final report.
    """
    if not isinstance(metrics, dict):
        raise TypeError(f"`metrics` must be dict got {type(metrics).__name__}")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("`split` must be a non-empty string")
    if run_name is not None and (not isinstance(run_name, str) or not run_name.strip()):
        raise ValueError("`run_name` must be a non-empty string when provided")

    base_dir = Path(output_dir).expanduser().resolve()
    metrics_dir = base_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{split.strip().lower()}_metrics.json"
    if run_name is not None:
        file_name = f"{split.strip().lower()}_{run_name.strip()}_metrics.json"

    output_path = metrics_dir / file_name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")

    return output_path


def save_error_analysis(
        analysis: dict[str, Any],
        output_dir: str | Path,
        *,
        split: str,
        run_name: str | None = None,
) -> Path:
    """
    Save structured error-analysis output to disk.

    Parameters
    ----------
    analysis: dict[str, Any]
        Error-analysis summary or sliced breakdowns.
    output_dir: str | Path
        Base directory where evaluation artifacts are written.
    split: str
        Evaluated dataset split name.
    run_name: str | None
        Optional run name appended to the output filename.

    Returns
    -------
    Path
        Resolved path to the saved error-analysis file.

    Notes
    -----
    - Keep this output separate from the core metrics report.
    - Use JSON so downstream inspection and tooling stay simple.
    """
    if not isinstance(analysis, dict):
        raise TypeError(f"`analysis` must be dict got {type(analysis).__name__}")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("`split` must be a non-empty string")
    if run_name is not None and (not isinstance(run_name, str) or not run_name.strip()):
        raise ValueError("`run_name` must be a non-empty string when provided")

    base_dir = Path(output_dir).expanduser().resolve()
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{split.strip().lower()}_error_analysis.json"
    if run_name is not None:
        file_name = f"{split.strip().lower()}_{run_name.strip()}_error_analysis.json"

    output_path = analysis_dir / file_name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")

    return output_path
