from __future__ import annotations

from pathlib import Path
from typing import Any


def is_wandb_enabled(config: dict[str, Any]) -> bool:
    """
    Determine whether Weights & Biases tracking is enabled in the current config.

    Parameters
    ----------
    config: dict[str, Any]
        Normalized runtime configuration dictionary.

    Returns
    -------
    bool
        True when WandB tracking is explicitly enabled, otherwise False.

    Notes
    -----
    - Missing WandB configuration is treated as disabled.
    - This helper does not import the `wandb` package.
    """
    if not isinstance(config, dict):
        raise TypeError(f"`config` must be dict got {type(config).__name__}")

    wandb_cfg = config.get("wandb", {})
    if not isinstance(wandb_cfg, dict):
        raise TypeError(f"`config['wandb']` must be dict got {type(wandb_cfg).__name__}")
    return bool(wandb_cfg.get("enabled", False))


def _flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested metric dictionaries into WandB-safe scalar key/value pairs."""
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        metric_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_metrics(value, prefix=metric_key))
        elif isinstance(value, bool):
            flattened[metric_key] = float(value)
        elif isinstance(value, (int, float)):
            flattened[metric_key] = float(value)
    return flattened


def init_wandb_run(
    config: dict[str, Any],
    runtime: dict[str, Any],
    *,
    job_type: str,
) -> Any | None:
    """
    Initialize a WandB run for the current process when tracking is enabled.

    Parameters
    ----------
    config: dict[str, Any]
        Normalized runtime configuration dictionary.
    runtime: dict[str, Any]
        Runtime metadata including process rank and output directory.
    job_type: str
        High-level job label such as `train` or `eval`.

    Returns
    -------
    Any | None
        WandB run object when enabled on the main process, else None.

    Notes
    -----
    - Only rank 0 initializes a run in distributed execution.
    - Raises a clear error if WandB is enabled but the package is unavailable.
    """
    if not isinstance(config, dict):
        raise TypeError(f"`config` must be dict got {type(config).__name__}")
    if not isinstance(runtime, dict):
        raise TypeError(f"`runtime` must be dict got {type(runtime).__name__}")
    if not isinstance(job_type, str) or not job_type.strip():
        raise ValueError("`job_type` must be a non-empty string")

    if not is_wandb_enabled(config) or not runtime.get("is_main_process", True):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("WandB tracking is enabled but the `wandb` package is not installed.") from exc

    wandb_cfg = config.get("wandb", {})
    output_dir = Path(str(config.get("output_dir", "."))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        key: value
        for key, value in config.items()
        if key != "wandb"
    }

    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        group=wandb_cfg.get("group"),
        name=wandb_cfg.get("name", runtime.get("run_name")),
        tags=wandb_cfg.get("tags"),
        notes=wandb_cfg.get("notes"),
        mode=wandb_cfg.get("mode", "online"),
        dir=str(output_dir),
        job_type=job_type.strip(),
        config=config_payload,
        reinit=True,
    )
    return run


def log_wandb_metrics(
    run: Any | None,
    metrics: dict[str, Any],
    *,
    step: int | None = None,
    prefix: str | None = None,
    commit: bool = True,
) -> None:
    """
    Log scalar metrics to an active WandB run.

    Parameters
    ----------
    run: Any | None
        WandB run object returned by `init_wandb_run()`.
    metrics: dict[str, Any]
        Metric payload to log.
    step: int | None
        Optional global step for x-axis alignment.
    prefix: str | None
        Optional namespace prefix such as `train` or `eval`.
    commit: bool
        Whether WandB should commit this log event immediately.

    Returns
    -------
    None

    Notes
    -----
    - Non-scalar nested values are ignored unless they can be flattened into numeric entries.
    - Safely no-ops when no active run is provided.
    """
    if run is None:
        return
    if not isinstance(metrics, dict):
        raise TypeError(f"`metrics` must be dict got {type(metrics).__name__}")

    payload = _flatten_metrics(metrics, prefix=prefix or "")
    if not payload:
        return
    run.log(payload, step=step, commit=commit)


def update_wandb_summary(
    run: Any | None,
    values: dict[str, Any],
    *,
    prefix: str | None = None,
) -> None:
    """
    Update the WandB run summary with final scalar values.

    Parameters
    ----------
    run: Any | None
        WandB run object returned by `init_wandb_run()`.
    values: dict[str, Any]
        Final metric or metadata payload.
    prefix: str | None
        Optional namespace prefix for summary keys.

    Returns
    -------
    None

    Notes
    -----
    - Only scalar values and flattened numeric entries are written to summary.
    - Safely no-ops when no active run is provided.
    """
    if run is None:
        return
    if not isinstance(values, dict):
        raise TypeError(f"`values` must be dict got {type(values).__name__}")

    payload = _flatten_metrics(values, prefix=prefix or "")
    for key, value in payload.items():
        run.summary[key] = value


def log_wandb_artifact(
    run: Any | None,
    path: str | Path,
    *,
    artifact_type: str,
    name: str | None = None,
) -> None:
    """
    Upload a local file or directory as a WandB artifact.

    Parameters
    ----------
    run: Any | None
        WandB run object returned by `init_wandb_run()`.
    path: str | Path
        Local path to the file or directory to upload.
    artifact_type: str
        Artifact type label such as `checkpoint` or `metrics`.
    name: str | None
        Optional explicit artifact name.

    Returns
    -------
    None

    Notes
    -----
    - Safely no-ops when no active run is provided.
    - Raises a clear error if the requested artifact path does not exist.
    """
    if run is None:
        return
    if not isinstance(artifact_type, str) or not artifact_type.strip():
        raise ValueError("`artifact_type` must be a non-empty string")

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Artifact logging requires the `wandb` package.") from exc

    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Artifact path not found: {resolved_path}")

    artifact_name = name or resolved_path.name
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type.strip())
    if resolved_path.is_dir():
        artifact.add_dir(str(resolved_path))
    else:
        artifact.add_file(str(resolved_path))
    run.log_artifact(artifact)


def finish_wandb_run(run: Any | None) -> None:
    """
    Finish an active WandB run cleanly.

    Parameters
    ----------
    run: Any | None
        WandB run object returned by `init_wandb_run()`.

    Returns
    -------
    None

    Notes
    -----
    - Safely no-ops when no active run is provided.
    - Keeps cleanup logic out of caller `finally` blocks.
    """
    if run is None:
        return
    run.finish()
