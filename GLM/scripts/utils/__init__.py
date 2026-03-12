"""Shared utility helpers for config loading, logging, HF access, and tracking."""

from .load_config import load_yaml_config
from .logging import setup_logger
from .wandb import (
    finish_wandb_run,
    init_wandb_run,
    is_wandb_enabled,
    log_wandb_artifact,
    log_wandb_metrics,
    update_wandb_summary,
)

__all__ = [
    "load_yaml_config",
    "setup_logger",
    "finish_wandb_run",
    "init_wandb_run",
    "is_wandb_enabled",
    "log_wandb_artifact",
    "log_wandb_metrics",
    "update_wandb_summary",
    "download_dataset_snapshot",
    "ensure_hf_login",
    "resolve_hf_token",
    "upload_model_folder",
]


def __getattr__(name: str):
    if name in {
        "download_dataset_snapshot",
        "ensure_hf_login",
        "resolve_hf_token",
        "upload_model_folder",
    }:
        from . import hf_utils

        return getattr(hf_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
