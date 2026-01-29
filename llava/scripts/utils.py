from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, List

import yaml
from huggingface_hub import login as hf_login
from lightning.pytorch.callbacks import Callback, EarlyStopping

# Damage Report Metrics 
import evaluate


def load_yaml_config(config_path: str | os.PathLike) -> dict:
    """Load a YAML configuration file."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found at {path}")

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_hf_login(token: Optional[str]) -> bool:
    """Authenticate with the Hugging Face Hub when a token is provided."""
    resolved_token = token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if resolved_token:
        hf_login(token=resolved_token, add_to_git_credential=True)
        return True
    return False


class PushToHubCallback(Callback):
    def __init__(self, repo_id: str, push_every_epoch: bool = True) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.push_every_epoch = push_every_epoch

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not self.push_every_epoch:
            return
        pl_module.model.push_to_hub(
            self.repo_id,
            commit_message=f"Training in progress, epoch {trainer.current_epoch}",
        )

    def on_train_end(self, trainer, pl_module) -> None:
        pl_module.processor.push_to_hub(self.repo_id, commit_message="Training done")
        pl_module.model.push_to_hub(self.repo_id, commit_message="Training done")


early_stop_callback = EarlyStopping(monitor="val_bertscore_f1", patience=3, verbose=False, mode="max")


def setup_logger(name: str = "llava_trainer", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    def _same_handler(existing: logging.Handler) -> bool:
        if isinstance(existing, logging.FileHandler) and isinstance(handler, logging.FileHandler):
            return getattr(existing, "baseFilename", None) == getattr(handler, "baseFilename", None)
        return isinstance(existing, type(handler))

    if not any(_same_handler(existing) for existing in logger.handlers):
        logger.addHandler(handler)

    return logger


class DamageReportMetrics:
    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')

    def compute(self, predictions: List[str], references: List[str]):
        # Rouge for structural similarity (damage type mentions, etc)
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )

        # BertScore for semantic similarity (different phrasing, same meaning)
        bert_scores = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang='en',
            model_type='microsoft/deberta-base-mnli'
        )

        return {
            'rouge1': rouge_scores['rouge1'],
            'rougeL': rouge_scores['rougeL'],
            'bertscore_f1': sum(bert_scores['f1']) / len(bert_scores['f1'])
        }