from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from wandb import login as wandb_login

from llava.scripts.lightning_module import LlavaPLModule
from llava.scripts.model_loader import (
    QuantizationConfig,
    apply_lora,
    find_all_linear_names,
    load_llava_model,
    load_llava_processor,
)
from llava.scripts.prepare_data import LlavaNextDatacollator, LlavaNextDataset
from llava.scripts.utils import (
    ensure_hf_login,
    early_stop_callback,
    load_yaml_config,
    setup_logger,
    PushToHubCallback,
)
from peft import LoraConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA NeXT for Insurance Adjuster domain.")
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    parser.add_argument("--config", type=Path, default=default_config, help="Path to YAML config file.")
    parser.add_argument("--devices", default="auto", help="Devices argument for PyTorch Lightning Trainer.")
    parser.add_argument("--precision", default="16-mixed", help="Numerical precision setting.")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (defaults to CPU count - 1).")
    parser.add_argument("--limit_val_batches", type=int, default=None, help="Limit validation batches for faster iterations.")
    return parser.parse_args()


def resolve_path(path_str: str | None, base_dir: Path) -> Path:
    if not path_str:
        raise ValueError("Expected a path value in the configuration, but found None or an empty string.")
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def maybe_login_wandb(api_key: str | None) -> bool:
    key = api_key or os.getenv("WANDB_API_KEY")
    if not key:
        return False
    wandb_login(key=key)
    return True


def build_lora_config(config_section: Dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=int(config_section.get("r", 8)),
        lora_alpha=int(config_section.get("lora_alpha", 16)),
        lora_dropout=float(config_section.get("lora_dropout", 0.05)),
        bias=config_section.get("bias", "none"),
        target_modules=[],
        init_lora_weights=config_section.get("init_lora_weights", "gaussian"),
    )


def create_dataloader(dataset: LlavaNextDataset, processor, batch_size: int, max_length: int, is_train: bool, num_workers: int) -> DataLoader:
    collator = LlavaNextDatacollator(processor=processor, max_length=max_length, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def run_training(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser().resolve()
    config = load_yaml_config(config_path)
    base_dir = config_path.parent

    torch.set_float32_matmul_precision("high")

    paths_cfg = config.get("paths", {})
    train_json = resolve_path(paths_cfg.get("train_json"), base_dir)
    train_images = resolve_path(paths_cfg.get("train_image_folder"), base_dir)
    val_json = resolve_path(paths_cfg.get("test_json"), base_dir)
    val_images = resolve_path(paths_cfg.get("test_image_folder"), base_dir)

    model_cfg = config.get("model", {})
    model_id = model_cfg.get("model_id")
    repo_id = model_cfg.get("repo_id")
    max_length = int(model_cfg.get("max_length", 256))
    wandb_project = model_cfg.get("wand_project")
    wandb_run_name = model_cfg.get("wandb_name")

    api_cfg = config.get("api_keys", {})
    hf_logged_in = ensure_hf_login(api_cfg.get("hugging_face"))
    use_wandb = maybe_login_wandb(api_cfg.get("wandb"))

    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 1))
    cpu_count = os.cpu_count() or 1
    num_workers = args.num_workers if args.num_workers is not None else max(cpu_count - 1, 0)
    trainer_kwargs = dict(
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=args.devices,
        max_epochs=int(training_cfg.get("max_epochs", 1)),
        accumulate_grad_batches=int(training_cfg.get("accumulate_grad_batches", 1)),
        check_val_every_n_epoch=int(training_cfg.get("check_val_every_n_epoch", 1)),
        gradient_clip_val=float(training_cfg.get("gradient_clip_val", 0.0)),
        precision=args.precision,
        num_sanity_val_steps=0,
        callbacks=[early_stop_callback],
    )

    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    log_file = config.get("logging", {}).get("log_file")
    log_path = resolve_path(log_file, base_dir) if log_file else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(log_path) if log_path else None)
    logger.info("Configuration loaded from %s", config_path)
    logger.info("Training data: %s", train_json)
    logger.info("Validation data: %s", val_json)

    quant_cfg = config.get("quantization", {})
    quantization = QuantizationConfig(
        use_4bit=bool(quant_cfg.get("use_4bit", True)),
        use_8bit=bool(quant_cfg.get("use_8bit", False)),
    )
    if not torch.cuda.is_available() and quantization.use_4bit:
        logger.warning("4-bit quantization requires CUDA; falling back to full precision on CPU.")
        quantization.use_4bit = False

    if not model_id:
        raise ValueError("`model.model_id` must be provided in the config.")

    processor = load_llava_processor(model_id)
    model = load_llava_model(model_id, quantization=quantization)

    lora_cfg_section = config.get("lora", {})
    use_lora = str(lora_cfg_section.get("USE_LORA", "false")).lower() == "true"
    if use_lora:
        lora_config = build_lora_config(lora_cfg_section)
        lora_config.target_modules = find_all_linear_names(model)
        model = apply_lora(model, lora_config)

    train_dataset = LlavaNextDataset(train_json, train_images, split="train")
    val_dataset = LlavaNextDataset(val_json, val_images, split="test")

    train_loader = create_dataloader(train_dataset, processor, batch_size, max_length, is_train=True, num_workers=num_workers)
    val_loader = create_dataloader(val_dataset, processor, batch_size, max_length, is_train=False, num_workers=max(num_workers // 2, 0))

    module_config = dict(training_cfg)
    module_config["lr"] = float(module_config.get("lr", 1e-4))

    lightning_module = LlavaPLModule(
        config=module_config,
        processor=processor,
        model=model,
        max_length=max_length,
    )

    callbacks = trainer_kwargs.pop("callbacks")
    if repo_id and hf_logged_in:
        callbacks.append(PushToHubCallback(repo_id=repo_id))
    elif repo_id:
        logger.info("Skipping push to Hub because no Hugging Face token was provided.")
    trainer_kwargs["callbacks"] = callbacks

    wandb_logger = WandbLogger(project=wandb_project, name=wandb_run_name) if use_wandb and wandb_project else None
    if wandb_logger:
        trainer_kwargs["logger"] = wandb_logger
    else:
        trainer_kwargs["logger"] = True
        if not use_wandb:
            logger.info("W&B credentials not provided. Running without experiment tracking.")

    trainer = L.Trainer(**trainer_kwargs)

    logger.info("Starting training")
    try:
        trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except Exception as exc:  
        logger.exception("Training failed due to an error: %s", exc)
        raise
    finally:
        logger.info("Training finished")


if __name__ == "__main__":
    run_training(parse_args())
