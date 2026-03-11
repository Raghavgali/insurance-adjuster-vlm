from __future__ import annotations
from contextlib import nullcontext
from typing import Any
from pathlib import Path
import ast
import re

import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from PIL import Image


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluation.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Runtime arguments including config path, checkpoint path, split, output directory,
        and optional overrides (batch size, num workers, precision).

    Notes
    -----
    - Keep arguments minimal and explicit for reproducible evaluation.
    - CLI overrides should take precedence over config values.
    """
    parser = argparse.ArgumentParser(description="Evaluation Endpoint for GLM VLM fine-tuning")

    parser.add_argument("--config",
                        type=Path,
                        required=True,
                        help="Path to evaluation/training config YAML.")
    
    parser.add_argument("--checkpoint",
                        type=Path,
                        required=True,
                        help="Path to model checkpoint (.pt) used for evaluation.")
    parser.add_argument("--split",
                        type=str,
                        default="test",
                        choices=["val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--output-dir",
                        type=Path,
                        required=True,
                        help="Directory where evaluation metrics/results will be saved.")
    
    parser.add_argument("--per-device-batch-size", 
                        type=int,
                        default=None,
                        help="Optional override for per-device evaluation batch size.")
    parser.add_argument("--num-workers",
                        type=int,
                        default=None,
                        help="Optional override for dataloader worker count.")
    parser.add_argument("--precision",
                        type=str,
                        default=None,
                        choices=["fp32", "fp16", "bf16"],
                        help="Optional precision override for evaluation forward pass.")
    
    parser.add_argument("--seed",
                        type=int,
                        default=42, 
                        help="Random seed for deterministic evaluation ordering")
    parser.add_argument("--save-predictions",
                        action="store_true",
                        help="If set, saves per-sample predictions to disk.")
    parser.add_argument("--predictions-file",
                        type=str,
                        default="predictions.jsonl",
                        help="Filename for saved per-sample predictions (under output-dir).")
    
    parser.add_argument("--log-every", type=int, default=50, help="Log progress every N evaluation steps.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bar output.")

    parser.add_argument("--run-name", type=str, default=None, help="Optional run name used in output artifact filenames.")

    args = parser.parse_args()

    return args


def setup_distributed_eval() -> dict[str, int | torch.device | bool]:
    """
    Initialize distributed context for evaluation and resolve process-local device placement.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, int | torch.device | bool]
        Runtime metadata including:
        - is_distributed
        - rank
        - local_rank
        - world_size
        - is_main_process
        - device

    Notes
    -----
    - Must support seamless fallback to single-process mode.
    - Should initialize torch.distributed only when world size > 1.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    is_distributed = world_size > 1
    is_cuda = torch.cuda.is_available()

    if is_distributed and not dist.is_initialized():
        backend = "nccl" if is_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if is_cuda:
        torch.cuda.set_device(local_rank)   
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return {
        "is_distributed": is_distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main_process": rank == 0,
        "device": device,
    }


def build_eval_components(config: dict, runtime: dict) -> dict[str, Any]:
    """
    Build model and data components required for evaluation.

    Parameters
    ----------
    config: dict
        Normalized configuration dictionary.
    runtime: dict
        Distributed/runtime metadata from setup_distributed_eval().

    Returns
    -------
    dict[str, Any]
        Evaluation components including model, processor/tokenizer, dataset, collator,
        and dataloader.

    Notes
    -----
    - Reuse training-side loaders/collator for schema consistency.
    - Ensure eval dataloader does not shuffle samples.
    """
    if not isinstance(config, dict):
        raise TypeError(f"`config` must be dict got {type(config).__name__}")
    if not isinstance(runtime, dict):
        raise TypeError(f"`runtime` must be dict got {type(runtime).__name__}")
    if not config:
        raise ValueError("`config` cannot be empty")
    if not runtime:
        raise ValueError("`runtime` cannot be empty")
    
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DistributedSampler
    
    from GLM.scripts.model_loader import load_model_bundle
    from GLM.data.dataset import build_dataset
    from GLM.scripts.collator import DataCollator
    
    eval_cfg = config.get("eval", config.get("test", {}))
    data_cfg = config.get("data", {})
    paths_cfg = config.get("paths", {})
    model_cfg = config.get("model", {})
    precision_cfg = config.get("precision", {})
    quant_cfg = config.get("quantization", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("train", config.get("training", {}))

    split = str(eval_cfg.get("split", "test")).lower()
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported eval split: {split}")
    
    annotation_path = (
        eval_cfg.get("annotation_path")
        or data_cfg.get(f"{split}_annotation_path")
        or data_cfg.get(f"{split}_json")
        or paths_cfg.get(f"{split}_json")
    )
    image_root = (
        eval_cfg.get("image_root")
        or data_cfg.get(f"{split}_image_root")
        or data_cfg.get(f"{split}_image_folder")
        or paths_cfg.get(f"{split}_image_folder")
    )

    if not annotation_path:
        raise ValueError(f"Missing annotation path for split='{split}'")
    if not image_root:
        raise ValueError(f"Missing image root for split='{split}'")
    
    per_device_batch_size = int(eval_cfg.get("per_device_batch_size", eval_cfg.get("batch_size", 1)))
    num_workers = int(eval_cfg.get("num_workers", 2))
    if per_device_batch_size <= 0:
        raise ValueError("`per_device_batch_size` must be > 0")
    if num_workers < 0:
        raise ValueError("`num_workers` must be >= 0")
    
    rank = int(runtime["rank"])
    local_rank = int(runtime["local_rank"])
    world_size = int(runtime["world_size"])
    device = runtime["device"]
    is_distributed = bool(runtime["is_distributed"])
    
    model_loader_cfg: dict[str, Any] = {
        "model_id": config.get("model_id", model_cfg.get("model_id")),
        "trust_remote_code": bool(config.get("trust_remote_code", model_cfg.get("trust_remote_code", True))),
        "attn_implementation": config.get("attn_implementation", model_cfg.get("attn_implementation")),
        "device_map": config.get("device_map", model_cfg.get("device_map", "auto")),
        "torch_dtype": str(config.get("torch_dtype", precision_cfg.get("torch_dtype", "bfloat16"))).lower(),
        "use_tf32": bool(config.get("use_tf32", precision_cfg.get("use_tf32", True))),
        "quantization_enabled": bool(config.get("quantization_enabled", quant_cfg.get("enabled", False))),
        "quantization_mode": str(config.get("quantization_mode", quant_cfg.get("mode", "8bit"))).lower(),
        "quantization_compute_dtype": str(
            config.get("quantization_compute_dtype", quant_cfg.get("compute_dtype", "bfloat16"))
        ).lower(),
        "quantization_quant_type": str(config.get("quantization_quant_type", quant_cfg.get("quant_type", "nf4"))).lower(),
        "quantization_double_quant": bool(config.get("quantization_double_quant", quant_cfg.get("double_quant", True))),
        "training_gradient_checkpointing": bool(
            config.get("training_gradient_checkpointing", train_cfg.get("gradient_checkpointing", False))
        ),
        "training_use_cache": bool(config.get("training_use_cache", True)),
        "lora_enabled": bool(config.get("lora_enabled", lora_cfg.get("enabled", False))),
        "lora_r": config.get("lora_r", lora_cfg.get("r")),
        "lora_alpha": config.get("lora_alpha", lora_cfg.get("alpha")),
        "lora_dropout": float(config.get("lora_dropout", lora_cfg.get("dropout", 0.05))),
        "lora_bias": str(config.get("lora_bias", lora_cfg.get("bias", "none"))).lower(),
        "lora_task_type": str(config.get("lora_task_type", lora_cfg.get("task_type", "CAUSAL_LM"))),
        "target_modules": config.get("target_modules", lora_cfg.get("target_modules")),
    }

    if not model_loader_cfg["model_id"]:
        raise ValueError("Missing model_id in config")
    if is_distributed and device.type == "cuda":
        model_loader_cfg["device_map"] = {"": local_rank}

    model_bundle = load_model_bundle(config=model_loader_cfg)
    model = model_bundle["model"]
    processor = model_bundle["processor"]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor does not expose tokenizer")

    eval_dataset = build_dataset(
        annotation_path=str(annotation_path),
        image_root=str(image_root),
        split=split,
        strict=False,
    )

    collator_cfg = {
        "max_length": int(eval_cfg.get("max_length", model_cfg.get("max_length", 2048))),
        "ignore_index": int(eval_cfg.get("ignore_index", -100)),
        "padding": eval_cfg.get("padding", "longest"),
        "truncation": bool(eval_cfg.get("truncation", True)),
    }

    eval_collator = DataCollator.from_config(
        processor=processor,
        tokenizer=tokenizer,
        config=collator_cfg,
    )

    eval_sampler = None
    if is_distributed:
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=per_device_batch_size,
        sampler=eval_sampler,
        shuffle=False,
        collate_fn=eval_collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    
    if device.type == "cuda" and not bool(model_loader_cfg.get("quantization_enabled", False)):
        model = model.to(device)
    if is_distributed:
        ddp_device_ids = [local_rank] if device.type == "cuda" else None
        ddp_output_device = local_rank if device.type == "cuda" else None
        model = DDP(model, device_ids=ddp_device_ids, output_device=ddp_output_device)

    eval_state = {
        "split": split,
        "num_samples": len(eval_dataset),
        "num_batches": len(eval_loader),
        "precision": str(eval_cfg.get("precision", "fp16")).lower(),
    }

    return {
        "model_bundle": model_bundle,
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "eval_dataset": eval_dataset,
        "eval_sampler": eval_sampler,
        "eval_collator": eval_collator,
        "eval_dataloader": eval_loader,
        "eval_state": eval_state,
    }


def load_checkpoint_for_eval(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    """
    Load model weights from a checkpoint for evaluation.

    Parameters
    ----------
    model: torch.nn.Module
        Model instance to load weights into.
    checkpoint_path: str | Path
        Path to checkpoint file.
    device: torch.device
        Device used for map_location when loading checkpoint tensors.

    Returns
    -------
    None

    Notes
    -----
    - Load only model state for evaluation (optimizer/scheduler are not required).
    - Support both plain and DDP-wrapped model objects.
    """
    if checkpoint_path is None:
        raise ValueError("`checkpoint_path` cannot be None for evaluation")

    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format in {ckpt_path}: expected dict")

    model_state = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if model_state is None:
        raise ValueError(f"Checkpoint {ckpt_path} missing model state")

    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(model_state, strict=False)


_DAMAGE_LABELS = [
    "glass shatter",
    "lamp broken",
    "scratch",
    "dent",
    "crack",
]


def _extract_prompt_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    """Extract evaluation metadata embedded in the user prompt when structured metadata is absent."""
    if not isinstance(sample, dict):
        return {}

    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        return {}

    user_content = None
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("role") == "user" and isinstance(turn.get("content"), str):
            user_content = turn["content"]
            break

    if not user_content:
        return {}

    patterns = {
        "shooting_angle": r"Shooting Angle:\s*(.+)",
        "view": r"View:\s*(.+)",
        "color": r"Color:\s*(.+)",
        "damage_category": r"Damage Category:\s*(.+)",
        "area": r"Area:\s*([0-9]+(?:\.[0-9]+)?)",
        "bbox": r"BBox:\s*(\[[^\]]+\])",
        "iscrowd": r"IsCrowd:\s*([0-9]+)",
    }

    metadata: dict[str, Any] = {"file_name": Path(str(sample.get("image", ""))).name}
    for key, pattern in patterns.items():
        match = re.search(pattern, user_content, flags=re.IGNORECASE)
        if not match:
            continue

        value = match.group(1).strip().strip('"')
        if key == "area":
            try:
                metadata[key] = float(value)
            except ValueError:
                continue
        elif key == "bbox":
            try:
                parsed_bbox = ast.literal_eval(value)
                if isinstance(parsed_bbox, list):
                    metadata[key] = [float(item) for item in parsed_bbox]
            except (ValueError, SyntaxError, TypeError):
                continue
        elif key == "iscrowd":
            try:
                metadata[key] = int(value)
            except ValueError:
                continue
        else:
            metadata[key] = value.lower()

    return metadata


def _extract_cost_value(text: str) -> float | None:
    """Extract a numeric repair cost from free-form text when present."""
    if not isinstance(text, str):
        return None

    lowered = text.lower()
    if "$" not in text and "cost" not in lowered and "estimate" not in lowered:
        return None

    normalized = text.replace(",", "")
    matches = re.findall(r"\$?\s*([0-9]+(?:\.[0-9]+)?)", normalized)
    values = [float(match) for match in matches]
    if not values:
        return None
    if len(values) >= 2:
        return sum(values[:2]) / 2.0
    return values[0]


def _extract_damage_label(text: str, *, candidates: list[str] | None = None) -> str | None:
    """Extract a damage-category label from free-form text when present."""
    if not isinstance(text, str):
        return None

    normalized = text.strip().lower()
    label_candidates = candidates or _DAMAGE_LABELS
    for label in sorted(label_candidates, key=len, reverse=True):
        if label in normalized:
            return label
    return None


def _build_generation_text(conversations: list[dict[str, Any]], collator: Any, image_path: str | None) -> str:
    """Build prompt-only text for generation without leaking assistant reference text."""
    return collator.build_generation_text(conversations, image_path=image_path)


def run_evaluation(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict,
    runtime: dict,
) -> dict[str, float]:
    """
    Execute evaluation loop and compute aggregate metrics.

    Parameters
    ----------
    model: torch.nn.Module
        Evaluation model (plain or DDP-wrapped).
    eval_loader: torch.utils.data.DataLoader
        Dataloader for evaluation split.
    device: torch.device
        Device used for forward pass.
    config: dict
        Evaluation configuration.
    runtime: dict
        Distributed/runtime metadata.

    Returns
    -------
    dict[str, float]
        Aggregated evaluation metrics (for example loss, token accuracy, sample count).

    Notes
    -----
    - Must run under model.eval() and torch.no_grad().
    - Keep per-step logging light and rank-aware.
    """
    if not isinstance(config, dict):
        raise TypeError(f'`config` must be dict got {type(config).__name__}')
    if not isinstance(runtime, dict):
        raise TypeError(f'`runtime` must be dict got {type(runtime).__name__}') 
    if not config:
        raise ValueError(f"`config` cannot be empty")
    if not runtime:
        raise ValueError(f"`runtime` cannot be empty")
    
    eval_cfg = config.get("eval", config.get("test", {}))
    log_every = int(eval_cfg.get("log_every", 50))
    precision = str(eval_cfg.get("precision", "fp16")).lower()

    from GLM.evaluation.io import build_rank_predictions_path, write_prediction_records
    from GLM.evaluation.prediction_schema import (
        build_metadata_view,
        build_prediction_record,
        extract_reference_text,
    )

    collator = getattr(eval_loader, "collate_fn", None)
    if collator is None:
        raise ValueError("Evaluation dataloader must expose a collate function")
    tokenizer = getattr(collator, "tokenizer", None)
    processor = getattr(collator, "processor", None)
    if tokenizer is None or processor is None:
        raise ValueError("Evaluation collator must expose tokenizer and processor")

    dataset = getattr(eval_loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "samples"):
        raise ValueError("Evaluation dataloader dataset must expose normalized samples")

    id_to_sample = {str(sample["id"]): sample for sample in dataset.samples}
    target_model = model.module if hasattr(model, "module") else model

    total_loss = 0.0
    total_batches = 0.0
    total_examples = 0.0
    token_correct = 0.0
    token_count = 0.0
    prediction_records: list[dict[str, Any]] = []

    use_autocast = device.type == "cuda" and precision in {"fp16", "bf16"}

    def _autocast_context() -> Any:
        if not use_autocast:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
        )

    model.eval()
    with torch.no_grad():
        for step_idx, batch in enumerate(eval_loader, start=1):
            sample_ids = [str(sample_id) for sample_id in batch.get("sample_ids", [])]
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            if "sample_ids" not in batch:
                raise KeyError("Evaluation batch is missing required key 'sample_ids'")

            with _autocast_context():
                outputs = model(**tensor_batch)

            loss = getattr(outputs, "loss", None)
            if loss is None:
                raise ValueError("Model output missing `loss` during evaluation")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite eval loss at step {step_idx}: {float(loss.detach().cpu())}")

            batch_size = float(len(sample_ids))
            total_loss += float(loss.detach().cpu())
            total_batches += 1.0
            total_examples += batch_size

            logits = getattr(outputs, "logits", None)
            labels = tensor_batch.get("labels")
            if logits is not None and labels is not None and logits.ndim == 3 and labels.ndim == 2:
                shifted_logits = logits[:, :-1, :].argmax(dim=-1)
                shifted_labels = labels[:, 1:]
                valid_mask = shifted_labels.ne(-100)
                token_correct += float((shifted_logits[valid_mask] == shifted_labels[valid_mask]).sum().item())
                token_count += float(valid_mask.sum().item())

            raw_samples = [id_to_sample[sample_id] for sample_id in sample_ids]
            generation_texts = [
                _build_generation_text(sample["conversations"], collator, sample["image"])
                for sample in raw_samples
            ]
            generation_images: list[Image.Image] = []
            for sample in raw_samples:
                image_path = sample["image"]
                try:
                    with Image.open(image_path) as img:
                        generation_images.append(img.convert("RGB"))
                except OSError as exc:
                    raise ValueError(f"Failed to load evaluation image '{image_path}': {exc}") from exc

            generation_inputs = processor(
                text=generation_texts,
                images=generation_images,
                padding=collator.padding,
                pad_to_multiple_of=getattr(collator, "pad_to_multiple_of", None),
                truncation=collator.truncation,
                max_length=collator.max_length,
                return_tensors="pt",
            )
            generation_inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in generation_inputs.items()
                if torch.is_tensor(value)
            }
            generation_inputs.pop("token_type_ids", None)

            with _autocast_context():
                generated_ids = target_model.generate(
                    **generation_inputs,
                    max_new_tokens=int(eval_cfg.get("max_new_tokens", 128)),
                    num_beams=int(eval_cfg.get("num_beams", 1)),
                    do_sample=bool(eval_cfg.get("do_sample", False)),
                )

            attention_mask = generation_inputs.get("attention_mask")
            decoded_predictions: list[str] = []
            for row_idx, output_ids in enumerate(generated_ids):
                prompt_len = int(attention_mask[row_idx].sum().item()) if attention_mask is not None else 0
                new_token_ids = output_ids[prompt_len:]
                prediction_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
                if not prediction_text:
                    prediction_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                decoded_predictions.append(prediction_text)

            for sample_id, sample, prediction_text in zip(sample_ids, raw_samples, decoded_predictions):
                reference_text = extract_reference_text(sample)
                metadata = build_metadata_view(sample)
                if not metadata:
                    metadata = _extract_prompt_metadata(sample)

                reference_label = None
                if isinstance(metadata.get("damage_category"), str):
                    reference_label = str(metadata["damage_category"]).strip().lower()
                predicted_label = _extract_damage_label(
                    prediction_text,
                    candidates=[reference_label] if reference_label else None,
                )

                prediction_records.append(
                    build_prediction_record(
                        sample_id=sample_id,
                        prediction_text=prediction_text,
                        reference_text=reference_text,
                        metadata=metadata,
                        predicted_cost=_extract_cost_value(prediction_text),
                        reference_cost=_extract_cost_value(reference_text),
                        predicted_label=predicted_label,
                        reference_label=reference_label,
                    )
                )

            if runtime.get("is_main_process", True) and (step_idx % max(1, log_every) == 0):
                print(f"[eval] step={step_idx}/{len(eval_loader)}")

    prediction_path = build_rank_predictions_path(
        output_dir=config["output_dir"],
        split=str(eval_cfg.get("split", "test")).lower(),
        rank=int(runtime["rank"]),
    )
    write_prediction_records(prediction_records, prediction_path)

    return {
        "loss_sum": total_loss,
        "num_batches": total_batches,
        "num_examples": total_examples,
        "token_correct": token_correct,
        "token_count": token_count,
        "num_prediction_records": float(len(prediction_records)),
    }


def reduce_eval_metrics(metrics: dict[str, float], runtime: dict) -> dict[str, float]:
    """
    Reduce evaluation metrics across distributed ranks.

    Parameters
    ----------
    metrics: dict[str, float]
        Per-rank metric dictionary produced by run_evaluation().
    runtime: dict
        Distributed/runtime metadata.

    Returns
    -------
    dict[str, float]
        Global reduced metric dictionary.

    Notes
    -----
    - In single-process mode, return metrics unchanged.
    - Use all-reduce with consistent key ordering across ranks.
    """
    if not isinstance(metrics, dict):
        raise TypeError(f"`metrics` must be dict got {type(metrics).__name__}")
    if not isinstance(runtime, dict):
        raise TypeError(f"`runtime` must be dict got {type(runtime).__name__}")

    metric_keys = [
        "loss_sum",
        "num_batches",
        "num_examples",
        "token_correct",
        "token_count",
        "num_prediction_records",
    ]
    metric_values = [float(metrics.get(key, 0.0)) for key in metric_keys]

    if runtime.get("is_distributed", False):
        device = runtime["device"]
        tensor = torch.tensor(metric_values, dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        metric_values = tensor.cpu().tolist()

    reduced = dict(zip(metric_keys, metric_values))
    reduced["eval_loss"] = reduced["loss_sum"] / max(1.0, reduced["num_batches"])
    reduced["token_accuracy"] = reduced["token_correct"] / max(1.0, reduced["token_count"])
    return reduced


def save_eval_results(
    metrics: dict[str, Any],
    output_dir: str | Path,
    split: str,
    runtime: dict,
) -> Path | None:
    """
    Save evaluation metrics report to disk.

    Parameters
    ----------
    metrics: dict[str, float]
        Final reduced evaluation metrics.
    output_dir: str | Path
        Directory where result files are written.
    split: str
        Evaluated split name (for example val/test).
    runtime: dict
        Distributed/runtime metadata for rank-aware saving.

    Returns
    -------
    Path | None
        Path to written result file on main process, else None.

    Notes
    -----
    - Save only on main process in distributed runs.
    - Persist results in JSON for downstream analysis and reproducibility.
    """
    if not isinstance(metrics, dict):
        raise TypeError(f"`metrics` must be dict got {type(metrics).__name__}")
    if not isinstance(runtime, dict):
        raise TypeError(f"`runtime` must be dict got {type(runtime).__name__}")
    if not runtime.get("is_main_process", True):
        return None

    from GLM.evaluation.io import save_metrics_report

    return save_metrics_report(
        metrics=metrics,
        output_dir=output_dir,
        split=split,
        run_name=runtime.get("run_name"),
    )


def cleanup_distributed(runtime: dict) -> None:
    """
    Tear down distributed process group resources after evaluation.

    Parameters
    ----------
    runtime: dict
        Distributed/runtime metadata returned by setup_distributed_eval().

    Returns
    -------
    None

    Notes
    -----
    - Safe no-op when not running distributed evaluation.
    - Should be called in finally blocks to avoid leaked process groups.
    """
    if not isinstance(runtime, dict):
        return

    if runtime.get("is_distributed", False) and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    """
    Evaluation entrypoint coordinating setup, loading, execution, and teardown.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - Owns top-level control flow and error handling.
    - Must guarantee distributed cleanup even when evaluation fails.
    """
    from GLM.evaluation.classification_metrics import compute_classification_metrics
    from GLM.evaluation.generation_metrics import compute_generation_metrics
    from GLM.evaluation.io import (
        collect_rank_prediction_files,
        merge_prediction_records,
        write_prediction_records,
    )
    from GLM.evaluation.regression_metrics import compute_regression_metrics
    from GLM.scripts.utils.load_config import load_yaml_config
    from GLM.scripts.utils.wandb import (
        finish_wandb_run,
        init_wandb_run,
        log_wandb_artifact,
        log_wandb_metrics,
        update_wandb_summary,
    )

    args = parse_args()
    runtime = setup_distributed_eval()
    runtime["run_name"] = args.run_name
    wandb_run = None

    try:
        config = load_yaml_config(args.config)
        if not isinstance(config, dict):
            raise TypeError("Config loader must return a dictionary")

        config.setdefault("eval", config.get("test", {}))
        config.setdefault("data", {})
        config.setdefault("paths", {})
        config.setdefault("wandb", {})
        config["output_dir"] = str(args.output_dir)

        eval_cfg = config["eval"]
        eval_cfg["split"] = args.split
        if args.per_device_batch_size is not None:
            eval_cfg["per_device_batch_size"] = args.per_device_batch_size
        if args.num_workers is not None:
            eval_cfg["num_workers"] = args.num_workers
        if args.precision is not None:
            eval_cfg["precision"] = args.precision
        if args.log_every is not None:
            eval_cfg["log_every"] = args.log_every
        eval_cfg["save_predictions"] = bool(args.save_predictions)
        eval_cfg["predictions_file"] = str(args.predictions_file)

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        wandb_run = init_wandb_run(config=config, runtime=runtime, job_type="eval")

        components = build_eval_components(config=config, runtime=runtime)
        model = components["model"]
        eval_loader = components["eval_dataloader"]
        eval_state = components["eval_state"]

        load_checkpoint_for_eval(
            model=model,
            checkpoint_path=args.checkpoint,
            device=runtime["device"],
        )

        raw_metrics = run_evaluation(
            model=model,
            eval_loader=eval_loader,
            device=runtime["device"],
            config=config,
            runtime=runtime,
        )
        reduced_metrics = reduce_eval_metrics(raw_metrics, runtime=runtime)

        if runtime.get("is_distributed", False):
            dist.barrier()

        final_metrics: dict[str, Any] = dict(reduced_metrics)
        split = str(eval_state["split"])

        if runtime.get("is_main_process", True):
            prediction_files = collect_rank_prediction_files(output_dir=output_dir, split=split)
            merged_records = merge_prediction_records([str(path) for path in prediction_files])
            final_metrics["num_prediction_records"] = len(merged_records)
            final_metrics["generation"] = compute_generation_metrics(merged_records)

            if merged_records and all(
                record.get("predicted_cost") is not None and record.get("reference_cost") is not None
                for record in merged_records
            ):
                final_metrics["regression"] = compute_regression_metrics(merged_records)

            if merged_records and all(
                record.get("predicted_label") is not None and record.get("reference_label") is not None
                for record in merged_records
            ):
                final_metrics["classification"] = compute_classification_metrics(merged_records)

            if eval_cfg.get("save_predictions", False):
                merged_prediction_path = output_dir / str(eval_cfg.get("predictions_file", "predictions.jsonl"))
                write_prediction_records(merged_records, merged_prediction_path)

            metrics_path = save_eval_results(
                metrics=final_metrics,
                output_dir=output_dir,
                split=split,
                runtime=runtime,
            )
            log_wandb_metrics(
                wandb_run,
                final_metrics,
                step=int(final_metrics.get("num_examples", 0)),
                prefix="eval",
            )
            update_wandb_summary(wandb_run, final_metrics, prefix="eval")
            if metrics_path is not None and config["wandb"].get("log_metrics_artifact", True):
                log_wandb_artifact(
                    wandb_run,
                    metrics_path,
                    artifact_type="metrics",
                    name=f"{split}-metrics",
                )
            if metrics_path is not None:
                print(f"[eval_done] metrics={metrics_path}")
    finally:
        finish_wandb_run(wandb_run)
        cleanup_distributed(runtime)
