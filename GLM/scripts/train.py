from __future__ import annotations
from typing import Any
from pathlib import Path

import os
import json
import time
import numpy as np
import random
import argparse
import torch 
import torch.distributed as dist
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training endpoint.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace: argparse.Namespace
        argparse.Namespace containing runtime inputs such as config path, resume checkpoint, 
        output directory, profiling toggles, and optional CLI overrides.

    Notes
    -----
    - Keep arguments explicit and minimal for reproducibility.
    - Prefer config-driven defaults with CLI override precedence.
    """
    def _positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("Value must be a positive integer")
        return parsed

    def _non_negative_int(value: str) -> int:
        parsed = int(value)
        if parsed < 0:
            raise argparse.ArgumentTypeError("Value must be a non-negative integer")
        return parsed

    parser = argparse.ArgumentParser(description="Training entrypoint for GLM VLM fine-tuning.")

    parser.add_argument("--config", type=Path, required=True, help="Path to training config YAML.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for logs/checkpoints.")

    parser.add_argument(
        "--resume",
        "--resume-checkpoint",
        dest="resume",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Global random seed override.")
    parser.add_argument("--epochs", type=_positive_int, default=None, help="Number of training epochs.")
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=None,
        help="Maximum optimizer steps. Overrides epoch-based stopping when set.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=_positive_int,
        default=None,
        help="Per-device micro-batch size override.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=_positive_int,
        default=None,
        help="Gradient accumulation steps override.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader worker count override.")
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="Precision override.",
    )
    parser.add_argument("--save-every", type=_positive_int, default=None, help="Checkpoint interval (steps).")
    parser.add_argument("--log-every", type=_positive_int, default=None, help="Logging interval (steps).")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiling during training.")
    parser.add_argument(
        "--profile-steps",
        type=_positive_int,
        default=None,
        help="Number of profiling steps when --profile is enabled.",
    )
    parser.add_argument(
        "--profile-wait",
        type=_non_negative_int,
        default=None,
        help="Profiler schedule wait steps before warmup.",
    )
    parser.add_argument(
        "--profile-warmup",
        type=_non_negative_int,
        default=None,
        help="Profiler warmup steps (not included in final trace metrics).",
    )
    parser.add_argument(
        "--profile-active",
        type=_positive_int,
        default=None,
        help="Profiler active steps captured into trace output.",
    )
    parser.add_argument(
        "--profile-epoch",
        type=_non_negative_int,
        default=None,
        help="Epoch index at which profiler is enabled (0-based).",
    )

    args = parser.parse_args()

    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.lr is not None and args.lr <= 0:
        parser.error("--lr must be > 0")
    return args


def setup_distributed() -> dict[str, int | torch.device | bool]:
    """
    Initialize distributed training context and resolve process-local device placement.

    Parameters
    ----------
    None

    Returns 
    -------
    A dictionary containing distributed runtime metadata, typically:
        - is_distributed: bool
        - rank: int 
        - local_rank: int 
        - world_size: int 
        - device: torch.device

    Notes
    -----
    - Must support seamless fallback to single-GPU or single-process mode.
    - Should initialize torch.distributed only when required by environment varibles.
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


def seed_everything(seed: int, rank: int) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for deterministic and reproducible training behavior.

    Parameters
    ----------
    seed: int 
        Base random seed
    rank: int 
        Process rank used to derive per-rank seed offsets in DDP.

    Returns
    -------
    None

    Notes
    -----
    - Use rank-adjusted seeds to avoid identical data order/augmentations across ranks.
    - Keep cudnn/deterministric flags aligned with performance vs reproducibility policy.
    """
    if not isinstance(seed, int):
        raise TypeError(f"`seed` must be int, got {type(seed).__name__}")
    if not isinstance(rank, int):
        raise TypeError(f"`rank` must be int, got {type(rank).__name__}")
    
    final_seed = seed + rank

    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(final_seed)
        torch.cuda.manual_seed_all(final_seed)

    # Reproducibility-first defaults (Relaxed for max speed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_profiler(
    output_dir: Path,
    rank: int,
    *,
    enabled: bool = True,
    wait: int = 2,
    warmup: int = 2,
    active: int = 10,
) -> Any | None:
    """
    Build an optional torch profiler instance for rank-local trace capture.

    Parameters
    ----------
    output_dir: Path
        Base output directory used to store profiler traces.
    rank: int
        Process rank used to isolate per-rank trace outputs.
    enabled: bool, default=True
        Whether to construct and return a profiler object.
    wait: int, default=2
        Number of wait steps before profiling warmup begins.
    warmup: int, default=2
        Number of warmup steps excluded from measured profiling window.
    active: int, default=10
        Number of active steps captured into the emitted trace.

    Returns
    -------
    Any | None
        A configured `torch.profiler.profile` object when enabled, else `None`.

    Notes
    -----
    - Traces are written to `output_dir/profiler/rank_<rank>`.
    - CUDA activity is included only when CUDA is available.
    - Keep stack/memory/shape collection disabled by default to limit overhead.
    """
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

    if not isinstance(output_dir, Path):
        raise TypeError(f'`output_dir` must be Path got {type(output_dir).__name__}')
    if not isinstance(rank, int):
        raise TypeError(f'`rank` must be Integer got {type(rank).__name__}')
    if not isinstance(wait, int) or wait < 0:
        raise ValueError(f"`wait` must be non-negative int, got {wait!r}")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"`warmup` must be non-negative int, got {warmup!r}")
    if not isinstance(active, int) or active <= 0:
        raise ValueError(f"`active` must be positive int, got {active!r}")

    if not enabled:
        return None

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    trace_dir = output_dir / "profiler" / f"rank_{rank}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    profiler = profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
    )
    return profiler


def reduce_float_dict(
    metrics: dict[str, float],
    runtime: dict[str, Any],
    *,
    reduction: str,
) -> dict[str, float]:
    """Reduce scalar metrics across ranks using sum or max."""
    if not metrics:
        return {}
    if reduction not in {"sum", "max"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    keys = list(metrics.keys())
    device = runtime.get("device", torch.device("cpu"))
    tensor_device = device if isinstance(device, torch.device) and device.type == "cuda" else torch.device("cpu")
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        dtype=torch.float64,
        device=tensor_device,
    )

    if runtime.get("is_distributed", False) and dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.SUM if reduction == "sum" else dist.ReduceOp.MAX
        dist.all_reduce(values, op=op)

    return {key: float(values[idx].item()) for idx, key in enumerate(keys)}


def aggregate_epoch_metrics(local_metrics: dict[str, float], runtime: dict[str, Any]) -> dict[str, float]:
    """Aggregate local epoch metrics into cluster-level benchmark metrics."""
    world_size = max(1, int(runtime.get("world_size", 1)))

    sum_metrics = reduce_float_dict(
        {
            "samples_seen": local_metrics["samples_seen"],
            "input_tokens": local_metrics["input_tokens"],
            "target_tokens": local_metrics["target_tokens"],
            "padded_tokens": local_metrics["padded_tokens"],
            "device_batches": local_metrics["device_batches"],
            "host_data_wait_time_sec": local_metrics["host_data_wait_time_sec"],
            "host_batch_time_sec": local_metrics["host_batch_time_sec"],
            "host_forward_backward_time_sec": local_metrics["host_forward_backward_time_sec"],
            "host_optimizer_time_sec": local_metrics["host_optimizer_time_sec"],
            "loss_sum": local_metrics["loss_sum"],
        },
        runtime,
        reduction="sum",
    )
    max_metrics = reduce_float_dict(
        {
            "epoch_wall_time_sec": local_metrics["epoch_wall_time_sec"],
            "max_memory_allocated_gb": local_metrics["max_memory_allocated_gb"],
            "max_memory_reserved_gb": local_metrics["max_memory_reserved_gb"],
            "optimizer_steps": local_metrics["optimizer_steps"],
            "global_step": local_metrics["global_step"],
            "stopped_early": local_metrics["stopped_early"],
        },
        runtime,
        reduction="max",
    )

    samples_seen = sum_metrics["samples_seen"]
    input_tokens = sum_metrics["input_tokens"]
    target_tokens = sum_metrics["target_tokens"]
    padded_tokens = sum_metrics["padded_tokens"]
    device_batches = sum_metrics["device_batches"]
    epoch_wall_time = max(1e-12, max_metrics["epoch_wall_time_sec"])
    optimizer_steps = max(0.0, max_metrics["optimizer_steps"])

    avg_loss = sum_metrics["loss_sum"] / max(1.0, float(world_size))
    samples_per_second = samples_seen / epoch_wall_time
    input_tokens_per_second = input_tokens / epoch_wall_time
    target_tokens_per_second = target_tokens / epoch_wall_time
    device_batches_per_second = device_batches / epoch_wall_time
    optimizer_steps_per_second = optimizer_steps / epoch_wall_time if optimizer_steps > 0 else 0.0

    return {
        "loss": avg_loss,
        "global_step": max_metrics["global_step"],
        "stopped_early": max_metrics["stopped_early"],
        "optimizer_steps": optimizer_steps,
        "epoch_wall_time_sec": epoch_wall_time,
        "samples_seen_global": samples_seen,
        "samples_seen_per_rank_avg": samples_seen / float(world_size),
        "device_batches_global": device_batches,
        "device_batches_per_rank_avg": device_batches / float(world_size),
        "input_tokens_global": input_tokens,
        "target_tokens_global": target_tokens,
        "padded_tokens_global": padded_tokens,
        "samples_per_second_global": samples_per_second,
        "samples_per_second_per_gpu_avg": samples_per_second / float(world_size),
        "input_tokens_per_second_global": input_tokens_per_second,
        "target_tokens_per_second_global": target_tokens_per_second,
        "optimizer_steps_per_second": optimizer_steps_per_second,
        "device_batches_per_second_global": device_batches_per_second,
        "avg_input_tokens_per_sample": input_tokens / max(1.0, samples_seen),
        "avg_target_tokens_per_sample": target_tokens / max(1.0, samples_seen),
        "avg_padded_tokens_per_sample": padded_tokens / max(1.0, samples_seen),
        "padding_ratio": 1.0 - (input_tokens / max(1.0, padded_tokens)),
        "seconds_per_optimizer_step": epoch_wall_time / max(1.0, optimizer_steps),
        "milliseconds_per_sample_global": (epoch_wall_time * 1000.0) / max(1.0, samples_seen),
        "host_data_wait_time_sec_avg": sum_metrics["host_data_wait_time_sec"] / float(world_size),
        "host_batch_time_sec_avg": sum_metrics["host_batch_time_sec"] / float(world_size),
        "host_forward_backward_time_sec_avg": sum_metrics["host_forward_backward_time_sec"] / float(world_size),
        "host_optimizer_time_sec_avg": sum_metrics["host_optimizer_time_sec"] / float(world_size),
        "host_data_wait_fraction_avg": (sum_metrics["host_data_wait_time_sec"] / float(world_size)) / epoch_wall_time,
        "host_forward_backward_fraction_avg": (sum_metrics["host_forward_backward_time_sec"] / float(world_size)) / epoch_wall_time,
        "host_optimizer_fraction_avg": (sum_metrics["host_optimizer_time_sec"] / float(world_size)) / epoch_wall_time,
        "max_memory_allocated_gb": max_metrics["max_memory_allocated_gb"],
        "max_memory_reserved_gb": max_metrics["max_memory_reserved_gb"],
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def collect_runtime_metadata(runtime: dict[str, Any]) -> dict[str, Any]:
    """Collect hardware/runtime metadata for benchmark artifacts."""
    device = runtime.get("device", torch.device("cpu"))
    metadata: dict[str, Any] = {
        "distributed": bool(runtime.get("is_distributed", False)),
        "world_size": int(runtime.get("world_size", 1)),
        "rank": int(runtime.get("rank", 0)),
        "local_rank": int(runtime.get("local_rank", 0)),
        "device_type": device.type if isinstance(device, torch.device) else str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }

    if torch.cuda.is_available() and isinstance(device, torch.device) and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": float(props.total_memory) / (1024 ** 3),
                "gpu_multi_processor_count": int(props.multi_processor_count),
            }
        )

    if dist.is_available() and dist.is_initialized():
        metadata["distributed_backend"] = str(dist.get_backend())

    return metadata


def build_train_components(config: dict, runtime: dict) -> dict[str, Any]:
    """
    Construct all objects required for training from configuration and runtime context.

    Parameters
    ----------
    config: dict
        Normalized training configuration dictionary.
    runtime: dict
        Distributed/runtime metadata from setup_distributed().

    Returns
    -------
        A dictionary containing initialized training components, typically:
        - model
        - tokenizer / processor
        - train_dataset
        - train_dataloader
        - optimizer
        - lr_scheduler
        - grad_scaler (if AMP enabled)
        - training_state metadata

    Notes
    -----
    - Keep construction deterministic and centralized.
    - Ensure dataloader/sampler choices are DDP-safe.
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
    from torch.optim import AdamW
    from transformers import get_scheduler

    from GLM.scripts.model_loader import load_model_bundle
    from GLM.data.dataset import build_dataset
    from GLM.data.sampler import LengthBucketBatchSampler
    from GLM.scripts.collator import DataCollator, render_chat_text

    train_cfg = config.get("train", config.get("training", {}))
    data_cfg = config.get("data", {})
    optim_cfg = config.get("optimizer", {})
    paths_cfg = config.get("paths", {})
    quant_cfg = config.get("quantization", {})
    model_cfg = config.get("model", {})
    precision_cfg = config.get("precision", {})
    lora_cfg = config.get("lora", {})

    train_annotation_path = data_cfg.get("train_annotation_path") or data_cfg.get("train_json") or paths_cfg.get("train_json")
    image_root = (
    data_cfg.get("train_image_root")
    or data_cfg.get("image_root")
    or data_cfg.get("train_image_folder")
    or paths_cfg.get("train_image_folder")
    )
    if not train_annotation_path:
        raise ValueError("Missing training annotation path. Expected data.train_annotation_path or paths.train_json")
    if not image_root:
        raise ValueError(
            "Missing training image root. Expected data.train_image_root, "
            "data.image_root, or paths.train_image_folder"
        )

    per_device_batch_size = int(train_cfg.get("per_device_batch_size", train_cfg.get("batch_size", 1)))
    num_workers = int(train_cfg.get("num_workers", 4))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    max_steps_cfg = train_cfg.get("max_steps")
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else None
    epochs_cfg = train_cfg.get("epochs", train_cfg.get("max_epochs", 1))
    epochs = int(epochs_cfg)
    drop_last = bool(train_cfg.get("drop_last", True))

    lr = float(optim_cfg.get("lr", train_cfg.get("lr", 2e-4)))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    rank = int(runtime["rank"])
    local_rank = int(runtime["local_rank"])
    world_size = int(runtime["world_size"])
    device = runtime["device"]
    is_distributed = bool(runtime["is_distributed"])

    # Build normalized model-loader config for current runtime.
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
        "quantization_double_quant": bool(
            config.get("quantization_double_quant", quant_cfg.get("double_quant", True))
        ),
        "training_gradient_checkpointing": bool(
            config.get(
                "training_gradient_checkpointing",
                train_cfg.get("gradient_checkpointing", train_cfg.get("training_gradient_checkpointing", True)),
            )
        ),
        "training_use_cache": bool(config.get("training_use_cache", train_cfg.get("use_cache", False))),
        "lora_enabled": bool(config.get("lora_enabled", lora_cfg.get("enabled", False))),
        "lora_r": config.get("lora_r", lora_cfg.get("r")),
        "lora_alpha": config.get("lora_alpha", lora_cfg.get("alpha")),
        "lora_dropout": float(config.get("lora_dropout", lora_cfg.get("dropout", 0.05))),
        "lora_bias": str(config.get("lora_bias", lora_cfg.get("bias", "none"))).lower(),
        "lora_task_type": str(config.get("lora_task_type", lora_cfg.get("task_type", "CAUSAL_LM"))),
        "target_modules": config.get("target_modules", lora_cfg.get("target_modules")),
    }

    if not model_loader_cfg["model_id"]:
        raise ValueError("Missing model_id in config (root.model_id or model.model_id required)")

    # For DDP, each rank should own the full model on its local device.
    if is_distributed and device.type == "cuda":
        model_loader_cfg["device_map"] = {"": local_rank}

    model_bundle = load_model_bundle(config=model_loader_cfg)
    model = model_bundle["model"]
    processor = model_bundle["processor"]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor does not expose a tokenizer. DataCollator requires processor.tokenizer.")

    train_dataset = build_dataset(
        annotation_path=str(train_annotation_path),
        image_root=str(image_root),
        split="train",
        strict=False,
    )

    lengths: list[int] = []
    for sample in train_dataset.samples:
        text = render_chat_text(
            processor,
            conversations=sample["conversations"],
            image_path=sample["image"],
            add_generation_prompt=False,
            include_assistant=True,
        )
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        lengths.append(max(1, len(token_ids)))

    train_batch_sampler = LengthBucketBatchSampler(
        lengths=lengths,
        batch_size=per_device_batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_replicas=world_size,
        rank=rank,
    )

    collator = DataCollator.from_config(
        processor=processor,
        tokenizer=tokenizer,
        config=train_cfg,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    if device.type == "cuda" and not bool(model_loader_cfg.get("quantization_enabled", False)):
        model = model.to(device)
    if is_distributed:
        ddp_device_ids = [local_rank] if device.type == "cuda" else None
        ddp_output_device = local_rank if device.type == "cuda" else None
        model = DDP(model, device_ids=ddp_device_ids, output_device=ddp_output_device)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )

    steps_per_epoch = max(1, len(train_dataloader))
    total_steps = max_steps if max_steps is not None else max(1, epochs * steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=str(train_cfg.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    precision = str(train_cfg.get("precision", "fp16")).lower()
    use_fp16 = precision == "fp16"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_fp16 and torch.cuda.is_available(),
    )

    train_state = {
        "start_epoch": 0,
        "global_step": 0,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": total_steps,
        "epochs": epochs,
        "precision": precision,
        "train_dataset_size": len(train_dataset),
        "per_device_batch_size": per_device_batch_size,
        "grad_accum_steps": int(train_cfg.get("grad_accum_steps", train_cfg.get("accumulate_grad_batches", 1))),
        "world_size": world_size,
        "effective_local_batch_size": per_device_batch_size * int(
            train_cfg.get("grad_accum_steps", train_cfg.get("accumulate_grad_batches", 1))
        ),
        "effective_global_batch_size": per_device_batch_size
        * int(train_cfg.get("grad_accum_steps", train_cfg.get("accumulate_grad_batches", 1)))
        * world_size,
    }

    return {
        "model_bundle": model_bundle,
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "train_dataset": train_dataset,
        "train_batch_sampler": train_batch_sampler,
        "collator": collator,
        "train_dataloader": train_dataloader,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "scaler": scaler,
        "train_state": train_state,
    }


def maybe_load_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: torch.amp.GradScaler | None,
    resume_path: str | None,
    device: torch.device,
) -> dict[str, int]:
    """
    Optionally restore model and training state from a checkpoint for resumed training.

    Parameters
    ----------
    model: torch.nn.Module
        Model to restore weights into.
    optimizer: torch.optim.Optimizer | None
        Optimizer to restore state into, if available.
    scheduler: Any | None
        LR scheduler to restore state into, if available.
    scaler: torch.amp.GradScaler | None
        AMP GradScaler to restore state into, if available.
    resume_path: str | None
        Checkpoint path; if None, resume is skipped.
    device: torch.device
        Device used for map_location while loading checkpoint tensors.

    Returns:
        A state dictionary with resume metadata, typically including:
        - start_epoch: int
        - global_step: int

    Notes:
    - Handle missing optional states gracefully with clear warnings.
    - Validate checkpoint schema/version to avoid silent partial restores.
    """
    if resume_path is None:
        return {"start_epoch": 0, "global_step": 0}

    ckpt_path = Path(resume_path).expanduser().resolve()
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

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    stored_epoch = int(checkpoint.get("epoch", -1))
    start_epoch = max(0, stored_epoch + 1)
    global_step = int(checkpoint.get("global_step", 0))
    return {"start_epoch": start_epoch, "global_step": global_step}


def train_one_epoch(
    *,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
    global_step: int,
    config: dict,
    runtime: dict,
    profiler: Any | None = None,
) -> dict[str, float]:
    """
    Execute one full training epoch over the training dataloader.

    Parameters
    ----------
    model: torch.nn.Module
        Training model (DDP-wrapped or plain module).
    train_loader: torch.utils.data.DataLoader
        Training dataloader.
    optimizer: torch.optim.Optimizer
        Optimizer instance.
    scheduler: Any | None
        Optional scheduler stepped per step or per epoch.
    scaler: torch.cuda.amp.GradScaler | None
        Optional AMP GradScaler for mixed-precision updates.
    device: torch.device
        Device to place batch tensors on.
    epoch: int
        Current epoch index.
    global_step: int
        Global optimizer step count at the start of the epoch.
    config: dict
        Training configuration dictionary.
    runtime: dict
        Distributed/runtime metadata.

    Returns
    -------
        Dictionary of aggregated epoch metrics (for example loss, lr, throughput).

    Notes
    -----
    - Support gradient accumulation, AMP autocast, and gradient clipping.
    - Keep logging/checkpoint side effects rank-aware (rank 0 only).
    """
    if not isinstance(config, dict):
        raise TypeError(f"`config` must be dict, got {type(config).__name__}")
    if not isinstance(runtime, dict):
        raise TypeError(f"`runtime` must be dict, got {type(runtime).__name__}")

    train_cfg = config.get("train", config.get("training", {}))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", train_cfg.get("accumulate_grad_batches", 1)))
    grad_accum_steps = max(1, grad_accum_steps)
    max_steps_cfg = train_cfg.get("max_steps")
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else None
    max_grad_norm = float(train_cfg.get("max_grad_norm", train_cfg.get("gradient_clip_val", 1.0)))
    log_every = int(train_cfg.get("log_every", 10))
    precision = str(train_cfg.get("precision", "fp16")).lower()

    autocast_enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epoch_start_time = time.perf_counter()
    last_batch_end_time = epoch_start_time
    running_loss = 0.0
    optimizer_steps = 0
    current_global_step = int(global_step)
    stop_training = False
    processed_batches = 0
    samples_seen = 0
    input_tokens = 0
    target_tokens = 0
    padded_tokens = 0
    host_data_wait_time = 0.0
    host_batch_time = 0.0
    host_forward_backward_time = 0.0
    host_optimizer_time = 0.0

    for step_idx, batch in enumerate(train_loader, start=1):
        batch_ready_time = time.perf_counter()
        host_data_wait_time += batch_ready_time - last_batch_end_time
        processed_batches = step_idx
        tensor_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
        batch_size = int(tensor_batch["input_ids"].shape[0]) if "input_ids" in tensor_batch else 0
        samples_seen += batch_size
        if "attention_mask" in tensor_batch:
            input_tokens += int(tensor_batch["attention_mask"].sum().item())
        elif "input_ids" in tensor_batch:
            input_tokens += int(tensor_batch["input_ids"].numel())
        if "labels" in tensor_batch:
            target_tokens += int((tensor_batch["labels"] != -100).sum().item())
        if "input_ids" in tensor_batch:
            padded_tokens += int(tensor_batch["input_ids"].numel())

        forward_backward_start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            outputs = model(**tensor_batch)
            loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"Encountered non-finite loss at step {step_idx}: {float(loss.detach().cpu())}")

        running_loss += float(loss.detach().cpu())
        loss_to_backprop = loss / grad_accum_steps

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()
        host_forward_backward_time += time.perf_counter() - forward_backward_start

        should_step = (step_idx % grad_accum_steps == 0) or (step_idx == len(train_loader))
        if should_step:
            optimizer_start = time.perf_counter()
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            host_optimizer_time += time.perf_counter() - optimizer_start
            optimizer_steps += 1
            current_global_step += 1

            if max_steps is not None and current_global_step >= max_steps:
                stop_training = True

            if runtime.get("is_main_process", True) and (optimizer_steps % max(1, log_every) == 0):
                current_lr = float(optimizer.param_groups[0]["lr"])
                avg_so_far = running_loss / step_idx
                print(
                    f"[train] epoch={epoch} step={optimizer_steps} "
                    f"loss={avg_so_far:.6f} lr={current_lr:.6e}"
                )

        if profiler is not None:
            profiler.step()

        last_batch_end_time = time.perf_counter()
        host_batch_time += last_batch_end_time - batch_ready_time
        if stop_training:
            break

    avg_loss = running_loss / max(1, processed_batches)
    current_lr = float(optimizer.param_groups[0]["lr"])
    epoch_wall_time = time.perf_counter() - epoch_start_time
    max_memory_allocated_gb = 0.0
    max_memory_reserved_gb = 0.0
    if device.type == "cuda":
        max_memory_allocated_gb = float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3)
        max_memory_reserved_gb = float(torch.cuda.max_memory_reserved(device)) / (1024 ** 3)

    return aggregate_epoch_metrics(
        {
            "loss_sum": avg_loss,
            "lr": current_lr,
            "optimizer_steps": float(optimizer_steps),
            "global_step": float(current_global_step),
            "stopped_early": float(stop_training),
            "epoch_wall_time_sec": epoch_wall_time,
            "samples_seen": float(samples_seen),
            "input_tokens": float(input_tokens),
            "target_tokens": float(target_tokens),
            "padded_tokens": float(padded_tokens),
            "device_batches": float(processed_batches),
            "host_data_wait_time_sec": host_data_wait_time,
            "host_batch_time_sec": host_batch_time,
            "host_forward_backward_time_sec": host_forward_backward_time,
            "host_optimizer_time_sec": host_optimizer_time,
            "max_memory_allocated_gb": max_memory_allocated_gb,
            "max_memory_reserved_gb": max_memory_reserved_gb,
        },
        runtime,
    ) | {"lr": current_lr}


def save_checkpoint(
    *,
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    config: dict,
    runtime: dict,
) -> None:
    """
    Save model and optimizer training state to disk for recovery and reproducibility.

    Parameters
    ----------
    checkpoint_path: str
        Destination checkpoint file path.
    model: torch.nn.Module
        Model whose parameters are saved (unwrap DDP module when needed).
    optimizer: torch.optim.Optimizer | None
        Optimizer state to save, if available.
    scheduler: Any | None
        Scheduler state to save, if available.
    scaler: torch.amp.GradScaler | None
        AMP GradScaler state to save, if available.
    epoch: int
        Current epoch index.
    global_step: int
        Current global optimization step.
    config: dict
        Training configuration snapshot to embed in checkpoint.
    runtime: dict
        Distributed/runtime metadata for rank-aware save behavior.

    Returns
    -------
    None.

    Notes:
    - Save only on rank 0 in distributed runs.
    - Write atomically where possible to avoid partial/corrupt checkpoint files.
    """
    if not runtime.get("is_main_process", True):
        return

    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config,
    }

    tmp_path = ckpt_path.with_suffix(f"{ckpt_path.suffix}.tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(ckpt_path)


def cleanup_distributed(runtime: dict) -> None:
    """
    Tear down distributed process group resources when training finishes or fails.

    Parameters
    ----------
    runtime: dict
        Distributed/runtime metadata returned by setup_distributed().

    Returns
    -------
    None.

    Notes:
    - Safe to call in single-process mode (no-op).
    - Should be executed in finally blocks to prevent leaked process groups.
    """
    if not isinstance(runtime, dict):
        return

    if runtime.get("is_distributed", False) and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    """
    Training entrypoint that orchestrates config loading, setup, training loop, and teardown.

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    Notes:
    - Owns top-level control flow and error handling.
    - Ensures deterministic setup and consistent cleanup in all exit paths.
    """
    from GLM.scripts.utils.load_config import load_yaml_config
    from GLM.scripts.utils.wandb import (
        finish_wandb_run,
        init_wandb_run,
        log_wandb_artifact,
        log_wandb_metrics,
        update_wandb_summary,
    )

    args = parse_args()
    runtime = setup_distributed()
    wandb_run = None
    benchmark_history: list[dict[str, Any]] = []

    try:
        config = load_yaml_config(args.config)
        if not isinstance(config, dict):
            raise TypeError("Config loader must return a dictionary")

        config.setdefault("train", config.get("training", {}))
        config.setdefault("optimizer", {})
        config.setdefault("data", {})
        config.setdefault("paths", {})
        config.setdefault("wandb", {})
        config.setdefault("profiling", {})
        config.setdefault("benchmark", {})

        train_cfg = config["train"]
        optim_cfg = config["optimizer"]
        profile_cfg = config["profiling"]
        benchmark_cfg = config["benchmark"]

        if args.seed is not None:
            train_cfg["seed"] = args.seed
        if args.epochs is not None:
            train_cfg["epochs"] = args.epochs
        if args.max_steps is not None:
            train_cfg["max_steps"] = args.max_steps
        if args.per_device_batch_size is not None:
            train_cfg["per_device_batch_size"] = args.per_device_batch_size
        if args.grad_accum_steps is not None:
            train_cfg["grad_accum_steps"] = args.grad_accum_steps
        if args.num_workers is not None:
            train_cfg["num_workers"] = args.num_workers
        if args.precision is not None:
            train_cfg["precision"] = args.precision
        if args.log_every is not None:
            train_cfg["log_every"] = args.log_every
        if args.save_every is not None:
            train_cfg["save_every"] = args.save_every
        if args.lr is not None:
            optim_cfg["lr"] = args.lr

        resolved_epochs = train_cfg.get("epochs", train_cfg.get("max_epochs"))
        resolved_max_steps = train_cfg.get("max_steps")
        if resolved_epochs is None and resolved_max_steps is None:
            raise ValueError(
                "Specify at least one training limit in config or CLI: train.epochs or train.max_steps."
            )

        config["output_dir"] = str(args.output_dir)
        config["profile"] = bool(config.get("profile", profile_cfg.get("enabled", False)) or args.profile)
        config["profile_steps"] = int(
            args.profile_steps
            if args.profile_steps is not None
            else config.get("profile_steps", profile_cfg.get("steps", 50))
        )
        config["profile_wait"] = int(
            args.profile_wait
            if args.profile_wait is not None
            else config.get("profile_wait", profile_cfg.get("wait", 2))
        )
        config["profile_warmup"] = int(
            args.profile_warmup
            if args.profile_warmup is not None
            else config.get("profile_warmup", profile_cfg.get("warmup", 2))
        )
        config["profile_active"] = int(
            args.profile_active
            if args.profile_active is not None
            else config.get("profile_active", profile_cfg.get("active", config["profile_steps"]))
        )
        config["profile_epoch"] = int(
            args.profile_epoch
            if args.profile_epoch is not None
            else config.get("profile_epoch", profile_cfg.get("epoch", 0))
        )

        wandb_run = init_wandb_run(config=config, runtime=runtime, job_type="train")

        base_seed = int(train_cfg.get("seed", 42))
        seed_everything(seed=base_seed, rank=int(runtime["rank"]))

        components = build_train_components(config=config, runtime=runtime)
        model = components["model"]
        train_loader = components["train_dataloader"]
        train_batch_sampler = components["train_batch_sampler"]
        optimizer = components["optimizer"]
        scheduler = components["lr_scheduler"]
        scaler = components["scaler"]
        train_state = components["train_state"]

        resume_state = maybe_load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resume_path=str(args.resume) if args.resume is not None else None,
            device=runtime["device"],
        )
        train_state["start_epoch"] = int(resume_state["start_epoch"])
        train_state["global_step"] = int(resume_state["global_step"])

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        total_epochs = int(train_state["epochs"])
        max_steps = int(train_state["max_steps"])
        benchmark_dir = output_dir / "benchmark"
        benchmark_epoch_path = benchmark_dir / "epoch_metrics.jsonl"
        benchmark_summary_path = benchmark_dir / "summary.json"
        runtime_metadata = collect_runtime_metadata(runtime)

        for epoch in range(int(train_state["start_epoch"]), total_epochs):
            if train_state["global_step"] >= max_steps:
                break
            if hasattr(train_batch_sampler, "set_epoch"):
                train_batch_sampler.set_epoch(epoch)
            profile_enabled = bool(config.get("profile", False))
            profile_epoch = int(config.get("profile_epoch", 0))
            profiler = build_profiler(
                output_dir=output_dir,
                rank=int(runtime["rank"]),
                enabled=(profile_enabled and epoch == profile_epoch),
                wait=int(config.get("profile_wait", 2)),
                warmup=int(config.get("profile_warmup", 2)),
                active=int(config.get("profile_active", max(1, int(config.get("profile_steps", 50))))),
            )

            if profiler is not None:
                with profiler:
                    metrics = train_one_epoch(
                        model=model,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        device=runtime["device"],
                        epoch=epoch,
                        global_step=int(train_state["global_step"]),
                        config=config,
                        runtime=runtime,
                        profiler=profiler,
                    )
            else:
                metrics = train_one_epoch(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    device=runtime["device"],
                    epoch=epoch,
                    global_step=int(train_state["global_step"]),
                    config=config,
                    runtime=runtime,
                    profiler=None,
                )

            train_state["global_step"] = int(metrics.get("global_step", train_state["global_step"]))
            epoch_record = {
                "epoch": int(epoch),
                "lr": float(metrics.get("lr", 0.0)),
                **metrics,
            }
            benchmark_history.append(epoch_record)

            if runtime.get("is_main_process", True) and bool(benchmark_cfg.get("save_json", True)):
                append_jsonl(benchmark_epoch_path, epoch_record)

            if runtime.get("is_main_process", True):
                print(
                    f"[epoch_end] epoch={epoch} loss={metrics.get('loss', 0.0):.6f} "
                    f"lr={metrics.get('lr', 0.0):.6e} global_step={train_state['global_step']} "
                    f"samples_per_sec={metrics.get('samples_per_second_global', 0.0):.2f} "
                    f"target_tokens_per_sec={metrics.get('target_tokens_per_second_global', 0.0):.2f} "
                    f"epoch_time_s={metrics.get('epoch_wall_time_sec', 0.0):.2f}"
                )

            log_wandb_metrics(
                wandb_run,
                {
                    "epoch": float(epoch),
                    "global_step": float(train_state["global_step"]),
                    "loss": float(metrics.get("loss", 0.0)),
                    "lr": float(metrics.get("lr", 0.0)),
                    "optimizer_steps": float(metrics.get("optimizer_steps", 0.0)),
                    "epoch_wall_time_sec": float(metrics.get("epoch_wall_time_sec", 0.0)),
                    "samples_seen_global": float(metrics.get("samples_seen_global", 0.0)),
                    "samples_per_second_global": float(metrics.get("samples_per_second_global", 0.0)),
                    "samples_per_second_per_gpu_avg": float(metrics.get("samples_per_second_per_gpu_avg", 0.0)),
                    "input_tokens_per_second_global": float(metrics.get("input_tokens_per_second_global", 0.0)),
                    "target_tokens_per_second_global": float(metrics.get("target_tokens_per_second_global", 0.0)),
                    "optimizer_steps_per_second": float(metrics.get("optimizer_steps_per_second", 0.0)),
                    "seconds_per_optimizer_step": float(metrics.get("seconds_per_optimizer_step", 0.0)),
                    "avg_input_tokens_per_sample": float(metrics.get("avg_input_tokens_per_sample", 0.0)),
                    "avg_target_tokens_per_sample": float(metrics.get("avg_target_tokens_per_sample", 0.0)),
                    "avg_padded_tokens_per_sample": float(metrics.get("avg_padded_tokens_per_sample", 0.0)),
                    "padding_ratio": float(metrics.get("padding_ratio", 0.0)),
                    "host_data_wait_time_sec_avg": float(metrics.get("host_data_wait_time_sec_avg", 0.0)),
                    "host_batch_time_sec_avg": float(metrics.get("host_batch_time_sec_avg", 0.0)),
                    "host_forward_backward_time_sec_avg": float(metrics.get("host_forward_backward_time_sec_avg", 0.0)),
                    "host_optimizer_time_sec_avg": float(metrics.get("host_optimizer_time_sec_avg", 0.0)),
                    "host_data_wait_fraction_avg": float(metrics.get("host_data_wait_fraction_avg", 0.0)),
                    "host_forward_backward_fraction_avg": float(metrics.get("host_forward_backward_fraction_avg", 0.0)),
                    "host_optimizer_fraction_avg": float(metrics.get("host_optimizer_fraction_avg", 0.0)),
                    "max_memory_allocated_gb": float(metrics.get("max_memory_allocated_gb", 0.0)),
                    "max_memory_reserved_gb": float(metrics.get("max_memory_reserved_gb", 0.0)),
                },
                step=int(train_state["global_step"]),
                prefix="train",
            )

            last_checkpoint_path = output_dir / "last.pt"
            save_checkpoint(
                checkpoint_path=str(last_checkpoint_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=int(train_state["global_step"]),
                config=config,
                runtime=runtime,
            )
            if config["wandb"].get("log_checkpoints", False):
                log_wandb_artifact(
                    wandb_run,
                    last_checkpoint_path,
                    artifact_type="checkpoint",
                    name=f"last-epoch-{epoch}",
                )

            save_every = int(train_cfg.get("save_every", 0))
            if save_every > 0 and (train_state["global_step"] % save_every == 0):
                save_checkpoint(
                    checkpoint_path=str(output_dir / f"step_{train_state['global_step']}.pt"),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=int(train_state["global_step"]),
                    config=config,
                    runtime=runtime,
                )

            if train_state["global_step"] >= max_steps:
                break
    finally:
        benchmark_dir = (output_dir / "benchmark") if "output_dir" in locals() else None
        benchmark_summary_path = (benchmark_dir / "summary.json") if benchmark_dir is not None else None
        if (
            runtime.get("is_main_process", True)
            and "output_dir" in locals()
            and "train_state" in locals()
            and "config" in locals()
            and bool(config.get("benchmark", {}).get("save_json", True))
            and benchmark_summary_path is not None
        ):
            benchmark_summary = {
                "runtime": runtime_metadata if "runtime_metadata" in locals() else collect_runtime_metadata(runtime),
                "model_id": config.get("model", {}).get("model_id"),
                "output_dir": str(output_dir),
                "train_dataset_size": int(train_state.get("train_dataset_size", 0)),
                "steps_per_epoch": int(train_state.get("steps_per_epoch", 0)),
                "epochs_configured": int(train_state.get("epochs", 0)),
                "effective_local_batch_size": int(train_state.get("effective_local_batch_size", 0)),
                "effective_global_batch_size": int(train_state.get("effective_global_batch_size", 0)),
                "per_device_batch_size": int(train_state.get("per_device_batch_size", 0)),
                "grad_accum_steps": int(train_state.get("grad_accum_steps", 0)),
                "profiling_enabled": bool(config.get("profile", False)),
                "profiling": {
                    "steps": int(config.get("profile_steps", 0)),
                    "wait": int(config.get("profile_wait", 0)),
                    "warmup": int(config.get("profile_warmup", 0)),
                    "active": int(config.get("profile_active", 0)),
                    "epoch": int(config.get("profile_epoch", 0)),
                },
                "epochs": benchmark_history,
            }
            write_json(benchmark_summary_path, benchmark_summary)

            if wandb_run is not None and bool(config.get("wandb", {}).get("log_metrics_artifact", True)):
                log_wandb_artifact(
                    wandb_run,
                    benchmark_dir,
                    artifact_type="benchmark",
                    name=f"benchmark-{output_dir.name}",
                )

        profiler_dir = output_dir / "profiler" if "output_dir" in locals() else None
        if (
            wandb_run is not None
            and "config" in locals()
            and bool(config.get("profile", False))
            and bool(config.get("wandb", {}).get("log_profiler_artifact", True))
            and profiler_dir is not None
            and profiler_dir.exists()
        ):
            log_wandb_artifact(
                wandb_run,
                profiler_dir,
                artifact_type="profiler",
                name=f"profiler-{profiler_dir.parent.name}",
            )
        update_wandb_summary(
            wandb_run,
            {
                "train/global_step": float(train_state["global_step"]) if "train_state" in locals() else 0.0,
            },
        )
        finish_wandb_run(wandb_run)
        cleanup_distributed(runtime)


if __name__ == "__main__":
    main()
