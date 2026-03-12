from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from GLM.scripts.model_loader import (
    attach_lora,
    build_lora_config,
    load_model,
    load_processor,
)
from GLM.scripts.utils.hf_utils import upload_model_folder
from GLM.scripts.utils.load_config import load_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for exporting and uploading a merged model."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into the base GLM model and upload the merged export to Hugging Face Hub."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to training config YAML.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the training checkpoint (.pt).")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face model repo id (defaults to HF_MODEL_REPO_ID env var).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Local directory for the merged Hugging Face export.",
    )
    parser.add_argument("--revision", type=str, default="main", help="Target Hub revision/branch.")
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload merged GLM LoRA model",
        help="Commit message used for the Hub upload.",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to create the Hub repo as private if it does not already exist.",
    )
    parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face token override.")
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Optional transformers device_map override used while rebuilding the model.",
    )
    parser.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the merged model with safetensors when supported.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Export the merged model locally without uploading to Hugging Face Hub.",
    )
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow export to continue even if checkpoint/model keys do not match exactly.",
    )

    args = parser.parse_args()
    if not args.commit_message.strip():
        parser.error("--commit-message must be non-empty")
    return args


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype_map: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    normalized = str(dtype_name).lower()
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype_name!r}")
    return dtype_map[normalized]


def build_export_loader_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Build a model-loader config suitable for merge/export, not low-bit training."""
    model_cfg = config.get("model", {})
    precision_cfg = config.get("precision", {})
    lora_cfg = config.get("lora", {})

    model_id = model_cfg.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Missing required config field: model.model_id")

    target_modules = lora_cfg.get("target_modules")
    if isinstance(target_modules, list):
        target_modules = [str(item).strip() for item in target_modules]
    elif isinstance(target_modules, str):
        target_modules = target_modules.strip()

    device_map: str | None
    if args.device_map is not None:
        device_map = args.device_map
    elif torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = None

    return {
        "model_id": model_id.strip(),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "attn_implementation": model_cfg.get("attn_implementation"),
        "device_map": device_map,
        "torch_dtype": str(precision_cfg.get("torch_dtype", "bfloat16")).lower(),
        "lora_enabled": bool(lora_cfg.get("enabled", False)),
        "lora_r": lora_cfg.get("r"),
        "lora_alpha": lora_cfg.get("alpha"),
        "lora_dropout": float(lora_cfg.get("dropout", 0.05)),
        "lora_bias": str(lora_cfg.get("bias", "none")).lower(),
        "lora_task_type": str(lora_cfg.get("task_type", "CAUSAL_LM")),
        "target_modules": target_modules,
    }


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, Any]:
    """Load a training checkpoint and return its state payload."""
    resolved_path = checkpoint_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")

    checkpoint = torch.load(resolved_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format in {resolved_path}: expected dict")

    model_state = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if model_state is None:
        raise ValueError(f"Checkpoint {resolved_path} is missing model weights")

    return checkpoint


def validate_checkpoint_load(
    incompatible_keys: Any,
    *,
    allow_partial_load: bool,
) -> dict[str, Any]:
    """Validate checkpoint loading results before exporting a merged model."""
    missing_keys = list(getattr(incompatible_keys, "missing_keys", []) or [])
    unexpected_keys = list(getattr(incompatible_keys, "unexpected_keys", []) or [])

    report = {
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "missing_key_preview": missing_keys[:20],
        "unexpected_key_preview": unexpected_keys[:20],
    }

    print(
        "[push_to_hub] checkpoint_load "
        f"missing={report['missing_key_count']} unexpected={report['unexpected_key_count']}"
    )
    if report["missing_key_preview"]:
        print(f"[push_to_hub] missing_key_preview={report['missing_key_preview']}")
    if report["unexpected_key_preview"]:
        print(f"[push_to_hub] unexpected_key_preview={report['unexpected_key_preview']}")

    if (missing_keys or unexpected_keys) and not allow_partial_load:
        raise RuntimeError(
            "Checkpoint load reported missing or unexpected keys. "
            "Refusing to export a merged model. Re-run with --allow-partial-load "
            "only if you have manually inspected the mismatch."
        )

    return report


def save_export_metadata(
    export_dir: Path,
    *,
    config_path: Path,
    checkpoint_path: Path,
    repo_id: str | None,
    revision: str,
    upload_reference: str | None,
    checkpoint_load_report: dict[str, Any] | None = None,
) -> None:
    """Persist export metadata next to the merged model."""
    payload = {
        "source_config": str(config_path.expanduser().resolve()),
        "source_checkpoint": str(checkpoint_path.expanduser().resolve()),
        "repo_id": repo_id,
        "revision": revision,
        "upload_reference": upload_reference,
        "checkpoint_load_report": checkpoint_load_report,
    }
    metadata_path = export_dir / "export_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    """Export a merged model locally and optionally upload it to Hugging Face Hub."""
    args = parse_args()

    repo_id = args.repo_id or os.getenv("HF_MODEL_REPO_ID")
    if not args.skip_upload and (repo_id is None or not repo_id.strip()):
        raise ValueError("Missing repo id. Provide --repo-id or set HF_MODEL_REPO_ID.")
    if repo_id is not None:
        repo_id = repo_id.strip()

    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config = load_yaml_config(config_path)

    loader_cfg = build_export_loader_config(config, args)
    if not bool(loader_cfg.get("lora_enabled", False)):
        raise ValueError("LoRA must be enabled in the config to merge adapters into the base model.")

    processor = load_processor(
        model_id=loader_cfg["model_id"],
        trust_remote_code=bool(loader_cfg.get("trust_remote_code", True)),
    )
    model_kwargs: dict[str, Any] = {}
    if loader_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = loader_cfg["attn_implementation"]
    model = load_model(
        model_id=loader_cfg["model_id"],
        quant_config=None,
        torch_dtype=_resolve_dtype(loader_cfg["torch_dtype"]),
        device_map=loader_cfg.get("device_map"),
        trust_remote_code=bool(loader_cfg.get("trust_remote_code", True)),
        **model_kwargs,
    )
    model = attach_lora(model, build_lora_config(loader_cfg))

    checkpoint = load_checkpoint_state(checkpoint_path)
    model_state = checkpoint.get("model", checkpoint.get("model_state_dict"))
    incompatible_keys = model.load_state_dict(model_state, strict=False)
    checkpoint_load_report = validate_checkpoint_load(
        incompatible_keys,
        allow_partial_load=bool(args.allow_partial_load),
    )

    if not hasattr(model, "merge_and_unload"):
        raise RuntimeError("Loaded model does not expose `merge_and_unload()`. Cannot create a merged export.")

    merged_model = model.merge_and_unload()
    if hasattr(merged_model, "eval"):
        merged_model.eval()
    if hasattr(merged_model, "config"):
        merged_model.config.use_cache = True

    export_dir = (
        args.export_dir.expanduser().resolve()
        if args.export_dir is not None
        else checkpoint_path.parent / "hf_merged_export"
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        str(export_dir),
        safe_serialization=bool(args.safe_serialization),
    )
    processor.save_pretrained(str(export_dir))
    save_export_metadata(
        export_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        repo_id=repo_id,
        revision=args.revision,
        upload_reference=None,
        checkpoint_load_report=checkpoint_load_report,
    )

    upload_reference = None
    if not args.skip_upload:
        upload_reference = upload_model_folder(
            repo_id=repo_id,
            model_dir=export_dir,
            revision=args.revision,
            private=args.private,
            commit_message=args.commit_message,
            token=args.token,
        )

    save_export_metadata(
        export_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        repo_id=repo_id,
        revision=args.revision,
        upload_reference=upload_reference,
        checkpoint_load_report=checkpoint_load_report,
    )

    print(f"[push_to_hub] export_dir={export_dir}")
    if upload_reference is not None:
        print(f"[push_to_hub] upload_reference={upload_reference}")
    else:
        print("[push_to_hub] upload skipped")


if __name__ == "__main__":
    main()
