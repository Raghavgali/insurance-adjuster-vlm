from __future__ import annotations

from pathlib import Path 
from typing import Any

import torch
from transformers import BitsAndBytesConfig

from peft import LoraConfig, get_peft_model

from GLM.scripts.utils.load_config import load_yaml_config


def load_model_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load and validate model loader configuration.

    Expected config fields include model identifier, precision/quantization flag, 
    and optional LoRA settings used during fine-tuning.

    Parameters
    ----------
    config_path: str | Path
        Path to YAML config file.

    Returns 
    -------
    dict[str, Any]
        Parsed config dictionary for model loading and preparation.
    """
    config_path = Path(config_path).expanduser().resolve()
    config = load_yaml_config(config_path=config_path)

    model_section = config.get("model", {})
    precision_section = config.get("precision", {})
    quant_section = config.get("quantization", {})
    training_section = config.get("train", config.get("training", {}))
    lora_section = config.get("lora", {})

    model_id = model_section.get("model_id")
    if not model_id or not isinstance(model_id, str):
        raise ValueError("Missing required config field: model.model_id")

    normalized: dict[str, Any] = {
        "model_id": model_id.strip(),
        "trust_remote_code": bool(model_section.get("trust_remote_code", True)),
        "attn_implementation": model_section.get("attn_implementation"),
        "device_map": model_section.get("device_map", "auto"),
        "torch_dtype": str(precision_section.get("torch_dtype", "bfloat16")).lower(),
        "use_tf32": bool(precision_section.get("use_tf32", True)),
        "quantization_enabled": bool(quant_section.get("enabled", False)),
        "quantization_mode": str(quant_section.get("mode", "4bit")).lower(),
        "quantization_compute_dtype": str(quant_section.get("compute_dtype", "bfloat16")).lower(),
        "quantization_quant_type": str(quant_section.get("quant_type", "nf4")).lower(),
        "quantization_double_quant": bool(quant_section.get("double_quant", True)),
        "training_gradient_checkpointing": bool(training_section.get("gradient_checkpointing", True)),
        "training_use_cache": bool(training_section.get("use_cache", False)),
        "lora_enabled": bool(lora_section.get("enabled", False)),
        "lora_r": lora_section.get("r"),
        "lora_alpha": lora_section.get("alpha"),
        "lora_dropout": float(lora_section.get("dropout", 0.05)),
        "lora_bias": str(lora_section.get("bias", "none")).lower(),
        "lora_task_type": str(lora_section.get("task_type", "CAUSAL_LM")),
        "target_modules": lora_section.get("target_modules"),
    }

    if normalized["quantization_mode"] not in {"4bit", "8bit", "16bit"}:
        raise ValueError("quantization.mode must be one of: '4bit', '8bit', '16bit'")
    
    if normalized["torch_dtype"] not in {"float16", "fp16", "bfloat16", "bf16", "float32", "fp32"}:
        raise ValueError("precision.torch_dtype must be one of: float16/fp16, bfloat16/bf16, float32/fp32")
    
    if normalized["quantization_compute_dtype"] not in {"float16", "fp16", "bfloat16", "bf16", "float32", "fp32"}:
        raise ValueError("quantization.compute_dtype must be one of: float16/fp16, bfloat16/bf16, float32/fp32")
    
    if not (0.0 <= normalized["lora_dropout"] < 1.0):
        raise ValueError("lora.dropout must be in [0.0, 1.0]")
    
    if normalized["lora_bias"] not in {"none", "all", "lora_only"}:
        raise ValueError("lora.bias must be one of: none, all, lora_only")
    
    if normalized["lora_enabled"]:
        if not isinstance(normalized["lora_r"], int) or normalized["lora_r"] <= 0:
            raise ValueError("lora.r must be a positive integer when LoRA is enabled")
        if not isinstance(normalized["lora_alpha"], (int, float)) or normalized["lora_alpha"] <= 0:
            raise ValueError("lora.alpha must be a positive number when LoRA is enabled")
        target_modules = normalized["target_modules"]
        if isinstance(target_modules, str):
            normalized["target_modules"] = target_modules.strip()
            if not normalized["target_modules"]:
                raise ValueError("lora.target_modules must be a non-empty string when LoRA is enabled")
        else:
            if not isinstance(target_modules, list) or not target_modules:
                raise ValueError("lora.target_modules must be a non-empty list or string when LoRA is enabled")
            cleaned = []
            for m in target_modules:
                if not isinstance(m, str) or not m.strip():
                    raise ValueError(f"Invalid target module entry: {m!r}")
                cleaned.append(m.strip())
            normalized["target_modules"] = list(dict.fromkeys(cleaned))

    return normalized


def build_quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig | None:
    """
    Build BitsAndBytes quantization config from runtime settings.

    Parameters
    ----------
    config: dict[str, Any]
        Model/runtime section containing quantization options.

    Returns
    -------
    BitsAndBytesConfig | None
        Quantization config when enabled, otherwise `None`.
    """
    if not config.get("quantization_enabled", False):
        return None

    mode = str(config.get("quantization_mode", "8bit")).lower()
    if mode == "16bit":
        # Full precision / mixed precision training path (no bitsandbytes quantization).
        return None

    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    if mode == "4bit":
        dtype_name = str(config.get("quantization_compute_dtype", "bfloat16")).lower()
        dtype_map: dict[str, torch.dtype] = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        compute_dtype = dtype_map.get(dtype_name)
        if compute_dtype is None:
            raise ValueError(
                f"Unsupported quantization_compute_dtype: {dtype_name}. "
                f"Expected one of: {sorted(dtype_map.keys())}"
            )

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.get("quantization_quant_type", "nf4")).lower(),
            bnb_4bit_use_double_quant=bool(config.get("quantization_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    raise ValueError("quantization_mode must be one of: '4bit', '8bit', '16bit'")


def load_processor(model_id: str,
                   *,
                   trust_remote_code: bool = True,
                   padding_side: str = "right",
                   **kwargs: Any): 
    """
    Load processor/tokenization components for the target vision-language model.

    Parameters
    ----------
    model_id: str
        Hugging Face model identifier.
    **kwargs: Any
        Extra kwargs passed to processor loader.

    Returns 
    -------
    Any 
        Loaded processor object used by dataset/collator.
    """
    from transformers import AutoProcessor 

    try: 
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load processor for model '{model_id}'.") from exc 
    
    # Keep batching behavior explicit and stable.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = padding_side
        if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError(
            f"Processor for '{model_id}' does not expose `apply_chat_template`, "
            "which is required for GLM-4.6V-Flash multimodal prompting."
        )

    return processor


def load_model(model_id: str, 
               *,
               quant_config: BitsAndBytesConfig | None = None, 
               torch_dtype: torch.dtype | None = None, 
               device_map: str | dict[str, Any] | None = "auto",
               trust_remote_code: bool = True,
               **kwargs: Any):
    """
    Load base model with consistent inference/training runtime options.

    Parameters 
    ----------
    model_id: str
        Hugging Face model identifier.
    quant_config: BitsAndBytesConfig | None, optional
        Quantization config if low-bit loading is enabled.
    torch_dtype: torch.dtype | None, optional
        Target dtype for model weights.
    device_map: str | dict[str, Any] | None, optional
        Device placement strategy.
    **kwargs: Any
        Additional kwargs forwarded to model loader.
    
    Returns
    -------
    Any
        Loaded model instance ready for optional training preparation.
    """
    model_id_lower = model_id.strip().lower()
    is_glm4v_flash = "glm-4.6v-flash" in model_id_lower or "glm4v" in model_id_lower

    if is_glm4v_flash:
        try:
            from transformers import Glm4vForConditionalGeneration as ModelClass
        except ImportError:
            try:
                from transformers import Glm4vMoeForConditionalGeneration as ModelClass
            except ImportError as exc:
                raise RuntimeError(
                    "GLM-4.6V-Flash requires a transformers build that exposes "
                    "`Glm4vForConditionalGeneration` (or the older "
                    "`Glm4vMoeForConditionalGeneration`). Install the version "
                    "recommended by the model card."
                ) from exc
    else:
        from transformers import AutoModelForCausalLM as ModelClass

    try:
        model = ModelClass.from_pretrained(
            model_id,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate model '{model_id}'.") from exc 
    
    return model

def prepare_model_for_training(model, 
                               use_gradient_checkpointing: bool = True,
                               use_cache: bool = False,) -> Any:
    """
    Apply training-safe model settings before optimizer setup

    This may include enabling gradient checkpointing and disabling cache usage.

    Parameters
    ----------
    model: Any
        Loaded model object
    use_gradient_checkpointing: bool, default=True
        Whether to enable activation checkpointing.
    
    Returns
    -------
    Any 
        Prepared model object.
    """
    model.train()

    if hasattr(model, "config"):
        model.config.use_cache = use_cache

    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model


def build_lora_config(config: dict[str, Any]) -> LoraConfig:
    """
    Build PEFT LoRA configuration from config values.

    Parameters
    ----------
    config: dict[str, Any]
        LoRA config section (rank, alpha, dropout, targets, etc.).
    
    Returns
    -------
    LoraConfig
        LoRA configuration object for adapter injection.
    """
    if not bool(config.get("lora_enabled", False)):
        raise ValueError("LoRA is disabled in config (`lora_enabled=False`).")
    
    r = config.get("lora_r")
    alpha = config.get("lora_alpha")
    dropout = config.get("lora_dropout", 0.05)
    bias = config.get("lora_bias", "none")
    task_type = config.get("lora_task_type", "CAUSAL_LM")
    target_modules = config.get("target_modules")

    if not isinstance(r, int) or r <= 0:
        raise ValueError(f"Invalid LoRA rank `lora_r`: {r}")
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError(f"Invalid LoRA alpha `lora_alpha`: {alpha}")
    if not isinstance(dropout, (int, float)) or not (0.0 <= float(dropout) < 1.0):
        raise ValueError(f"Invalid LoRA dropout `lora_dropout`: {dropout}")
    if bias not in {"none", "all", "lora_only"}:
        raise ValueError(f"Invalid LoRA bias `lora_bias`: {bias}")
    if isinstance(target_modules, str):
        target_modules = target_modules.strip()
        if not target_modules:
            raise ValueError("`target_modules` must be non-empty when provided as a string.")
        deduped_target_modules: str | list[str] = target_modules
    else:
        if not isinstance(target_modules, list) or not target_modules:
            raise ValueError("`target_modules` must be a non-empty list or a non-empty string.")

        cleaned_target_modules = []
        for item in target_modules:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Invalid target module entry: {item!r}")
            cleaned_target_modules.append(item.strip())

        # Deduplicate while preserving order.
        deduped_target_modules = list(dict.fromkeys(cleaned_target_modules))

    return LoraConfig(
        r=r,
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias=bias,
        task_type=task_type,
        target_modules=deduped_target_modules,
    )    
            

def attach_lora(model: Any, lora_config: LoraConfig) -> Any:
    """
    Attach LoRA adapters to the base model using PEFT.

    Parameters
    ----------
    model: Any
        Base model to adapt.
    lora_config: LoraConfig
        LoRA configuration object.

    Returns 
    -------
    Any
        Adapter-augmented model.
    """
    try:
        lora_model = get_peft_model(model, lora_config)
    except Exception as exc:
        raise RuntimeError("Failed to attach LoRA adapters to model.") from exc
    
    # Helpful sanity signal for logs.
    if hasattr(lora_model, "print_trainable_parameters"):
        lora_model.print_trainable_parameters()

    return lora_model

def load_model_bundle(config: dict[str, Any]) -> dict[str, Any]:
    """
    Orchestrate end-to-end model loading pipeline.

    This function should:
    1. Resolve model/quantization settings,
    2. Load processor and base model,
    3. Apply training preparation and optional LoRA.

    Parameters
    ----------
    config: dict[str, Any]
        Full runtime config dictionary.
    
    Returns
    -------
    dict[str, Any]
        Bundle containing model, processor, and resolved runtime metadata.
    """
    
    if not isinstance(config, dict):
        raise TypeError("`config` must be a normalized config dictionary.")
    
    dtype_map: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    torch_dtype_name = str(config.get("torch_dtype", "bfloat16")).lower()
    torch_dtype = dtype_map.get(torch_dtype_name)
    if torch_dtype is None:
        raise ValueError(
            f"Unsupported torch_dtype: {torch_dtype_name}."
            f"Expected one of: {sorted(dtype_map.keys())}"
        )
    
    use_tf32 = bool(config.get("use_tf32", True))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32

    quant_config = build_quantization_config(config)

    processor = load_processor(
        model_id = config["model_id"],
        trust_remote_code = bool(config.get("trust_remote_code", True)),
    )

    model_kwargs: dict[str, Any] = {}
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config["attn_implementation"]

    model = load_model(
        model_id = config["model_id"],
        quant_config=quant_config,
        torch_dtype=torch_dtype,
        device_map=config.get("device_map", "auto"),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        **model_kwargs,
    )

    model = prepare_model_for_training(
        model=model,
        use_gradient_checkpointing=bool(config.get("training_gradient_checkpointing", True)),
        use_cache=bool(config.get("training_use_cache", False)),
    )

    lora_config = None
    if bool(config.get("lora_enabled", False)):
        lora_config = build_lora_config(config)
        model = attach_lora(model, lora_config)

    return {
        "model": model,
        "processor": processor,
        "lora_config": lora_config,
        "quantization_config": quant_config,
        "torch_dtype": torch_dtype,
        "resolved_config": config,
    }
