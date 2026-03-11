from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import argparse
import json

import torch
from PIL import Image


DEFAULT_PROMPT = (
    "You are an insurance adjuster. Analyze the vehicle damage shown in the image and "
    "write a concise professional assessment covering the damage type, location, severity, "
    "and estimated repair cost."
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the inference entrypoint.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Runtime arguments including config path, image path, optional checkpoint,
        prompt overrides, generation settings, and output destination.

    Notes
    -----
    - Keep inference arguments focused on deployment-style usage.
    - CLI overrides should take precedence over config defaults.
    """
    def _positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("Value must be a positive integer")
        return parsed

    parser = argparse.ArgumentParser(description="Single-image inference for GLM VLM.")
    parser.add_argument("--config", type=Path, required=True, help="Path to inference/training config YAML.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local checkpoint (.pt) to load on top of the base model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional user prompt override. Defaults to an insurance-adjuster instruction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. If omitted, the prediction is printed to stdout.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="Optional precision override for inference.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=_positive_int,
        default=None,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--num-beams",
        type=_positive_int,
        default=None,
        help="Beam width for generation. Use 1 for greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Only used when --do-sample is enabled.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic decoding instead of deterministic greedy/beam search.",
    )
    return parser.parse_args()


def build_inference_components(config: dict[str, Any]) -> dict[str, Any]:
    """
    Build model and processor components required for single-image inference.

    Parameters
    ----------
    config: dict[str, Any]
        Normalized configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Inference components including model, processor, tokenizer, device, and precision.

    Notes
    -----
    - Reuses the shared model loader so inference follows the same runtime configuration.
    - Forces a single-device placement instead of DDP-oriented loading.
    """
    if not isinstance(config, dict):
        raise TypeError(f"`config` must be dict got {type(config).__name__}")
    if not config:
        raise ValueError("`config` cannot be empty")

    from GLM.scripts.model_loader import load_model_bundle

    model_cfg = config.get("model", {})
    precision_cfg = config.get("precision", {})
    quant_cfg = config.get("quantization", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("train", config.get("training", {}))
    infer_cfg = config.get("inference", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision_name = str(infer_cfg.get("precision", "fp16")).lower()

    model_loader_cfg: dict[str, Any] = {
        "model_id": config.get("model_id", model_cfg.get("model_id")),
        "trust_remote_code": bool(config.get("trust_remote_code", model_cfg.get("trust_remote_code", True))),
        "attn_implementation": config.get("attn_implementation", model_cfg.get("attn_implementation")),
        "device_map": {"": 0} if device.type == "cuda" else None,
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
        "training_use_cache": True,
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

    model_bundle = load_model_bundle(config=model_loader_cfg)
    model = model_bundle["model"]
    processor = model_bundle["processor"]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor does not expose tokenizer")

    if device.type == "cuda" and not bool(model_loader_cfg.get("quantization_enabled", False)):
        model = model.to(device)
    model.eval()

    return {
        "model_bundle": model_bundle,
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "device": device,
        "precision": precision_name,
    }


def load_checkpoint_for_inference(
    model: torch.nn.Module,
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> None:
    """
    Optionally load model weights from a local checkpoint for inference.

    Parameters
    ----------
    model: torch.nn.Module
        Model instance to load weights into.
    checkpoint_path: str | Path | None
        Optional checkpoint path. When None, loading is skipped.
    device: torch.device
        Device used for map_location while loading checkpoint tensors.

    Returns
    -------
    None

    Notes
    -----
    - Supports checkpoints that store either `model` or `model_state_dict`.
    - Leaves the model unchanged when no checkpoint path is provided.
    """
    if checkpoint_path is None:
        return

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


def build_prompt_text(prompt: str) -> str:
    """
    Convert a raw user instruction into the text format expected by the model.

    Parameters
    ----------
    prompt: str
        User instruction for the model.

    Returns
    -------
    str
        Serialized prompt text used for processor tokenization.

    Notes
    -----
    - Keeps inference aligned with the simple role-prefixed format used elsewhere in the project.
    - Does not inject any dataset metadata into the prompt.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"`prompt` must be str got {type(prompt).__name__}")

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("`prompt` must be a non-empty string")

    return prompt


def run_inference(
    *,
    model: torch.nn.Module,
    processor: Any,
    tokenizer: Any,
    image_path: str | Path,
    prompt: str,
    device: torch.device,
    precision: str,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    temperature: float | None,
) -> dict[str, Any]:
    """
    Run single-image generation and return the structured inference result.

    Parameters
    ----------
    model: torch.nn.Module
        Loaded model in evaluation mode.
    processor: Any
        Multimodal processor used to tokenize text and image inputs.
    tokenizer: Any
        Tokenizer used for decoding generated token ids.
    image_path: str | Path
        Path to the input image.
    prompt: str
        User instruction for the model.
    device: torch.device
        Device used for inference.
    precision: str
        Precision mode (`fp32`, `fp16`, or `bf16`) for autocast selection.
    max_new_tokens: int
        Maximum number of generated tokens.
    num_beams: int
        Beam-search width.
    do_sample: bool
        Whether stochastic sampling is enabled.
    temperature: float | None
        Optional sampling temperature.

    Returns
    -------
    dict[str, Any]
        Structured inference payload containing prompt, image path, and generated text.

    Notes
    -----
    - Uses prompt-only generation with no label leakage.
    - Returns JSON-safe output suitable for saving or serving.
    """
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("`max_new_tokens` must be a positive integer")
    if not isinstance(num_beams, int) or num_beams <= 0:
        raise ValueError("`num_beams` must be a positive integer")
    if temperature is not None and temperature <= 0:
        raise ValueError("`temperature` must be > 0 when provided")

    resolved_image_path = Path(image_path).expanduser().resolve()
    if not resolved_image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {resolved_image_path}")

    prompt_text = build_prompt_text(prompt)
    from GLM.scripts.collator import render_chat_text

    with Image.open(resolved_image_path) as img:
        rgb_image = img.convert("RGB")

    rendered_text = render_chat_text(
        processor,
        conversations=[{"role": "user", "content": prompt_text}],
        image_path=str(resolved_image_path),
        add_generation_prompt=True,
        include_assistant=False,
    )

    inputs = processor(
        text=[rendered_text],
        images=[rgb_image],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    inputs.pop("token_type_ids", None)

    use_autocast = device.type == "cuda" and precision in {"fp16", "bf16"}
    if use_autocast:
        autocast_context = torch.autocast(
            device_type="cuda",
            dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
        )
    else:
        autocast_context = nullcontext()

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": do_sample,
    }
    if do_sample and temperature is not None:
        generation_kwargs["temperature"] = float(temperature)

    target_model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        with autocast_context:
            generated_ids = target_model.generate(**inputs, **generation_kwargs)

    attention_mask = inputs.get("attention_mask")
    prompt_length = int(inputs["input_ids"].shape[1]) if attention_mask is not None else int(inputs["input_ids"].shape[1])
    new_token_ids = generated_ids[0][prompt_length:]
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    if not generated_text:
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    return {
        "image_path": str(resolved_image_path),
        "prompt": prompt,
        "response": generated_text,
        "generation_config": generation_kwargs,
    }


def save_inference_output(result: dict[str, Any], output_path: str | Path) -> Path:
    """
    Save a structured inference result to disk as JSON.

    Parameters
    ----------
    result: dict[str, Any]
        Inference payload produced by `run_inference()`.
    output_path: str | Path
        Destination JSON file path.

    Returns
    -------
    Path
        Resolved output path of the written JSON file.

    Notes
    -----
    - Creates parent directories automatically.
    - Writes human-readable JSON for debugging and downstream usage.
    """
    if not isinstance(result, dict):
        raise TypeError(f"`result` must be dict got {type(result).__name__}")

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    return resolved_output_path


def main() -> None:
    """
    Inference entrypoint coordinating config loading, model setup, generation, and output.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - Keeps inference independent from evaluation-only metrics and distributed logic.
    - Prints the generated response when no output path is provided.
    """
    from GLM.scripts.utils.load_config import load_yaml_config

    args = parse_args()
    config = load_yaml_config(args.config)
    if not isinstance(config, dict):
        raise TypeError("Config loader must return a dictionary")

    config.setdefault("inference", {})
    infer_cfg = config["inference"]

    if args.precision is not None:
        infer_cfg["precision"] = args.precision
    if args.max_new_tokens is not None:
        infer_cfg["max_new_tokens"] = args.max_new_tokens
    if args.num_beams is not None:
        infer_cfg["num_beams"] = args.num_beams
    if args.temperature is not None:
        infer_cfg["temperature"] = args.temperature
    infer_cfg["do_sample"] = bool(args.do_sample)

    components = build_inference_components(config=config)
    load_checkpoint_for_inference(
        model=components["model"],
        checkpoint_path=args.checkpoint,
        device=components["device"],
    )

    prompt = args.prompt or str(infer_cfg.get("prompt", DEFAULT_PROMPT))
    result = run_inference(
        model=components["model"],
        processor=components["processor"],
        tokenizer=components["tokenizer"],
        image_path=args.image,
        prompt=prompt,
        device=components["device"],
        precision=str(infer_cfg.get("precision", components["precision"])).lower(),
        max_new_tokens=int(infer_cfg.get("max_new_tokens", 256)),
        num_beams=int(infer_cfg.get("num_beams", 1)),
        do_sample=bool(infer_cfg.get("do_sample", False)),
        temperature=float(infer_cfg["temperature"]) if infer_cfg.get("temperature") is not None else None,
    )

    if args.output is not None:
        output_path = save_inference_output(result=result, output_path=args.output)
        print(f"[inference_done] output={output_path}")
    else:
        print(result["response"])


if __name__ == "__main__":
    main()
