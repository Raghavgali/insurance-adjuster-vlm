from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
)


@dataclass
class QuantizationConfig:
    use_4bit: bool = False
    use_8bit: bool = False
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    torch_dtype: torch.dtype = torch.float16

    def to_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        if self.use_4bit and self.use_8bit:
            raise ValueError("Only one of `use_4bit` or `use_8bit` can be enabled.")

        if self.use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            )

        if self.use_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)

        return None


def load_llava_model(model_id: str, quantization: Optional[QuantizationConfig] = None) -> LlavaNextForConditionalGeneration:
    quantization = quantization or QuantizationConfig()
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=quantization.torch_dtype,
        quantization_config=quantization.to_bnb_config(),
    )
    return model


def load_llava_processor(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
    return processor


def find_all_linear_names(model: torch.nn.Module) -> list[str]:
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_model"]
    for name, module in model.named_modules():
        if any(keyword in name for keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    lora_module_names.discard("lm_head")
    return sorted(lora_module_names)


def apply_lora(model: torch.nn.Module, lora_config: LoraConfig) -> torch.nn.Module:
    return get_peft_model(model, lora_config)
