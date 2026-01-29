import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

def load_qwen_model(model_id="Qwen/Qwen2.5-VL-7B-Instruct", quantize=True):
    """
    Loads the Qwen2.5-VL-Instruct model optionally in 8-bit quantized form.
    Returns (model, processor).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_8bit = True
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=(torch.float16 if not quantize else None)
    )

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')

    return model, processor, tokenizer