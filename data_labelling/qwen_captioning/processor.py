import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from src.model_loader import load_qwen_model
from PIL import Image

model, processor, tokenizer = load_qwen_model()

class StopOnEos(StoppingCriteria):
    """
    Custom Stopping criteria to halt generation when EOS token is encountered.
    """
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return (input_ids[0, -1].item() == self.eos_token_id)
    
def get_stopping_criteria(tokenizer):
    """
    Constructs the stopping criteria list using tokenizer's EOS token.
    """
    return StoppingCriteriaList([StopOnEos(tokenizer.eos_token_id)])


def build_message(image):
    """Constructs the input message structure for Qwen Multimodal model.
    
    Args:
        image (PIL.Image or tensor): Image Input
        prompt_text (str): Optional custom prompt

    Returns:
        list: Qwen-formatted message
    """

    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": """You are an Insurance adjuster. \
                        Complete a damage report for this vehicle in this image. Include details about the damage,
                        including the location and type of damage. If there is no damage, say 'No damage'."""
            }
        ]
    }]

    return message

