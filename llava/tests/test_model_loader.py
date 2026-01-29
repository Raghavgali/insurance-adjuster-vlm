from unittest.mock import MagicMock, patch

import pytest
import torch

from llava.scripts.model_loader import (
    QuantizationConfig,
    apply_lora,
    find_all_linear_names,
    load_llava_model,
    load_llava_processor,
)


def test_find_all_linear_names_discards_multimodal_layers():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_proj = torch.nn.Linear(4, 4)
            self.multi_modal_projector = torch.nn.Linear(4, 4)

    model = DummyModel()
    names = find_all_linear_names(model)

    assert names == ["text_proj"]


@patch("llava.scripts.model_loader.LlavaNextForConditionalGeneration.from_pretrained")
def test_load_llava_model_without_quantization(mock_from_pretrained):
    model_id = "test/model"
    expected = MagicMock()
    mock_from_pretrained.return_value = expected

    result = load_llava_model(model_id)

    mock_from_pretrained.assert_called_once_with(
        model_id,
        torch_dtype=QuantizationConfig().torch_dtype,
        quantization_config=None,
    )
    assert result is expected


@patch("llava.scripts.model_loader.LlavaNextForConditionalGeneration.from_pretrained")
def test_load_llava_model_with_4bit(mock_from_pretrained):
    model_id = "test/model"
    expected = MagicMock()
    mock_from_pretrained.return_value = expected

    quant = QuantizationConfig(use_4bit=True)
    result = load_llava_model(model_id, quantization=quant)

    _, kwargs = mock_from_pretrained.call_args
    quant_config = kwargs["quantization_config"]

    assert quant_config.load_in_4bit is True
    assert quant_config.bnb_4bit_compute_dtype == torch.float16
    assert result is expected


def test_quantization_config_raises_when_both_modes_enabled():
    quant = QuantizationConfig(use_4bit=True, use_8bit=True)
    with pytest.raises(ValueError):
        quant.to_bnb_config()


@patch("llava.scripts.model_loader.AutoProcessor.from_pretrained")
def test_load_llava_processor_sets_padding_side(mock_from_pretrained):
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    mock_from_pretrained.return_value = processor

    result = load_llava_processor("model-id")

    assert processor.tokenizer.padding_side == "right"
    assert result is processor


@patch("llava.scripts.model_loader.get_peft_model")
@patch("llava.scripts.model_loader.prepare_model_for_kbit_training")
def test_apply_lora_invokes_peft_helpers(mock_prepare_kbit, mock_get_peft):
    model = MagicMock()
    mock_prepare_kbit.return_value = model
    mock_get_peft.return_value = model
    lora_config = MagicMock()

    result = apply_lora(model, lora_config)

    mock_prepare_kbit.assert_called_once_with(model)
    mock_get_peft.assert_called_once_with(model, lora_config)
    assert result is model
