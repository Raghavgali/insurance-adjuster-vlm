import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from llava.scripts.lightning_module import LlavaPLModule


def _build_mock_batch(include_labels=True):
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        "pixel_values": torch.randn(2, 3, 4, 4),
        "answers": ["hello", "world"],
    }
    if include_labels:
        batch["labels"] = torch.tensor([[1, 2, -100], [3, 4, -100]])
    return batch


def test_training_step_logs_loss(monkeypatch):
    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_outputs.loss = torch.tensor(0.5)
    mock_model.return_value = mock_outputs

    mock_processor = MagicMock()

    module = LlavaPLModule(
        config={"lr": 1e-4},
        processor=mock_processor,
        model=mock_model,
        max_length=16,
    )

    batch = _build_mock_batch()
    loss = module.training_step(batch, batch_idx=0)

    assert torch.isclose(loss, torch.tensor(0.5))
    mock_model.assert_called_once()


def test_validation_step_drops_prompt_tokens(monkeypatch):
    mock_model = MagicMock()
    generated = torch.tensor(
        [
            [0, 1, 2, 7, 8],
            [0, 1, 2, 9, 10],
        ]
    )
    mock_model.generate.return_value = generated

    mock_processor = MagicMock()
    mock_processor.batch_decode.return_value = ["hi", "there"]

    module = LlavaPLModule(
        config={"lr": 1e-4},
        processor=mock_processor,
        model=mock_model,
        max_length=5,
    )

    batch = _build_mock_batch()
    module.validation_step(batch, batch_idx=0)

    mock_model.generate.assert_called_once()
    mock_processor.batch_decode.assert_called_once()
    decoded_input = mock_processor.batch_decode.call_args[0][0]
    assert decoded_input.shape[1] == 2  # prompt trimmed


def test_configure_optimizers_returns_adamw():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, **inputs):
            return SimpleNamespace(loss=torch.tensor(0.5, requires_grad=True))

        def generate(self, **inputs):
            return torch.ones((2, 2), dtype=torch.long)

    dummy_model = DummyModel()
    mock_processor = MagicMock()

    module = LlavaPLModule(
        config={"lr": 2e-4},
        processor=mock_processor,
        model=dummy_model,
        max_length=16,
    )

    optimizer = module.configure_optimizers()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert optimizer.param_groups[0]["params"], "Expected optimizer to receive model parameters."
