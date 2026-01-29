from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def minimal_config():
    return {
        "paths": {
            "train_json": "data/train.json",
            "train_image_folder": "data/train_images",
            "test_json": "data/val.json",
            "test_image_folder": "data/val_images",
        },
        "model": {
            "model_id": "llava/test-model",
            "repo_id": "user/repo",
            "max_length": 32,
            "wand_project": "proj",
            "wandb_name": "run",
        },
        "api_keys": {"hugging_face": None, "wandb": None},
        "training": {
            "batch_size": 2,
            "max_epochs": 1,
            "accumulate_grad_batches": 1,
            "check_val_every_n_epoch": 1,
            "gradient_clip_val": 1.0,
            "lr": 1e-4,
        },
        "logging": {},
        "quantization": {"use_4bit": False, "use_8bit": False},
    }


@pytest.mark.usefixtures("minimal_config")
def test_run_training_invokes_trainer(monkeypatch, minimal_config):
    import llava.scripts.train as train_module

    mock_load_yaml = MagicMock(return_value=minimal_config)
    mock_hf_login = MagicMock(return_value=False)
    mock_wandb_login = MagicMock(return_value=False)
    monkeypatch.setattr(train_module, "load_yaml_config", mock_load_yaml)
    monkeypatch.setattr(train_module, "ensure_hf_login", mock_hf_login)
    monkeypatch.setattr(train_module, "maybe_login_wandb", mock_wandb_login)
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_module.torch, "set_float32_matmul_precision", lambda *_, **__: None)

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_load_processor = MagicMock(return_value=mock_processor)
    mock_load_model = MagicMock(return_value=mock_model)
    monkeypatch.setattr(train_module, "load_llava_processor", mock_load_processor)
    monkeypatch.setattr(train_module, "load_llava_model", mock_load_model)

    mock_dataset = MagicMock()
    mock_dataset_ctor = MagicMock(return_value=mock_dataset)
    monkeypatch.setattr(train_module, "LlavaNextDataset", mock_dataset_ctor)

    train_loader = MagicMock(name="train_loader")
    val_loader = MagicMock(name="val_loader")
    mock_create_dataloader = MagicMock(side_effect=[train_loader, val_loader])
    monkeypatch.setattr(train_module, "create_dataloader", mock_create_dataloader)

    module_instance = MagicMock()
    mock_module_ctor = MagicMock(return_value=module_instance)
    monkeypatch.setattr(train_module, "LlavaPLModule", mock_module_ctor)

    trainer_instance = MagicMock()
    mock_trainer_ctor = MagicMock(return_value=trainer_instance)
    monkeypatch.setattr(train_module.L, "Trainer", mock_trainer_ctor)

    args = SimpleNamespace(
        config=Path("/tmp/test_config.yaml"),
        devices="auto",
        precision="16-mixed",
        num_workers=None,
        limit_val_batches=None,
    )

    train_module.run_training(args)

    mock_load_yaml.assert_called_once()
    mock_hf_login.assert_called_once_with(None)
    mock_wandb_login.assert_called_once_with(None)
    mock_load_processor.assert_called_once_with("llava/test-model")
    mock_load_model.assert_called_once()
    assert mock_create_dataloader.call_count == 2
    mock_module_ctor.assert_called_once()
    trainer_instance.fit.assert_called_once_with(
        module_instance,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
