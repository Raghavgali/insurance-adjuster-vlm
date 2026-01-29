from llava.scripts.prepare_data import LlavaNextDataset, LlavaNextDatacollator
from unittest.mock import MagicMock
from PIL import Image
import torch

def test_data_collator():
    """
    Tests whether the data collator is batching the data correctly
    """
    train_path = '/Users/raghavg/Desktop/Datasets and Projects/Insurance/llava/dataset/train.json'
    train_images_path = '/Users/raghavg/Desktop/Datasets and Projects/Insurance/llava/dataset/images/train'
    test_path = '/Users/raghavg/Desktop/Datasets and Projects/Insurance/llava/dataset/test.json'
    test_images_path = '/Users/raghavg/Desktop/Datasets and Projects/Insurance/llava/dataset/images/test'

    try: 
        train_dataset = LlavaNextDataset(annotations_file=train_path,
                                         images_dir=train_images_path,
                                         split='train')
        test_dataset = LlavaNextDataset(annotations_file=test_path,
                                        images_dir=test_images_path,
                                        split='test')
    except Exception as e:
        raise AssertionError(f"Failed to load datasets: {e}")
    
    # Create a mock processor
    mock_processor = MagicMock()
    mock_processor.tokenizer.pad_token_id = 0
    mock_processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
        "pixel_values": torch.randn(2, 3, 224, 224)
    }

    collator = LlavaNextDatacollator(processor=mock_processor, max_length=4, is_train=True)

    batch_samples = [train_dataset[0], train_dataset[1]]  # simulate a small batch
    collated = collator(batch_samples)

    assert "input_ids" in collated
    assert "attention_mask" in collated
    assert "pixel_values" in collated
    assert "labels" in collated
    assert collated["labels"].shape == collated["input_ids"].shape