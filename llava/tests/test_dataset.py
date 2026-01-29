from llava.scripts.prepare_data import LlavaNextDataset
import os

def test_data_preparation():
    """
    Tests whether train and test datasets are loaded correctly and contain valid entries.
    """
    train_path = '/Users/raghavg/Desktop/Datasets and Projects/Insurance/llava/dataset/train.json'
    train_images_path = '/Users/raghavg/Desktop/Datasets and Projects/Insurance/llava/dataset/images/train'
    test_path = '/Users/raghavg/Desktop/Datasets and Projects/Insurance/llava/dataset/test.json'
    test_images_path = '/Users/raghavg/Desktop/Datasets and Projects/Insurance/llava/dataset/images/test'

    assert os.path.exists(train_path), f"Train path does not exist: {train_path}"
    assert os.path.exists(test_path), f"Test path does not exist: {test_path}"

    try:
        train_dataset = LlavaNextDataset(annotations_file=train_path,
                                         images_dir=train_images_path, split='train')
        test_dataset = LlavaNextDataset(annotations_file=test_path,
                                        images_dir=test_images_path, split='test')
    except Exception as e:
        raise AssertionError(f"Failed to load datasets: {e}")

    assert len(train_dataset) > 0, "Train dataset is empty"
    assert len(test_dataset) > 0, "Test dataset is empty"

    sample = train_dataset[0]
    assert 'image' in sample, "Sample missing 'image' key"
    assert 'prompt' in sample, "Sample missing 'prompt' key"

    assert isinstance(sample['image'], object), "Sample 'image' is not a valid image object"