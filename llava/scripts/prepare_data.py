import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from torch.utils.data import Dataset


class LlavaNextDataset(Dataset):
    """
    Custom Dataset creation: Loads JSON objects, opens images and packages data
    """

    def __init__(self, annotations_file: str, images_dir: str, split: str):
        annotations_path = Path(annotations_file).expanduser().resolve()
        if not annotations_path.is_file():
            raise FileNotFoundError(f"Missing annotations file at {annotations_path}")

        with annotations_path.open("r", encoding="utf-8") as fin:
            self.samples: List[Dict[str, Any]] = json.load(fin)

        self.images_dir = Path(images_dir).expanduser().resolve()
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing images directory at {self.images_dir}")

        self.split = split


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = self.images_dir / item["image"]
        image = Image.open(image_path).convert("RGB")

        messages = item["conversations"]
        prompt = ""
        answer = ""

        if self.split == "train":
            for msg in messages:
                role = msg["role"]
                content = msg["content"].strip()
                prompt += f"{role}: {content}\n"
                if role == "assistant":
                    answer = content

        elif self.split == "test":
            for msg in messages:
                role = msg["role"]
                content = msg["content"].strip()
                if role == "assistant":
                    answer = content
                    continue
                if role == "user":
                    prompt += f"user: {content}\n"

        return {
            "prompt": prompt.strip(),
            "image": image,
            "answer": answer.strip(),
        }


class LlavaNextDatacollator:
    def __init__(self, processor, max_length: int = 256, is_train: bool = True):
        self.processor = processor
        self.max_length = max_length
        self.is_train = is_train

    def __call__(self, batch):
        texts = [ex["prompt"] for ex in batch]
        images = [ex["image"] for ex in batch]

        batch_inputs = self.processor(
            texts=texts,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        if self.is_train:
            labels = batch_inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch_inputs["labels"] = labels

        # Preserve answers for evaluation, Lightning will keep them on CPU.
        batch_inputs["answers"] = [ex.get("answer", "") for ex in batch]

        return batch_inputs
