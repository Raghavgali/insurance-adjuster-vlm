from __future__ import annotations
from pathlib import Path

import torch
import json

def load_annotations(annotation_path: str) -> list[dict]:
    """
    Load annotation records from a JSON or JSONL file.

    Parameters
    ----------
    annotation_path: str
        Path to the annotation file.

    Returns
    -------
    list[dict]
        A list of raw annotation dictionaries, one per sample

    Notes
    -----
    - Raises a clear exception if the file is missing, empty, or malformed.
    - Keeps Validation lightweight; deep schema/content checks belong in validate_data.py
    """
    path = Path(annotation_path).expanduser().resolve()
    suffix = path.suffix.lower()

    try:
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                records = [data]
            elif isinstance(data, list):
                records = data
            else:
                raise ValueError(f"Expected JSON object or list, got {type(data).__name__}")
            
        elif suffix == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError(
                            f"Expected JSON object on line {line_no}, got {type(obj).__name__}"
                        )
                    records.append(obj)
        else:
            raise ValueError(f"Unsupported annotation format: {path.name} (use .json or .jsonl)")
        
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as e:
        raise RuntimeError(f"Failed to load annotations from '{path}': {e}") from e
    
    if not records:
        raise RuntimeError(f"No annotation records found in '{path}'")

    if not all(isinstance(item, dict) for item in records):
        raise RuntimeError(f"Annotations in '{path}' must be a list of JSON objects")

    return records     


def resolve_paths(sample: dict, image_root: str) -> dict:
    """
    Resolve sample file references (for example image paths) into absolute paths.

    Parameters
    ----------
    sample: dict 
        One raw annotation sample dictionary.
    image_root: str
        Base directory used to resolve relative image paths.

    Returns 
    -------
        A normalized sample dictionary with resolved absolute file paths.

    Notes
    -----
    - Verifies required keys exist before returning.
    - Validates that referenced files exist on disk.
    - Should not mutate the input sample in-place.
    """
    if not isinstance(sample, dict):
        raise TypeError(f"`sample` must be a dict, got {type(sample).__name__}")
    
    if "image" not in sample:
        raise KeyError("Missing required key 'image' in sample")
    
    image_value = sample["image"]
    if not isinstance(image_value, str) or not image_value.strip():
        raise ValueError("`sample['image']` must be a non-empty string path")
    
    root = Path(image_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"image_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"image_root is not a directory: {root}")
    
    img_path = Path(image_value).expanduser()
    abs_img_path = img_path if img_path.is_absolute() else (root / img_path)
    abs_img_path = abs_img_path.resolve()

    if not abs_img_path.exists():
        raise FileNotFoundError(f"Image file not found: {abs_img_path}")
    if not abs_img_path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {abs_img_path}")
    
    resolved_sample = dict(sample) # avoid in-place mutation
    resolved_sample["image"] = str(abs_img_path)

    return resolved_sample


def normalize_sample(sample: dict) -> dict:
    """
    Normalize one raw sample into the internal dataset schema.

    Parameters
    ----------
    sample: dict
        One raw annotation sample dictionary.
    
    Returns 
    -------
        A normalized sample dictionary with standardized field names and structure.

    Notes
    -----
    - Keep this deterministric and side-effect free.
    - Do not run tokenization or tensor conversion here.
    - Ensure output keys are stable for Dataset and collator usage.
    """
    if not isinstance(sample, dict):
        raise TypeError(f"`sample` must be dict got {type(sample).__name__}")
    
    normalized: dict[str, object] = {}

    # id
    if "id" not in sample:
        raise KeyError("Missing required key `id` in sample")
    raw_id = sample["id"]
    if isinstance(raw_id, (int, float)):
        raw_id = str(raw_id)
    elif not isinstance(raw_id, str):
        raise TypeError("`sample[id]` must be str/int/float") 
    norm_id = raw_id.strip()
    if not norm_id:
        raise ValueError("`sample['id']` must be non-empty")
    normalized["id"] = norm_id

    # image
    if "image" not in sample:
        raise KeyError("Missing required key `image` in sample")
    raw_img = sample["image"]
    if not isinstance(raw_img, str):
        raise TypeError("`sample['image']` must be str")
    norm_img = raw_img.strip()
    if not norm_img:
        raise ValueError("`sample['image']` must be non-empty")
    normalized["image"] = norm_img

    # conversations/message
    raw_conv = sample.get("conversations", sample.get("messages"))
    if raw_conv is None:
        raise KeyError("Missing required key 'conversations' (or 'messages') in sample")
    
    if not isinstance(raw_conv, list) or len(raw_conv) == 0:
        raise ValueError("'conversations' must be a non-empty list")

    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }

    norm_conv: list[dict[str, str]] = []
    for i, turn in enumerate(raw_conv):
        if not isinstance(turn, dict):
            raise TypeError(f"Conversation turn {i} must be dict, got {type(turn).__name__}")

        if "from" in turn and "value" in turn:
            raw_role = turn["from"]
            raw_text = turn["value"]
        elif "role" in turn and "content" in turn:
            raw_role = turn["role"]
            raw_text = turn["content"]
        else:
            raise KeyError(
                f"Conversation turn {i} must contain ('from','value') or ('role','content')"
            )

        if not isinstance(raw_role, str):
            raise TypeError(f"Conversation role at turn {i} must be str")
        role_key = raw_role.strip().lower()
        if role_key not in role_map:
            raise ValueError(f"Unsupported conversation role '{raw_role}' at turn {i}")

        if not isinstance(raw_text, str):
            raise TypeError(f"Conversation content at turn {i} must be str")
        text = raw_text.strip()
        if not text:
            raise ValueError(f"Conversation content at turn {i} cannot be empty")

        norm_conv.append({"role": role_map[role_key], "content": text})

    normalized["conversations"] = norm_conv
    return normalized


def filter_invalid_samples(samples: list[dict],
                           *,
                           image_root: str | None = None,
                           strict: bool = False) -> list[dict]:
    """
    Filter out invalid samples before dataset construction.

    Parameters
    ----------
        samples: list[dict]
            List of normalized sample dictionaries.
        strict: bool = False
            If True, raise an error on first invalid sample instead of skipping.

    Returns 
    -------
        A list containing only valid samples.

    Notes
    -----
    - Non-strict mode should skip bad samples and continue.
    - Strict mode should fail fast with actionable error context. 
    - Log or report dropped-sample counts for observability.
    """
    if not isinstance(samples, list):
        raise TypeError(f"`samples` must be a list got {type(samples).__name__}")
    
    valid_samples: list[dict] = []

    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            err = TypeError(f"Sample at index {idx} must be dict, got {type(sample).__name__}")
            if strict:
                raise err
            continue

        try:
            normalized = normalize_sample(sample)
            if image_root is not None:
                normalized = resolve_paths(normalized, image_root=image_root)
            valid_samples.append(normalized)
        except (TypeError, KeyError, ValueError, FileNotFoundError, NotADirectoryError) as e:
            sample_id = sample.get("id", f"idx={idx}")
            if strict:
                raise ValueError(f"Invalid sample ({sample_id}) at index {idx}: {e}") from e
            continue

    return valid_samples


class VisionLanguageDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for vision-language training data.

    Parameters
    ----------
        samples: list[dict]
            List of validated and normalized sample dictionaries
        split: str
            Dataset split name (for example 'train' or 'val').

    Returns 
    -------
        An indexable dataset object that yields one sample dictionary per item.

    Notes
    -----
    - Load heavy assets (for example image) lazily in __getitem___.
    - Return raw sample-level objects expected by the collator.
    - Avoid batch-level logic in this class.
    """
    def __init__(self, samples: list[dict], split: str) -> None:
        if not isinstance(samples, list):
            raise TypeError(f"`samples` must be list, got {type(samples).__name__}")
        if not isinstance(split, str) or not split.strip():
            raise ValueError("`split` must be a non-empty string")
        self.samples = samples
        self.split = split.strip()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if not isinstance(idx, int):
            raise TypeError(f"`idx` must be int, got {type(idx).__name__}")
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index out of range: {idx}")
        return dict(self.samples[idx])


def build_dataset(
        annotation_path: str,
        image_root: str,
        *,
        split: str,
        strict: bool = False,
) -> VisionLanguageDataset:
    """
    Build a dataset instance from annotation and image-root inputs.

    Parameters
    ----------
    annotation_path: str
        Path to the annotation file for the requested split.
    image_root: str
        Root directory containing image files.
    split: str
        Split Identifier used for logging and metadata.
    strict: bool = False
        If True, fail on invalid samples; otherwise skip invalid entries

    Returns 
    -------
        A VisionLanguageDataset instance ready for DataLoader usage.

    Notes
    -----
    - Centralizes loading, normalization, path resolution, and filtering.
    - Keeps training code simple by exposing one consistent entry point.
    """
    if not isinstance(annotation_path, str) or not annotation_path.strip():
        raise ValueError("`annotation_path` must be a non-empty string")
    if not isinstance(image_root, str) or not image_root.strip():
        raise ValueError("`image_root` must be a non-empty string")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("`split` must be a non-empty string")

    raw_samples = load_annotations(annotation_path)
    valid_samples = filter_invalid_samples(
        raw_samples,
        image_root=image_root,
        strict=strict,
    )

    if not valid_samples:
        raise ValueError(
            f"No valid samples after filtering (annotation_path='{annotation_path}', split='{split}')"
        )

    return VisionLanguageDataset(samples=valid_samples, split=split)
