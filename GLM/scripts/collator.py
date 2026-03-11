from __future__ import annotations

from typing import Any

import torch
from PIL import Image


def build_multimodal_messages(
    *,
    conversations: list[dict[str, Any]],
    image_path: str | None,
    include_assistant: bool = True,
) -> list[dict[str, Any]]:
    """Build GLM-style multimodal chat messages from normalized dataset turns."""
    if not isinstance(conversations, list):
        raise TypeError(f"`conversations` must be list, got {type(conversations).__name__}")
    if not conversations:
        raise ValueError("`conversations` cannot be empty")

    messages: list[dict[str, Any]] = []
    image_attached = False

    for idx, item in enumerate(conversations):
        if not isinstance(item, dict):
            raise TypeError(f"Conversation turn {idx} must be dict, got {type(item).__name__}")

        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str):
            raise TypeError(f"`role` at turn {idx} must be str, got {type(role).__name__}")
        if not isinstance(content, str):
            raise TypeError(f"`content` at turn {idx} must be str, got {type(content).__name__}")

        role = role.strip().lower()
        content = content.strip()
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported role '{role}' at turn {idx}")
        if not content:
            raise ValueError(f"`content` at turn {idx} cannot be empty")

        if role == "assistant" and not include_assistant:
            continue

        content_items: list[dict[str, Any]] = []
        if role == "user" and not image_attached and image_path is not None:
            content_items.append({"type": "image", "path": image_path})
            image_attached = True
        content_items.append({"type": "text", "text": content})
        messages.append({"role": role, "content": content_items})

    return messages


def render_chat_text(
    processor: Any,
    *,
    conversations: list[dict[str, Any]],
    image_path: str | None,
    add_generation_prompt: bool,
    include_assistant: bool = True,
) -> str:
    """Render GLM chat text from multimodal messages using the processor chat template."""
    messages = build_multimodal_messages(
        conversations=conversations,
        image_path=image_path,
        include_assistant=include_assistant,
    )
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str) or not rendered.strip():
        raise ValueError("Processor chat template returned empty rendered text")
    return rendered


class DataCollator:
    """
    Batch collator for vision-language causal language model training.

    Parameters
    ----------
    processor: Hugging Face processor used for multimodal batching (text + images).
    tokenizer: Hugging Face tokenizer used for token-level operations and label masking.
    max_length: Maximum token length for truncation/padding.
    ignore_index: Label value used to mask tokens excluded from loss (default: -100).
    padding: Padding strategy passed to processor/tokenizer (for example True or "longest").
    truncation: Whether to truncate sequences longer than max_length.

    Returns
    -------
        A callable collator instance that can be passed to torch DataLoader.

    Notes
    -----
    - Input samples are executed to follow normalized schema from dataset.py.
    - Output batch keys should align with model forward arguments.
    - Keep heavy per-batch logic inside __call__ only.
    """
    def __init__(
        self,
        *,
        processor: Any,
        tokenizer: Any,
        max_length: int,
        ignore_index: int = -100,
        padding: bool | str = "longest",
        pad_to_multiple_of: int = 8,
        truncation: bool = True,
    ) -> None:
        if processor is None:
            raise ValueError("`processor` cannot be None")
        if tokenizer is None:
            raise ValueError("`tokenizer` cannot be None")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("`max_length` must be a positive integer")
        if not isinstance(ignore_index, int):
            raise TypeError("`ignore_index` must be an integer")
        if not isinstance(truncation, bool):
            raise TypeError("`truncation` must be a bool")
        if not (isinstance(padding, bool) or isinstance(padding, str)):
            raise TypeError("`padding` must be bool or str")
        if not isinstance(pad_to_multiple_of, int):
            raise TypeError("`pad_to_multuple_of` must be a int")

        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.truncation = truncation

    def _build_chat_text(self, conversations: list[dict], image_path: str | None = None) -> str:
        """
        Convert one sample's normalized conversation turns into a single training text string.

        Parameters
        ----------
        conversations: list[dict]
            list of turn dictionaries with canonical keys:
                - role: One of user/assistant/system
                - content: Text content for that role

        Returns
        -------
            A single serialized prompt/response string suitable for tokenization.

        Notes
        -----
        - Preserve role order exactly as provided.
        - Keep formatting deterministic so training is reproducible.
        - Raise a clear error for malformed or empty conversation turns.
        """
        return render_chat_text(
            self.processor,
            conversations=conversations,
            image_path=image_path,
            add_generation_prompt=False,
            include_assistant=True,
        )

    def build_generation_text(self, conversations: list[dict], image_path: str | None = None) -> str:
        """Render prompt-only text for generation."""
        return render_chat_text(
            self.processor,
            conversations=conversations,
            image_path=image_path,
            add_generation_prompt=True,
            include_assistant=False,
        )

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build language-model training labels from token IDs with proper masking.

        Parameters
        ---------
        input_ids: torch.Tensor
            Tensor of token IDs with shape [batch, seq_len].
        attention_mask: torch.Tensor
            Optional mask tensor where 1 indicates real tokens and 0 indicates padding.

        Returns 
        -------
            Label tensor with same shape as input_ids, where ignored positions are set to ignore_index.

        Notes
        -----
        - At minimum, mask padding tokens from loss.
        - If tokenizer has no pad_token_id, use attention_mask fallback for masking.
        - Function must not mutate input_ids in-place.
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"`input_ids` must be torch.Tensor, got {type(input_ids).__name__}")
        if input_ids.ndim != 2:
            raise ValueError(f"`input_ids` must be 2D [batch, seq_len], got shape {tuple(input_ids.shape)}")

        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(f"`attention_mask` must be torch.Tensor, got {type(attention_mask).__name__}")
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    f"`attention_mask` shape {tuple(attention_mask.shape)} must match "
                    f"`input_ids` shape {tuple(input_ids.shape)}"
                )

        labels = input_ids.clone()

        # 1) Mask padding tokens
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = self.ignore_index
        elif attention_mask is not None:
            labels[attention_mask == 0] = self.ignore_index

        # 2) Optional: mask common vision placeholder/image tokens if tokenizer defines them
        extra_vision_token_ids = [
            getattr(self, "image_token_id", None),
            getattr(self, "boi_token_id", None),
            getattr(self, "eoi_token_id", None),
        ]
        for token_id in extra_vision_token_ids:
            if token_id is not None:
                labels[labels == token_id] = self.ignore_index

        return labels

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate a list of normalized dataset samples into one model-ready batch.

        Parameters
        ----------
        samples: list[dict]
            List of sample dictionaries (each containing at least image path and conversations).

        Returns 
        -------
            Dictionary of tensors for model forward pass, typically including:
                - input_ids
                - attention_mask
                - labels
                - pixel_values (for vision inputs)

        Notes
        -----
        - Load/Prepare images for each sample before processor call.
        - Build text inputs using _build_chat_text.
        - Apply token/image processing in batched mode for efficiency.
        - Validate batch is non-empty and fail with actionable errors when malformed.
        """
        if not isinstance(samples, list):
            raise TypeError(f"`samples` must be list, got {type(samples).__name__}")
        if not samples:
            raise ValueError("`samples` cannot be empty")

        texts: list[str] = []
        images: list[Image.Image] = []
        sample_ids: list[str] = []

        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise TypeError(f"Sample at index {idx} must be dict, got {type(sample).__name__}")

            if "conversations" not in sample:
                raise KeyError(f"Missing 'conversations' in sample at index {idx}")
            if "image" not in sample:
                raise KeyError(f"Missing 'image' in sample at index {idx}")

            texts.append(self._build_chat_text(sample["conversations"], image_path=sample["image"]))
            sample_ids.append(str(sample.get("id", idx)))

            image_path = sample["image"]
            if not isinstance(image_path, str) or not image_path.strip():
                raise ValueError(f"`image` at index {idx} must be a non-empty string path")

            try:
                with Image.open(image_path) as img:
                    images.append(img.convert("RGB"))
            except OSError as e:
                raise ValueError(f"Failed to load image at index {idx} ('{image_path}'): {e}") from e

        batch = self.processor(
            text=texts,
            images=images,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if "input_ids" not in batch:
            raise KeyError("Processor output missing required key 'input_ids'")
        batch.pop("token_type_ids", None)

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        batch["labels"] = self._build_labels(input_ids=input_ids, attention_mask=attention_mask)
        batch["sample_ids"] = sample_ids
        return batch


    @classmethod
    def from_config(
        cls,
        *,
        processor,
        tokenizer,
        config: dict,
    ) -> "DataCollator":
        """
        Construct a DataCollator from normalized training/config dictionary values.

        Parameters
        ----------
        processor: Hugging Face processor instance.
        tokenizer: Hugging Face tokenizer instance.
        config: Configuration dictionary containing collator-related fields 
            (for example max_length, ignore_index, padding, truncation)

        Returns 
        -------
            Initialized DataCollator instance.

        Notes
        -----
        - Centralizes config extraction and default handling.
        - Validate required config keys and types before constructing the object.
        """
        if not isinstance(config, dict):
            raise TypeError(f"`config` must be dict, got {type(config).__name__}")

        max_length = config.get("max_length", config.get("model_max_length", 2048))
        ignore_index = config.get("ignore_index", -100)
        padding = config.get("padding", "longest")
        truncation = config.get("truncation", True)

        return cls(
            processor=processor,
            tokenizer=tokenizer,
            max_length=max_length,
            ignore_index=ignore_index,
            padding=padding,
            truncation=truncation,
        )
