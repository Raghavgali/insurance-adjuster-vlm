from __future__ import annotations
from pathlib import Path
import json
import argparse
import logging
import re


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[dict]:
    """
    Load a dataset from a JSON file.
       
    Args:
        path: Path to the JSON dataset file

    Returns:
        list of dataset entries

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        logger.error(f"Dataset file not found: {path}")
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected top-level JSON list in {path}, got {type(data).__name__}")
        logger.info(f"Loaded {len(data)} samples from {path}")
        return data 
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        raise


def clean_user_prompt(prompt: str) -> str:
    """
    Remove metadata leakage block from a user prompt while preserving the core instruction.

    Args:
        prompt: Raw user prompt text.

    Returns:
        Cleaned prompt text with metadata block removed.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"`prompt` must be str, got {type(prompt).__name__}")

    cleaned = re.sub(r"\n\s*Metadata:\s*[\s\S]*$", "", prompt, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_assistant_response(response: str) -> tuple[str, bool]:
    """
    Remove obvious annotation-derived leakage from an assistant response.

    Args:
        response: Raw assistant response text.

    Returns:
        Tuple of:
            - cleaned assistant response
            - boolean flag indicating whether any cleanup was applied
    """
    if not isinstance(response, str):
        raise TypeError(f"`response` must be str, got {type(response).__name__}")

    cleaned = response.strip()
    if not cleaned:
        return cleaned, False

    original = cleaned

    phrase_replacements = [
        (r"\bwhich is confirmed by the metadata\b", ""),
        (r"\bas indicated by the metadata\b", ""),
        (r"\baccording to the metadata\b", ""),
        (r"\bthe metadata indicates that\b", ""),
        (r"\bthe metadata indicates\b", ""),
        (r"\bmetadata indicates that\b", ""),
        (r"\bmetadata indicates\b", ""),
        (r"\bthe damage category noted as\s+\"[^\"]+\"\b", "the visible damage"),
        (r"\bthe damage category is\b", "the visible damage is"),
        (r"\bdamage categorized as\s+(an?\s+)?", ""),
        (r"\bdamage category of\s+(an?\s+)?", ""),
        (r"\bgiven the damage category as\s+(an?\s+)?", "given the visible damage as "),
        (r"\bfalls under the\s+([a-z\s-]+?)\s+damage category\b", r"is best described as \1"),
        (r"\bconsistent with the provided damage category of\s+\"?([a-z\s-]+)\"?\b", r"consistent with \1 damage"),
        (r"\bconsistent with the\s+\"?([a-z\s-]+)\"?\s+damage category\b", r"consistent with \1 damage"),
        (r"\bbased on the damage category and area\b", "based on the visible damage"),
        (r"\bbased on the damage category and extent\b", "based on the visible damage"),
        (r"\bbased on the damage category\b", "based on the visible damage"),
        (
            r"\bthe damage is categorized as\s+(an?\s+)?(?P<label>[A-Za-z][A-Za-z\s-]*)",
            r"the vehicle shows \g<label>",
        ),
        (
            r"\bthe damage category is identified as\s+(an?\s+)?(?P<label>[A-Za-z][A-Za-z\s-]*)",
            r"the vehicle shows \g<label>",
        ),
        (
            r"\b(?:the vehicle|the damage|the visible damage|damage)\s+(?:is\s+)?categorized as\s+(an?\s+)?(?P<label>[A-Za-z][A-Za-z\s-]*)",
            r"the vehicle shows \g<label>",
        ),
        (
            r"\bcategorized as\s+(an?\s+)?(?P<label>[A-Za-z][A-Za-z\s-]*)",
            r"showing \g<label>",
        ),
    ]

    for pattern, replacement in phrase_replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    clause_patterns = [
        r",\s*as indicated by the [^.?!,;]+",
        r"\sas indicated by the [^.?!,;]+",
        r",\s*as suggested by the [^.?!,;]+",
        r"\sas suggested by the [^.?!,;]+",
        r",\s*based on the [^.?!,;]*damage category[^.?!,;]*",
        r"\sbased on the [^.?!,;]*damage category[^.?!,;]*",
    ]
    for pattern in clause_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    filtered_sentences: list[str] = []
    removed_any = False

    drop_patterns = [
        r"\bmetadata\b",
        r"\bbounding box\b",
        r"\bbbox\b",
        r"\bsegmentation\b",
        r"\biscrowd\b",
        r"\bannotation\b",
        r"\bsquare units\b",
        r"\barea of\b",
        r"\baffected area\b",
        r"\[[^\]]+(?:,\s*[^\]]+){2,}\]",
    ]

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in drop_patterns):
            removed_any = True
            continue

        filtered_sentences.append(sentence)

    if filtered_sentences:
        cleaned = " ".join(filtered_sentences)

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\bshowing\s+(a|an)\s+([a-z\s-]+?)\s+damage\b", r"showing \1 \2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bthe vehicle shows\s+\"([^\"]+)\"", r"the vehicle shows \1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,", ", ", cleaned)
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    return cleaned, (removed_any or cleaned != original)

        
def clean_conversation(conversation: dict) -> dict:
    """
    Remove metadata from the user prompt in a conversation
    
    Args:
        conversation: A single dataset entry with 'conversations' key

    Returns:
        A new conversation dict with cleaned user prompt 
    """
    if not isinstance(conversation, dict):
        raise TypeError(f"`conversation` must be dict, got {type(conversation).__name__}")

    conversations = conversation.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        logger.warning(f"Skipping invalid conversation: {conversation.get('id')}")
        return conversation

    cleaned_turns: list[dict] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            cleaned_turns.append(turn)
            continue

        role = turn.get("role", turn.get("from"))
        text_key = "content" if "content" in turn else "value" if "value" in turn else None
        if role in {"user", "human"} and text_key is not None and isinstance(turn.get(text_key), str):
            cleaned_text = clean_user_prompt(turn[text_key])
            if cleaned_text:
                cleaned_turns.append({**turn, text_key: cleaned_text})
            else:
                cleaned_turns.append(turn)
        elif role in {"assistant", "gpt"} and text_key is not None and isinstance(turn.get(text_key), str):
            cleaned_text, _ = clean_assistant_response(turn[text_key])
            if cleaned_text:
                cleaned_turns.append({**turn, text_key: cleaned_text})
            else:
                cleaned_turns.append(turn)
        else:
            cleaned_turns.append(turn)

    return {
        **conversation,
        "conversations": cleaned_turns,
    }
    
    
def cleanup_dataset(data: list[dict]) -> list[dict]:
    """
    Clean metadata for all conversations in the dataset
    
    Args:
        data: List of conversation entries

    Returns:
        A new list with all conversations cleaned
    """
    cleaned = [clean_conversation(item) for item in data]
    logger.info(f"Cleaned {len(cleaned)} conversations")
    return cleaned


def save_dataset(data: list[dict], path: Path) -> None:
    """
    Save a dataset to a JSON file.

    Args:
        data: List of conversation entries to save
        path: Path to the output JSON file

    Raises:
        OSError: If the file cannot be written
    """
    output_dir = Path(path).expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    
    output_file = output_dir / "cleaned_dataset.json"

    try:
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} samples to {output_file}")
    except OSError as e:
        logger.error(f"Failed to save dataset to {output_file}: {e}")
        raise


def main() -> None:
    """CLI entry point for dataset cleanup."""
    parser = argparse.ArgumentParser(
        description="Remove metadata from LLaVA dataset conversations"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input JSON dataset file"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to output JSON dataset file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without saving"
    )

    args = parser.parse_args()

    data = load_dataset(args.input)
    cleaned_data = cleanup_dataset(data)

    if args.dry_run:
        logger.info("Dry run - showing first cleaned sample:")
        print(json.dumps(cleaned_data[0], indent=2))
    else:
        save_dataset(cleaned_data, args.output)
        logger.info("Dataset cleanup complete")


if __name__ == "__main__":
    main()
