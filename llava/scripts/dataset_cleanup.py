from __future__ import annotations
from pathlib import Path
import json
import argparse
import logging


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
    if not path.exists():
        logger.error(f"Dataset file not found: {path}")
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples from {path}")
        return data 
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        raise
        
        
def clean_conversation(conversation: dict) -> dict:
    """
    Remove metadata from the user prompt in a conversation
    
    Args:
        conversation: A single dataset entry with 'conversations' key

    Returns:
        A new conversation dict with cleaned user prompt 
    """
    conversations = conversation.get('conversations')
    if not conversations or len(conversations) < 2:
        logger.warning(f"Skipping invalid conversation: {conversation.get('id')}")
        return conversation
    user_msg, assistant_msg = conversations[0], conversations[1]

    cleaned_prompt = (
        "You are an Insurance Adjuster. Evaluate the car damage shown in the image "
        "and provide a detailed report including damage type, location, severity,"
        "and estimated repair cost"
    )

    return {
        **conversation,
        "conversations": [
            {**user_msg, "content": cleaned_prompt},
            assistant_msg
        ]
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
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} samples to {path}")
    except OSError as e:
        logger.error(f"Failed to save dataset to {path}: {e}")
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
