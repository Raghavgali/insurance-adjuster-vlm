from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset splitting.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed runtime arguments including input dataset path, output directory,
        split ratio, seed, and optional image-copy settings.

    Notes
    -----
    - This script is dataset-format aware but not model-specific.
    - CLI inputs are preferred over hardcoded paths for reproducibility.
    """
    def _test_size(value: str) -> float:
        parsed = float(value)
        if not 0.0 < parsed < 1.0:
            raise argparse.ArgumentTypeError("`test_size` must be between 0 and 1")
        return parsed

    parser = argparse.ArgumentParser(description="Create deterministic train/test splits for the cleaned dataset.")
    parser.add_argument("input", type=Path, help="Path to the cleaned dataset JSON file.")
    parser.add_argument("output_dir", type=Path, help="Directory where split JSON files will be written.")
    parser.add_argument("--test-size", type=_test_size, default=0.1, help="Fraction of samples assigned to test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional directory containing source images referenced by the dataset.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="If set, copy split images into output_dir/images/train and output_dir/images/test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview split counts and sample ids without writing files.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict]:
    """
    Load and validate the cleaned dataset JSON file.

    Parameters
    ----------
    path: Path
        Path to the cleaned dataset JSON file.

    Returns
    -------
    list[dict]
        Dataset records ready for deterministic splitting.

    Notes
    -----
    - Expects a top-level JSON list of sample dictionaries.
    - Performs lightweight validation only; content-level cleanup should already be done.
    """
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {resolved_path}")

    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved_path}: {exc}") from exc

    if not isinstance(data, list):
        raise TypeError(f"Expected top-level JSON list in {resolved_path}, got {type(data).__name__}")
    if not data:
        raise ValueError(f"Dataset is empty: {resolved_path}")
    if not all(isinstance(item, dict) for item in data):
        raise TypeError(f"All dataset entries must be dicts in {resolved_path}")

    logger.info("Loaded %s samples from %s", len(data), resolved_path)
    return data


def validate_dataset_records(records: list[dict]) -> None:
    """
    Validate core fields required for splitting and optional image copying.

    Parameters
    ----------
    records: list[dict]
        Dataset records loaded from disk.

    Returns
    -------
    None

    Notes
    -----
    - Requires each sample to contain non-empty `id`, `image`, and `conversations`.
    - Raises early on duplicate ids to keep split outputs stable.
    """
    seen_ids: set[str] = set()
    for idx, record in enumerate(records):
        record_id = record.get("id")
        image_name = record.get("image")
        conversations = record.get("conversations")

        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"Record at index {idx} has invalid `id`")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate dataset id detected: {record_id}")
        seen_ids.add(record_id)

        if not isinstance(image_name, str) or not image_name.strip():
            raise ValueError(f"Record {record_id} has invalid `image` field")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f"Record {record_id} has invalid or empty `conversations` field")


def filter_records_with_images(records: list[dict], image_root: Path | None) -> list[dict]:
    """
    Optionally filter records whose referenced image files are missing.

    Parameters
    ----------
    records: list[dict]
        Dataset records to validate.
    image_root: Path | None
        Source image directory. When None, no filtering is applied.

    Returns
    -------
    list[dict]
        Filtered records containing only entries with existing images when image_root is provided.

    Notes
    -----
    - This does not mutate input records.
    - Missing-image filtering is conservative and logged clearly.
    """
    if image_root is None:
        return list(records)

    resolved_root = Path(image_root).expanduser().resolve()
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"`image_root` is not a directory: {resolved_root}")

    filtered = [record for record in records if (resolved_root / record["image"]).is_file()]
    dropped = len(records) - len(filtered)
    if dropped > 0:
        logger.warning("Dropped %s samples with missing images under %s", dropped, resolved_root)
    if not filtered:
        raise ValueError(f"No samples remain after image validation under {resolved_root}")
    return filtered


def split_dataset(records: list[dict], *, test_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    """
    Split dataset records into deterministic train and test partitions.

    Parameters
    ----------
    records: list[dict]
        Cleaned dataset records.
    test_size: float
        Fraction of samples assigned to the test split.
    seed: int
        Random seed used for deterministic splitting.

    Returns
    -------
    tuple[list[dict], list[dict]]
        Train records and test records.

    Notes
    -----
    - Records are sorted by `id` before splitting for deterministic input ordering.
    - Shuffling is performed with a dedicated RNG seeded by `seed`.
    """
    ordered_records = sorted(records, key=lambda item: str(item["id"]))
    rng = random.Random(seed)
    shuffled_records = list(ordered_records)
    rng.shuffle(shuffled_records)

    test_count = max(1, math.ceil(len(shuffled_records) * test_size))
    test_records = shuffled_records[:test_count]
    train_records = shuffled_records[test_count:]
    if not train_records:
        raise ValueError("Split configuration leaves no samples for the train split")
    return train_records, test_records


def save_json(data: list[dict], path: Path) -> Path:
    """
    Save one split to a JSON file.

    Parameters
    ----------
    data: list[dict]
        Split records to persist.
    path: Path
        Output JSON path.

    Returns
    -------
    Path
        Resolved output path.

    Notes
    -----
    - Creates parent directories automatically.
    - Uses stable indentation for human inspection and reproducibility.
    """
    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    logger.info("Saved %s samples to %s", len(data), resolved_path)
    return resolved_path


def copy_split_images(records: list[dict], *, image_root: Path, destination_dir: Path) -> int:
    """
    Copy split images into a destination directory.

    Parameters
    ----------
    records: list[dict]
        Split records whose image files should be copied.
    image_root: Path
        Source image directory.
    destination_dir: Path
        Destination directory for copied images.

    Returns
    -------
    int
        Number of images copied successfully.

    Notes
    -----
    - Missing source images are skipped with a warning.
    - The dataset JSON is not rewritten; image filenames remain unchanged.
    """
    resolved_root = Path(image_root).expanduser().resolve()
    resolved_destination = Path(destination_dir).expanduser().resolve()
    resolved_destination.mkdir(parents=True, exist_ok=True)

    copied = 0
    for record in records:
        src = resolved_root / record["image"]
        dst = resolved_destination / record["image"]
        if not src.is_file():
            logger.warning("Skipping missing image: %s", src)
            continue
        shutil.copy2(src, dst)
        copied += 1

    logger.info("Copied %s images to %s", copied, resolved_destination)
    return copied


def run_split(args: argparse.Namespace) -> dict[str, Any]:
    """
    Execute the complete split workflow from CLI arguments.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments from `parse_args()`.

    Returns
    -------
    dict[str, Any]
        Summary payload including split counts and output paths.

    Notes
    -----
    - Dry-run mode returns the same summary without writing files.
    - Image copying is optional and requires `--image-root`.
    """
    records = load_dataset(args.input)
    validate_dataset_records(records)
    records = filter_records_with_images(records, args.image_root)

    train_records, test_records = split_dataset(
        records,
        test_size=float(args.test_size),
        seed=int(args.seed),
    )

    summary: dict[str, Any] = {
        "input_path": str(Path(args.input).expanduser().resolve()),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "total_samples": len(records),
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "train_preview_ids": [record["id"] for record in train_records[:5]],
        "test_preview_ids": [record["id"] for record in test_records[:5]],
    }

    if args.dry_run:
        return summary

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_json_path = save_json(train_records, output_dir / "train.json")
    test_json_path = save_json(test_records, output_dir / "test.json")
    summary["train_json"] = str(train_json_path)
    summary["test_json"] = str(test_json_path)

    if args.copy_images:
        if args.image_root is None:
            raise ValueError("`--copy-images` requires `--image-root`")
        train_copy_count = copy_split_images(
            train_records,
            image_root=args.image_root,
            destination_dir=output_dir / "images" / "train",
        )
        test_copy_count = copy_split_images(
            test_records,
            image_root=args.image_root,
            destination_dir=output_dir / "images" / "test",
        )
        summary["train_images_copied"] = train_copy_count
        summary["test_images_copied"] = test_copy_count

    return summary


def main() -> None:
    """
    CLI entrypoint for deterministic train/test splitting.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - Exits non-zero on fatal errors.
    - Keeps dry-run output human-readable for quick validation.
    """
    args = parse_args()

    try:
        summary = run_split(args)
        if args.dry_run:
            logger.info("Dry run summary:")
            print(json.dumps(summary, indent=2))
        else:
            logger.info(
                "Split complete: train=%s test=%s output_dir=%s",
                summary["train_samples"],
                summary["test_samples"],
                summary["output_dir"],
            )
    except Exception:
        logger.exception("Train/test split failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
