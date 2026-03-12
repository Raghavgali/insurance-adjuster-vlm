from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair train/test image split directories so they match train.json and test.json."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset root containing train.json, test.json, and images/train + images/test.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files into the correct split instead of moving them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview repairs without changing any files.",
    )
    return parser.parse_args()


def _load_expected_names(json_path: Path) -> set[str]:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"Expected top-level list in {json_path}")

    names: set[str] = set()
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Record at index {idx} in {json_path} must be a dict")
        image_name = record.get("image")
        if not isinstance(image_name, str) or not image_name.strip():
            raise ValueError(f"Record at index {idx} in {json_path} has invalid `image`")
        names.add(Path(image_name).name)
    return names


def _disk_names(image_dir: Path) -> set[str]:
    return {path.name for path in image_dir.iterdir() if path.is_file()}


def repair_dataset_split_images(
    dataset_dir: str | Path,
    *,
    copy: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    root = Path(dataset_dir).expanduser().resolve()
    train_json = root / "train.json"
    test_json = root / "test.json"
    train_dir = root / "images" / "train"
    test_dir = root / "images" / "test"

    for path in (train_json, test_json):
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
    for path in (train_dir, test_dir):
        if not path.is_dir():
            raise NotADirectoryError(f"Missing split image directory: {path}")

    train_expected = _load_expected_names(train_json)
    test_expected = _load_expected_names(test_json)
    train_disk = _disk_names(train_dir)
    test_disk = _disk_names(test_dir)

    move_test_to_train = sorted((train_expected - train_disk) & test_disk)
    move_train_to_test = sorted((test_expected - test_disk) & train_disk)

    operation = shutil.copy2 if copy else shutil.move

    if not dry_run:
        for name in move_test_to_train:
            operation(str(test_dir / name), str(train_dir / name))
        for name in move_train_to_test:
            operation(str(train_dir / name), str(test_dir / name))

    train_disk_after = _disk_names(train_dir)
    test_disk_after = _disk_names(test_dir)

    summary = {
        "train_expected": len(train_expected),
        "test_expected": len(test_expected),
        "moved_to_train": len(move_test_to_train),
        "moved_to_test": len(move_train_to_test),
        "train_missing_after": len(train_expected - train_disk_after),
        "train_extra_after": len(train_disk_after - train_expected),
        "test_missing_after": len(test_expected - test_disk_after),
        "test_extra_after": len(test_disk_after - test_expected),
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = repair_dataset_split_images(
        args.dataset_dir,
        copy=bool(args.copy),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
