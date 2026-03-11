from __future__ import annotations
from pathlib import Path

import argparse
import json 


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset validation.

    Expected inputs include the dataset root path (or config path) and optional
    behavior flags (for example, strict mode or max error reporting).

    Returns 
    -------
    argparse.Namespace
        Parsed CLI arguments used by the validation workflow.

    Notes
    -----
    CLI overrides should take precedence over config-provided values.
    """
    parser = argparse.ArgumentParser(
        description="Validate dataset structure, annotations, and image links."
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=100,
        help="Stop after N errors to avoid huge output.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Validate only first N records for quick smoke checks.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Validate a specific split only.",
    )
    parser.add_argument(
        "--check-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable expensive image existence checks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-error details instead of summary only.",
    )

    return parser.parse_args()


def validate_structure(dataset_root: Path) -> list[str]:
    """
    Validate required dataset structure on disk.

    This check should verify that required files and directories exist, such as
    metadata/annotation files, split files, and image folders expected by training.

    Parameters
    ----------
    dataset_root : Path
        Root directory containing the dataset snapshot.

    Returns
    -------
    list[str]
        List of human-readable validation errors. Empty list means structure is valid.

    Notes
    -----
    Keep this check filesytem-only; do not parse full annotation content here.
    """
    errors = []
    
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        return [f"Dataset root does not exist: {dataset_root}"]
    if not dataset_root.is_dir():
        return [f"Dataset root is not a directory: {dataset_root}"]
    
    required_files = [
        dataset_root / "train.json",
        dataset_root / "test.json",
    ]

    required_dirs = [
        dataset_root / "images",
        dataset_root / "images" / "train",
        dataset_root / "images" / "test",
    ]

    for file_path in required_files:
        if not file_path.exists():
            errors.append(f"Missing required files: {file_path}")
        elif not file_path.is_file():
            errors.append(f"Expected file but found non-file path: {file_path}")

    for dir_path in required_dirs:
        if not dir_path.exists():
            errors.append(f"Missing required directory: {dir_path}")
        elif not dir_path.is_dir():
            errors.append(f"Expected directory but found non-directory path: {dir_path}")

    
    for split_dir in (dataset_root / "images" / "train", dataset_root / "images" / "test"):
        if split_dir.exists() and split_dir.is_dir():
            has_files = any(split_dir.iterdir())
            if not has_files:
                errors.append(f"Directory is empty: {split_dir}")
    
    return errors
    
    
def validate_annotations(dataset_root: Path) -> list[str]:
    """
    Validate annotation content and minimal schema integrity.

    This check should inspect JSON records and ensure required keys exist, expected
    value types are present, and conversation/message fields are well-formed.

    Parameters
    ----------
    dataset_root : Path
        Root directory containing annotation files to validate.

    Returns
    -------
    list[str]
        List of annotation/schema validation errors. Empty list means annotations pass.
    
    Notes
    -----
    Prefer collecting multiple errors instead of failing on the first bad record.
    """
    errors = []

    dataset_root = Path(dataset_root).expanduser().resolve()

    required_files = [
        dataset_root / "train.json",
        dataset_root / "test.json",
    ]

    for file_path in required_files:
        if not file_path.exists():
            errors.append(f"Missing annotation file: {file_path}")
            continue
        if not file_path.is_file():
            errors.append(f"Expected file but found non-file path: {file_path}")
            continue

        try:
            with file_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except json.JSONDecodeError as exc:
            errors.append(f"{file_path}: invalid JSON ({exc})")
            continue
        except OSError as exc:
            errors.append(f"{file_path}: failed to read file ({exc})")
            continue

        if not isinstance(records, list):
            errors.append(f"{file_path}: expected top-level list, got {type(records).__name__}")
            continue

        for idx, sample in enumerate(records):
            if not isinstance(sample, dict):
                errors.append(f"{file_path} [record {idx}]: each sample must be an object")
                continue

            if "id" not in sample or not isinstance(sample["id"], str) or not sample["id"].strip():
                errors.append(f"{file_path} [record {idx}]: invalid or missing 'id'")

            if "image" not in sample or not isinstance(sample["image"], str) or not sample["image"].strip():
                errors.append(f"{file_path} [record {idx}]: invalid or missing 'image'")

            convs = sample.get("conversations")
            if not isinstance(convs, list) or len(convs) < 2:
                errors.append(f"{file_path} [record {idx}]: 'conversations' must be a list with at least 2 entries")
                continue

            roles: list[str] = []
            for msg_idx, msg in enumerate(convs):
                if not isinstance(msg, dict):
                    errors.append(f"{file_path} [record {idx}] [msg {msg_idx}]: message must be an object")
                    continue

                role = msg.get("role")
                content = msg.get("content")

                if role not in {"user", "assistant"}:
                    errors.append(
                        f"{file_path} [record {idx}] [msg {msg_idx}]: role must be 'user' or 'assistant'"
                    )
                else:
                    roles.append(role)

                if not isinstance(content, str) or not content.strip():
                    errors.append(
                        f"{file_path} [record {idx}] [msg {msg_idx}]: content must be a non-empty string"
                    )

            if "user" not in roles or "assistant" not in roles:
                errors.append(
                    f"{file_path} [record {idx}]: conversations must include both user and assistant roles"
                )

    return errors


def validate_image_links(dataset_root: Path) -> list[str]:
    """
    Validate that annotation image references resolve to actual files.

    This check should confirm that each sample's `image` field maps to an existing 
    file under the expected image directories.

    Parameters
    ----------
    dataset_root: Path
        Root directory containing annotations and image assets.
    
    Returns 
    -------
    list[str]
        List of missing/broken image reference errors. Empty list means all links resolve.

    Notes
    -----
    Paths should be resolved relative to dataset structure, not current working directory. 
    """
    errors: list[str] = []
    dataset_root = dataset_root.expanduser().resolve()

    split_specs = [
        ("train", dataset_root / "train.json", dataset_root / "images" / "train"),
        ("test", dataset_root / "test.json", dataset_root / "images" / "test"),
    ]

    for split_name, ann_path, image_dir in split_specs:
        if not ann_path.exists() or not ann_path.is_file():
            errors.append(f"{split_name}: missing annotation file: {ann_path}")
            continue
        if not image_dir.exists() or not image_dir.is_dir():
            errors.append(f"{split_name}: missing image directory: {image_dir}")
            continue

        try:
            with ann_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except json.JSONDecodeError as exc:
            errors.append(f"{split_name}: invalid JSON in {ann_path} ({exc})")
            continue
        except OSError as exc:
            errors.append(f"{split_name}: failed to read {ann_path} ({exc})")
            continue

        if not isinstance(records, list):
            errors.append(f"{split_name}: expected top-level list in {ann_path}")
            continue

        for idx, sample in enumerate(records):
            image_name = sample.get("image") if isinstance(sample, dict) else None
            if not isinstance(image_name, str) or not image_name.strip():
                errors.append(f"{split_name}: record {idx} has invalid image field")
                continue
            image_path = image_dir / image_name
            if not image_path.exists() or not image_path.is_file():
                errors.append(f"{split_name}: record {idx} references missing image: {image_path}")

    return errors

def run_validation(args: argparse.Namespace) -> int:
    """
    Run the full validation workflow and aggregate errors.

    This functions should:
    1. Resolve effective dataset root/config values.
    2. Run structure, annotation, and image-link checks.
    3. Aggregate and report findings.
    4. Return total error count.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed arguments from `parse_args()`.
    
    Returns
    -------
    int 
        Number of validation errors found. `0` indicates success.

    Notes
    -----
    This function should be deterministic and side-effect free (no data mutation).
    """
    dataset_root = args.dataset_path.expanduser().resolve()
    all_errors: list[str] = []

    # Validate structure first.
    all_errors.extend(validate_structure(dataset_root))

    if args.split == "val" and not (dataset_root / "val.json").is_file():
        all_errors.append(f"Requested split 'val' but missing file: {dataset_root / 'val.json'}")
    if args.split == "val" and args.check_images and not (dataset_root / "images" / "val").is_dir():
        all_errors.append(
            f"Requested split 'val' but missing image directory: {dataset_root / 'images' / 'val'}"
        )

    # Validate annotations.
    ann_errors = validate_annotations(dataset_root)
    if args.split != "all":
        ann_errors = [e for e in ann_errors if f"{args.split}.json" in e]
    all_errors.extend(ann_errors)

    # Validate image references if enabled.
    if args.check_images:
        img_errors = validate_image_links(dataset_root)
        if args.split != "all":
            img_errors = [e for e in img_errors if e.startswith(f"{args.split}:")]
        all_errors.extend(img_errors)

    # Cap output size.
    if args.max_errors is not None and args.max_errors > 0:
        all_errors = all_errors[: args.max_errors]

    if args.verbose:
        for err in all_errors:
            print(f"[ERROR] {err}")
    else:
        if all_errors:
            print(f"[ERROR] Found {len(all_errors)} validation issue(s).")
        else:
            print("[OK] Dataset validation passed.")

    return len(all_errors)


def main() -> None:
    """
    Program entry point for dataset validation.

    Responsibilities:
    - Parse CLI arguments.
    - Execute validation workflow.
    - Print/log a concise summary.
    - Exit with status code `0` on success and non-zero on failure.

    Returns
    -------
    None

    Notes
    -----
    Use non-zero exit codes for CI/CD and automation compatibility.
    """
    args = parse_args()
    error_count = run_validation(args)
    raise SystemExit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
