from __future__ import annotations
from pathlib import Path
import sys

import argparse

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset download.

    Expected arguments include a config path and optional runtime overrides 
    such as `repo_id`, `local_dir`, and `revision`

    Returns
    -------
    argparse.Namespace 
        Parsed arguments object containing CLI values.

    Notes
    -----
    CLI arguments should take precedence over config values when both are provided
    """
    parser = argparse.ArgumentParser(description="Argument parser for downloading dataset from Hugging Face Hub")
    parser.add_argument('--config', type=Path, help='Path to the config file')
    parser.add_argument('--repo_id', default=None, help='Dataset repository identifier in the form `namespace/name`')
    parser.add_argument('--local_dir', default=None, help='Destination directory where the snapshot will be materialized')
    parser.add_argument('--revision', default=None, help='Branch, tag, or commit to download for reproducibility (Default=main)')

    return parser.parse_args()


def run_download(args: argparse.Namespace) -> Path:
    """
    Execute dataset download using config + CLI overrides.

    This function should:
    1. Load YAML configuration.
    2. Resolve effective values (CLI override > config default).
    3. Call `download_dataset_snapshot(...)`.
    4. Return the resolved local dataset path.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments from `parse_args()`.

    Returns 
    -------
    Path
        Local filesystem path where the dataset snapshot is available.

    Raises 
    ------
    FileNotFoundError
        If the provided config path does not exist.
    ValueError
        If required runtime values (e.g., `repo_id`) are missing/invalid.
    RuntimeError
        If download fails.
    """
    from GLM.scripts.utils.hf_utils import download_dataset_snapshot

    config: dict = {}
    if args.config is not None:
        from GLM.scripts.utils.load_config import load_yaml_config
        config = load_yaml_config(args.config)

    repo_id = args.repo_id or config.get("repo_id") or config.get("data", {}).get("repo_id")
    local_dir = args.local_dir or config.get("local_dir") or config.get("data", {}).get("local_dir")
    revision = args.revision or config.get("revision") or config.get("data", {}).get("revision") or "main"

    if not repo_id:
        raise ValueError("Missing required `repo_id` (CLI or config).")
    if not local_dir:
        raise ValueError("Missing required `local_dir` (CLI or config).")
    
    dataset_path = download_dataset_snapshot(
        repo_id=repo_id,
        local_dir=local_dir,
        revision=revision,
    )

    return dataset_path
    

def main() -> None:
    """
    Program entry point for dataset download.

    Responsibilites:
    - Parse CLI arguments.
    - Run download workflow.
    - Emit success/failure logs.
    - Exit with non-zero status on fatal failure.

    Returns 
    -------
    None
    """
    from GLM.scripts.utils.logging import setup_logger

    args = parse_args()
    logger = setup_logger(logger_name="download_dataset", level="INFO")

    try:
        dataset_path = run_download(args)
        logger.info("Dataset ready at: %s", dataset_path)
    except Exception:
        logger.exception("Dataset download failed.")
        raise SystemExit(1)

if __name__ == '__main__':
    main()
