from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset upload.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed arguments including config path and optional upload overrides.

    Notes
    -----
    - CLI values take precedence over config values when both are provided.
    - This script uploads a local folder into a Hugging Face dataset repository.
    """
    parser = argparse.ArgumentParser(description="Upload a local dataset folder to Hugging Face Hub.")
    parser.add_argument("--config", type=Path, default=None, help="Optional path to a YAML config file.")
    parser.add_argument("--repo-id", dest="repo_id", default=None, help="Dataset repo id in `namespace/name` format.")
    parser.add_argument("--dataset-dir", dest="dataset_dir", default=None, help="Local dataset directory to upload.")
    parser.add_argument("--revision", default=None, help="Target branch or revision for the upload.")
    parser.add_argument("--commit-message", default=None, help="Commit message for the dataset upload.")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private when it does not already exist.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token override. Prefer env vars for normal use.",
    )
    return parser.parse_args()


def run_upload(args: argparse.Namespace) -> str:
    """
    Execute dataset upload using config values with CLI override precedence.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments from `parse_args()`.

    Returns
    -------
    str
        Upload reference returned by the Hugging Face Hub client.

    Notes
    -----
    - Required values are `repo_id` and `dataset_dir`.
    - Config values are read from either top-level keys or the `data` section.
    """
    from GLM.scripts.utils.hf_utils import upload_dataset_folder

    config: dict = {}
    if args.config is not None:
        from GLM.scripts.utils.load_config import load_yaml_config
        config = load_yaml_config(args.config)

    data_cfg = config.get("data", {})

    repo_id = args.repo_id or config.get("repo_id") or data_cfg.get("repo_id")
    dataset_dir = args.dataset_dir or config.get("dataset_dir") or data_cfg.get("local_dir")
    revision = args.revision or config.get("revision") or data_cfg.get("revision") or "main"
    commit_message = (
        args.commit_message
        or config.get("commit_message")
        or data_cfg.get("commit_message")
        or "Upload dataset snapshot"
    )

    if not repo_id:
        raise ValueError("Missing required `repo_id` (CLI or config).")
    if not dataset_dir:
        raise ValueError("Missing required `dataset_dir` (CLI or config).")

    result = upload_dataset_folder(
        repo_id=repo_id,
        dataset_dir=dataset_dir,
        revision=revision,
        private=(True if args.private else data_cfg.get("private")),
        commit_message=commit_message,
        token=args.token,
    )
    return result


def main() -> None:
    """
    Program entrypoint for dataset upload.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - Emits a simple success line with the returned upload reference.
    - Exits non-zero on fatal upload failure.
    """
    from GLM.scripts.utils.logging import setup_logger

    args = parse_args()
    logger = setup_logger(logger_name="upload_dataset", level="INFO")

    try:
        upload_ref = run_upload(args)
        logger.info("Dataset upload completed: %s", upload_ref)
    except Exception:
        logger.exception("Dataset upload failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
