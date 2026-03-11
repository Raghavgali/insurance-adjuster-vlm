from __future__ import annotations
from pathlib import Path

import re
import os 
from huggingface_hub import HfApi
from huggingface_hub import login as hf_login
from huggingface_hub import snapshot_download


def resolve_hf_token(token: str | None = None) -> str | None:
    """
    Resolve the Hugging Face access token from runtime imports. 

    Token precedence is:
    1. Explicit `token` argument 
    2. `HF_TOKEN` environment variable 
    3. `HUGGINGFACE_HUB_TOKEN` environment variable

    Parameters
    ----------
    token: str | None, optional
        Explicit Hugging Face token passed by the caller. 

    Returns 
    -------
    str | None 
        Resolved token if available; otherwise `None`.

    Notes
    -----
    This function must not log or print token values.
    """
    if token is not None:
        token = token.strip()
        if token:
            return token 
        
    for env_var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.getenv(env_var)
        if value is not None:
            value = value.strip()
            if value:
                return value            
    return None


def ensure_hf_login(
        token: str | None = None,
        *,
        required: bool = True,
        add_to_git_credential: bool = False,
) -> str | None:
    """
    Authenticate the current process with Hugging Face Hub.

    Parameters
    ----------
    token: str | None, optional
        Token to use for login. If omitted, resolution should be relegated to 
        `resolve_hf_token`.
    required: bool, default=True
        If `True`, raise an error when no token can be resolved. If `False`, 
        return `None` and skip login when token is missing.
    add_to_git_credential: bool, default=False
        Whether to persist credentials to the local git credential helper.

    Returns
    -------
    str | None
        The token used for login, or `None` if login is skipped and not required.

    Raises
    ------
    ValueError
        If `required=True` and no token is available.
    RuntimeError
        If Hugging Face login fails.

    Notes 
    -----
    The function should be safe to call multiple times and should never expose secret value in logs.
    """
    resolved_token = resolve_hf_token(token)

    if not resolved_token:
        if required:
            raise ValueError(
                "No Hugging Face token found. Provide `token` or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
            )
        return None
    
    try:
        hf_login(token=resolved_token, add_to_git_credential=add_to_git_credential)
    except Exception as exc:
        raise RuntimeError("Hugging Face login failed.") from exc
    
    return resolved_token


def download_dataset_snapshot(
        repo_id: str,
        local_dir: str | Path,
        *,
        revision: str = "main",
        allow_patterns: list[str] | None = None,
        token: str | None = None,
) -> Path:
    """
    Download a dataset repository snapshot from Hugging Face Hub to local storage.

    Parameters 
    ---------
    repo_id: str
        Dataset repository identifier in the form `namespace/name`.
    local_dir: str | Path
        Destination directory where the snapshot will be materialized.
    revision: str, default="main"
        Branch, tag, or commit to download for reproducibility.
    allow_patterns: list[str] | None, optional
        Optional file globs to limit downloaded files (for faster startup).
    token: str | None, optional
        Huggin Face token used for authenticated downloads.

    Returns
    -------
    Path 
        Resolved local path of the downloaded snapshot.

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., malformed `repo_id`).
    RuntimeError
        If snapshot download fails.

    Notes
    -----
    Implementation should use `repo_type="dataset"` explicitly.
    """
    repo_id = repo_id.strip()
    if not re.fullmatch(r"^[^/]+/[^/]+$", repo_id):
        raise ValueError(f"repo_id must be in 'namespace/name' format")
    
    if not revision or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    
    local_dir = Path(local_dir).expanduser().resolve()
    if local_dir.exists() and not local_dir.is_dir():
        raise ValueError(f"local_dir points to a file: {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    resolved_token = ensure_hf_login(token=token, required=False)

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            token=resolved_token,
            repo_type="dataset",
            revision=revision.strip(),
            local_dir=local_dir,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset snapshot for repo '{repo_id}' at revision '{revision}'."
        ) from exc
    
    return Path(local_dir).resolve()


def upload_dataset_folder(
        repo_id: str,
        dataset_dir: str | Path,
        *,
        revision: str = "main",
        private: bool | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        commit_message: str = "Upload dataset snapshot",
        token: str | None = None,
) -> str:
    """
    Upload a local dataset directory to a Hugging Face dataset repository.

    Parameters
    ----------
    repo_id: str
        Dataset repository identifier in the form `namespace/name`.
    dataset_dir: str | Path
        Local directory containing dataset assets such as JSON files and images.
    revision: str, default="main"
        Branch name to upload into.
    private: bool | None, optional
        Whether to create the dataset repo as private when it does not already exist.
        If `None`, Hugging Face defaults are used.
    allow_patterns: list[str] | None, optional
        Optional file globs to include during upload.
    ignore_patterns: list[str] | None, optional
        Optional file globs to exclude during upload.
    commit_message: str, default="Upload dataset snapshot"
        Commit message attached to the upload operation.
    token: str | None, optional
        Hugging Face token used for authenticated uploads.

    Returns
    -------
    str
        The commit URL or upload reference returned by the Hugging Face Hub client.

    Raises
    ------
    ValueError
        If inputs are invalid (for example malformed `repo_id` or non-directory `dataset_dir`).
    RuntimeError
        If repository creation or folder upload fails.

    Notes
    -----
    - This function uploads the entire dataset directory, which is the simplest path for
      image-plus-JSON datasets.
    - Repository type is always set to `dataset`.
    """
    repo_id = repo_id.strip()
    if not re.fullmatch(r"^[^/]+/[^/]+$", repo_id):
        raise ValueError("repo_id must be in 'namespace/name' format")
    if not revision or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    if not isinstance(commit_message, str) or not commit_message.strip():
        raise ValueError("commit_message must be a non-empty string")

    resolved_dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not resolved_dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {resolved_dataset_dir}")
    if not resolved_dataset_dir.is_dir():
        raise NotADirectoryError(f"dataset_dir is not a directory: {resolved_dataset_dir}")

    resolved_token = ensure_hf_login(token=token, required=True)
    api = HfApi(token=resolved_token)

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to create or access dataset repo '{repo_id}'.") from exc

    try:
        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(resolved_dataset_dir),
            path_in_repo=".",
            revision=revision.strip(),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            commit_message=commit_message.strip(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to upload dataset folder '{resolved_dataset_dir}' to repo '{repo_id}'."
        ) from exc

    return str(commit_info)
