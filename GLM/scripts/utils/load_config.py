from __future__ import annotations

import os 
from pathlib import Path 

import yaml 

def load_yaml_config(config_path: str| os.PathLike) -> dict:
    """
    Loads a YAML configuration file
    
    :param config_path: The path of the config file
    :type config_path: str | os.PathLike
    :return: Returns a config json file
    :rtype: dict
    """
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config File not found at {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")
    
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    
    if config is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(config, dict):
        raise TypeError(f"Top-level YAML must be a mapping/dict in {path}")
    
    return config