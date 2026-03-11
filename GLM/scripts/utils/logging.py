from __future__ import annotations

import os
import logging 
from pathlib import Path 

def _get_rank() -> int:
    # torchrun sets RANK; LOCAL_RANK is a fallback for some launch setups.
    raw_rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
    try:
        return int(raw_rank)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Invalid rank value '%s'. Falling back to rank 0.", raw_rank
        )
        return 0

def _is_main_process() -> bool:
    return _get_rank() == 0


def setup_logger(
        logger_name: str = "glm", 
        level: int | str = logging.INFO,
        file_path: str | Path | None = None,
        console_main_only: bool = True,
        file_per_rank: bool = True,
    ) -> logging.Logger:
    """
    Sets up logging function 
    
    :param logger_name: Description
    :type logger_name: str
    :param level: Description
    :type level: int | str
    :param file_path: Description
    :type file_path: str | Path | None
    :param console_main_only: Description
    :type console_main_only: bool
    :param file_per_rank: Description
    :type file_per_rank: bool
    :return: Description
    :rtype: Logger
    """
    rank = _get_rank()

    if isinstance(level, str):
        level_value = getattr(logging, level.upper(), None)
        if not isinstance(level_value, int):
            raise ValueError(f"Invalid log level: {level}")
        else:
            level_value = level

    logger = logging.getLogger(logger_name)
    logger.setLevel(level_value)
    logger.propagate = False

    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s | rank={rank} | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S",
    )


    # Console Handler (typically rank 0 only in DDP)
    should_add_console = (not console_main_only) or _is_main_process()
    if should_add_console:
        has_console = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        )
        if not has_console:
            sh = logging.StreamHandler()
            sh.setLevel(level_value)
            sh.setFormatter(formatter)
            logger.addHandler(sh)

    # Optional File Handler 
    if file_path is not None:
        path = Path(file_path).expanduser().resolve()
        if file_per_rank and rank != 0:
            suffix = path.suffix if path.suffix else ".log"
            base = path.name[:-len(suffix)] if path.suffix else path.name
            path = path.with_name(f"{base}.rank{rank}{suffix}")

        path.parent.mkdir(parents=True, exist_ok=True)

        has_same_file = any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")).resolve() == path
            for h in logger.handlers
        )
        if not has_same_file:
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setLevel(level_value)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger