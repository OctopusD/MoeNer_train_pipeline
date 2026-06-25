"""日志设置辅助函数。"""

from __future__ import annotations

import logging
from pathlib import Path


def create_logger(name: str, output_dir: str, timestamp: str) -> logging.Logger:
    """创建同时写入 stdout 和带时间戳文件的 logger。"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(output_dir) / f"training_{timestamp}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

