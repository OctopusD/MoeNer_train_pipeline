"""序列化辅助函数。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def to_plain_data(value: Any) -> Any:
    """递归地将 dataclass 和容器转换为普通 Python 值。"""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_plain_data(item) for item in value)
    return value

