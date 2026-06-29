"""设备选择辅助函数。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DeviceContext:
    """保存本次运行选择的 torch 设备和可见 GPU 数量。"""

    device: torch.device
    n_gpu: int


def resolve_device(device_config: Any = "auto") -> DeviceContext:
    """将用户设备配置转换为 torch.device 和 GPU 数量。"""

    if isinstance(device_config, (list, tuple)):
        return _resolve_gpu_list(device_config)
    if isinstance(device_config, str):
        return _resolve_device_string(device_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    return DeviceContext(device=device, n_gpu=n_gpu)


def _resolve_gpu_list(device_config: list | tuple) -> DeviceContext:
    """将 CUDA 可见范围限制为配置的物理 GPU ID。"""

    if not device_config:
        return DeviceContext(device=torch.device("cpu"), n_gpu=0)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in device_config)
    if not torch.cuda.is_available():
        return DeviceContext(device=torch.device("cpu"), n_gpu=0)
    return DeviceContext(device=torch.device("cuda"), n_gpu=torch.cuda.device_count())


def _resolve_device_string(device_config: str) -> DeviceContext:
    """处理 cpu、auto、cuda 和 cuda:N 设备字符串。"""

    normalized = device_config.lower()
    if normalized == "cpu":
        return DeviceContext(device=torch.device("cpu"), n_gpu=0)
    if normalized == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
        return DeviceContext(device=device, n_gpu=n_gpu)
    if normalized.startswith("cuda") and torch.cuda.is_available():
        n_gpu = 1 if ":" in normalized else torch.cuda.device_count()
        return DeviceContext(device=torch.device(device_config), n_gpu=n_gpu)
    return DeviceContext(device=torch.device("cpu"), n_gpu=0)
