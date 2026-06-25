"""用于保存和恢复训练状态的 checkpoint manager。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from model_training.training.state import TrainingState


class CheckpointManager:
    """处理模型、优化器、scheduler、配置、指标和状态 checkpoint。"""

    def __init__(self, save_dir: str | Path):
        """如果 checkpoint 目录不存在则创建。"""

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        state: TrainingState,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> str:
        """写入 checkpoint 文件并以字符串返回路径。"""

        path = self.save_dir / (name or f"epoch_{state.epoch}_step_{state.global_step}.pth")
        payload = {
            "model_state_dict": _unwrap_model(model).state_dict(),
            "training_state": state.to_dict(),
            "config": config,
            "metrics": metrics or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(payload, path)
        return str(path)

    def load_model(self, model: nn.Module, checkpoint_path: str | Path, strict: bool = True) -> TrainingState:
        """加载模型权重并返回 checkpoint 中保存的 TrainingState。"""

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=strict)
        return TrainingState.from_dict(checkpoint.get("training_state"))

    def load_full(
        self,
        model: nn.Module,
        checkpoint_path: str | Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
    ) -> TrainingState:
        """加载模型以及可选的优化器和 scheduler 状态。"""

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return TrainingState.from_dict(checkpoint.get("training_state"))


def _unwrap_model(model: nn.Module) -> nn.Module:
    """当模型被 DataParallel 包装时返回底层 module。"""

    return model.module if hasattr(model, "module") else model
