"""优化器、scheduler 和训练阶段辅助逻辑。"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

import torch
from torch import nn

from model_training.training.config import OptimizerConfig, SchedulerConfig, StageConfig


class StageResolver:
    """查找指定 epoch 对应的当前训练阶段。"""

    def __init__(self, stages: List[StageConfig], total_epochs: int):
        """规范化配置的阶段；未配置时创建默认阶段。"""

        self.total_epochs = total_epochs
        self.stages = self._normalize(stages)

    def _normalize(self, stages: List[StageConfig]) -> List[StageConfig]:
        """将阶段范围限制在配置的训练长度内。"""

        if not stages:
            return [StageConfig(start_epoch=0, end_epoch=self.total_epochs, index=0)]
        normalized: List[StageConfig] = []
        for idx, stage in enumerate(stages):
            end_epoch = self.total_epochs if stage.end_epoch is None else min(stage.end_epoch, self.total_epochs)
            normalized.append(replace(stage, start_epoch=max(0, stage.start_epoch), end_epoch=end_epoch, index=idx))
        return normalized

    def stage_for_epoch(self, epoch: int) -> StageConfig:
        """返回 [start_epoch, end_epoch) 区间包含指定 epoch 的阶段。"""

        for stage in self.stages:
            if stage.start_epoch <= epoch < (stage.end_epoch or self.total_epochs):
                return stage
        return self.stages[-1]

def build_param_groups(
    model: nn.Module,
    stage_config: StageConfig,
    default_lr: float,
    default_weight_decay: float = 0.01,
) -> List[Dict]:
    """通过关键词匹配参数名称来构建优化器参数组。"""

    all_params = dict(model.named_parameters())
    used_ids = set()
    param_groups: List[Dict] = []

    for group_config in stage_config.param_groups:
        matched = []
        for name, param in all_params.items():
            is_match = not group_config.params or any(keyword in name for keyword in group_config.params)
            is_excluded = any(keyword in name for keyword in group_config.exclude)
            if is_match and not is_excluded and param.requires_grad and id(param) not in used_ids:
                matched.append(param)
                used_ids.add(id(param))
        if matched:
            param_groups.append(_build_group_dict(matched, group_config, default_lr, default_weight_decay))

    unmatched = [param for param in all_params.values() if param.requires_grad and id(param) not in used_ids]
    if unmatched:
        param_groups.append({"params": unmatched, "lr": default_lr, "weight_decay": default_weight_decay})
    return param_groups


def _build_group_dict(params: list, group_config, default_lr: float, default_weight_decay: float) -> Dict:
    """创建一个继承默认值的优化器参数组字典。"""

    group = {
        "params": params,
        "lr": default_lr if group_config.lr is None else group_config.lr,
        "weight_decay": default_weight_decay if group_config.weight_decay is None else group_config.weight_decay,
    }
    if group_config.betas is not None:
        group["betas"] = tuple(group_config.betas)
    if group_config.eps is not None:
        group["eps"] = group_config.eps
    return group


def create_optimizer(param_groups, optimizer_config: OptimizerConfig) -> torch.optim.Optimizer:
    """根据配置好的参数组创建优化器。"""

    if optimizer_config.name.lower() == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            betas=tuple(optimizer_config.betas),
            eps=optimizer_config.eps,
        )
    raise ValueError(f"不支持的优化器: {optimizer_config.name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: SchedulerConfig,
    steps_per_epoch: int,
    remaining_epochs: int,
) -> Optional[object]:
    """创建配置指定的 scheduler；禁用时返回 None。"""

    name = scheduler_config.name.lower()
    if name in {"", "none", "constant"}:
        return None
    total_steps = scheduler_config.total_steps or max(1, steps_per_epoch * remaining_epochs)
    if name == "one_cycle":
        max_lr = [group.get("lr", 0.0) for group in optimizer.param_groups]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=scheduler_config.warmup_ratio,
            anneal_strategy=scheduler_config.anneal_strategy,
            cycle_momentum=False,
        )
    raise ValueError(f"不支持的 scheduler: {scheduler_config.name}")
