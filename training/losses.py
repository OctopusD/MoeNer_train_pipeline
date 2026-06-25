"""损失函数实现。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """根据分类 logits 计算 Focal Loss。"""

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean", ignore_index: int = -100):
        """保存 Focal Loss 超参数。"""

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """根据 logits 和整数类别标签计算 Focal Loss。"""

        ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            focal_weight = focal_weight * self._alpha_for_targets(targets, inputs.device)
        loss = focal_weight * ce_loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

    def _alpha_for_targets(self, targets: torch.Tensor, device: torch.device) -> torch.Tensor:
        """返回每个目标标签的 alpha 权重，并忽略被 mask 的标签。"""

        valid_targets = targets.clamp_min(0)
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(device)
            return alpha[valid_targets] * (targets != self.ignore_index).float()
        return torch.full_like(targets, float(self.alpha), dtype=torch.float, device=device)


def weighted_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    entity_weight: float = 1.0,
    non_entity_weight: float = 0.1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """计算 token 交叉熵，并将类别 0 视为非实体标签。"""

    num_classes = logits.size(-1)
    weights = torch.full((num_classes,), float(entity_weight), device=logits.device, dtype=logits.dtype)
    weights[0] = float(non_entity_weight)
    return F.cross_entropy(logits.view(-1, num_classes), labels.view(-1), weight=weights, ignore_index=ignore_index)


class CompositeLoss(nn.Module):
    """组合 NER、领域分类和负载均衡 loss。"""

    def __init__(
        self,
        ner_loss: str = "weighted_ce",
        domain_loss: str = "ce",
        load_balance_loss: str = "kl_uniform",
        loss_weights: Optional[Dict[str, float]] = None,
        entity_weight: float = 1.0,
        non_entity_weight: float = 0.1,
        focal_gamma: float = 2.0,
        focal_alpha=None,
    ):
        """保存 loss 选择并构建可复用的 criterion 对象。"""

        super().__init__()
        self.ner_loss = ner_loss
        self.domain_loss = domain_loss
        self.load_balance_loss = load_balance_loss
        self.loss_weights = loss_weights or {"ner": 1.0, "domain": 1.0, "load_balance": 0.1}
        self.entity_weight = entity_weight
        self.non_entity_weight = non_entity_weight
        self.domain_focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma) if domain_loss == "focal" else None

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算配置的各项 loss，并返回总 loss 和单项 loss。"""

        loss_items: Dict[str, torch.Tensor] = {}
        if self.ner_loss:
            loss_items["ner"] = self._compute_ner_loss(outputs["logits"], batch["labels"])
        if self.domain_loss and "domain_logits" in outputs and "domain_ids" in batch:
            loss_items["domain"] = self._compute_domain_loss(outputs["domain_logits"], batch["domain_ids"])
        if self.load_balance_loss and "expert_gates" in outputs:
            loss_items["load_balance"] = self._compute_load_balance_loss(outputs["expert_gates"])

        total = None
        for name, value in loss_items.items():
            weighted = self.loss_weights.get(name, 1.0) * value
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("CompositeLoss 没有计算出任何 loss 项。")
        return {"loss": total, "loss_items": loss_items}

    def _compute_ner_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算配置指定的 token 分类 loss。"""

        if self.ner_loss == "weighted_ce":
            return weighted_token_cross_entropy(logits, labels, self.entity_weight, self.non_entity_weight)
        if self.ner_loss == "ce":
            return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        raise ValueError(f"不支持的 ner_loss: {self.ner_loss}")

    def _compute_domain_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算配置指定的领域分类 loss。"""

        if self.domain_loss == "focal":
            return self.domain_focal(logits, labels)
        if self.domain_loss == "ce":
            return F.cross_entropy(logits, labels)
        raise ValueError(f"不支持的 domain_loss: {self.domain_loss}")

    def _compute_load_balance_loss(self, expert_gates: torch.Tensor) -> torch.Tensor:
        """约束专家 gate 使用分布尽量接近均匀分布。"""

        if expert_gates.size(-1) <= 1:
            return expert_gates.sum() * 0.0
        importance = expert_gates.sum(dim=0)
        importance = importance / (importance.sum() + 1e-8)
        uniform = torch.ones_like(importance) / importance.numel()
        return F.kl_div(torch.log(importance + 1e-8), uniform, reduction="batchmean")


def create_loss(loss_config) -> CompositeLoss:
    """根据配置创建 CompositeLoss。"""

    return CompositeLoss(
        ner_loss=loss_config.ner_loss,
        domain_loss=loss_config.domain_loss,
        load_balance_loss=loss_config.load_balance_loss,
        loss_weights=loss_config.loss_weights,
        entity_weight=loss_config.entity_weight,
        non_entity_weight=loss_config.non_entity_weight,
        focal_gamma=loss_config.focal_gamma,
        focal_alpha=loss_config.focal_alpha,
    )
