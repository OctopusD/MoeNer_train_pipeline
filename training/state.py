"""可序列化的训练状态。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class TrainingState:
    """保存可写入 checkpoint 的可变训练进度。"""

    epoch: int = 0
    global_step: int = 0
    stage_idx: int = 0
    best_metrics: Dict[str, float] = field(default_factory=dict)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """将状态序列化为字典，便于保存到 checkpoint。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any] | None) -> "TrainingState":
        """从 checkpoint 元数据构建 TrainingState，并兼容缺失字段。"""

        if not raw:
            return cls()
        return cls(
            epoch=raw.get("epoch", 0),
            global_step=raw.get("global_step", 0),
            stage_idx=raw.get("stage_idx", 0),
            best_metrics=dict(raw.get("best_metrics", {})),
            train_losses=list(raw.get("train_losses", [])),
            val_losses=list(raw.get("val_losses", [])),
        )

