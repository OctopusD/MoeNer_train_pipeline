"""可复用训练 pipeline 的配置 dataclass。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    """保存每次运行都应记录的实验级元数据。"""

    name: str = "experiment"
    output_dir: str = "training_outputs"
    seed: int = 42
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


@dataclass
class DataConfig:
    """定义原始数据如何加载并转换为 batch。"""

    name: str = "rasa_bo"
    data_dir: str = ""
    domain_names: List[str] = field(default_factory=list)
    tokenizer_name_or_path: str = ""
    max_seq_length: int = 512
    train_batch_size: int = 128
    eval_batch_size: int = 4
    num_workers: int = 1


@dataclass
class ModelConfig:
    """定义模型结构以及从 datamodule 派生的元数据。"""

    name: str = "moe_ner"
    pretrained_model: str = ""
    dropout_prob: float = 0.1
    num_experts: int = 1
    use_topk_routing: bool = True
    top_k: int = 1
    router_hidden_dim: int = 256
    expert_hidden_size: int = 512
    domain_tag_sizes_dict: Dict[str, int] = field(default_factory=dict)
    num_domains: int = 0
    domain_names: List[str] = field(default_factory=list)


@dataclass
class LossConfig:
    """定义 loss 函数选择及其权重。"""

    ner_loss: str = "weighted_ce"
    domain_loss: str = "ce"
    load_balance_loss: str = "kl_uniform"
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"ner": 1.0, "domain": 1.0, "load_balance": 0.1}
    )
    entity_weight: float = 1.0
    non_entity_weight: float = 0.1
    focal_gamma: float = 2.0
    focal_alpha: Optional[Any] = None


@dataclass
class OptimizerConfig:
    """定义优化器类型和默认超参数。"""

    name: str = "adamw"
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """定义 scheduler 类型和调度参数。"""

    name: str = "one_cycle"
    warmup_ratio: float = 0.1
    total_steps: Optional[int] = None
    anneal_strategy: str = "cos"


@dataclass
class ParamGroupConfig:
    """描述一个通过关键词匹配的优化器参数组。"""

    params: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    betas: Optional[tuple] = None
    eps: Optional[float] = None


@dataclass
class StageConfig:
    """定义一个 epoch 区间及该区间内的优化器参数组。"""

    start_epoch: int = 0
    end_epoch: Optional[int] = None
    param_groups: List[ParamGroupConfig] = field(default_factory=list)
    index: int = 0


@dataclass
class CheckpointConfig:
    """定义 checkpoint 加载、保存和指标监控行为。"""

    init_checkpoint: str = ""
    resume_from: Optional[str] = None
    save_dir: str = "training_outputs"
    save_interval: int = 1
    save_optimizer: bool = True
    resume_optimizer: bool = True
    monitor: str = "entity_level.overall.f1"
    mode: str = "max"


@dataclass
class TrainConfig:
    """定义与模型结构无关的训练循环设置。"""

    num_epochs: int = 1
    device: Any = "auto"
    gradient_clip_norm: float = 1.0
    early_stopping_patience: Optional[int] = None
    validation_interval: int = 1
    use_gt_domains_for_classification_train: bool = False
    use_gt_domains_for_classification_eval: bool = False
    use_gt_domains_for_routing_epochs: int = 0
    use_gt_domains_for_routing_eval: bool = False
    compute_expert_frequency: bool = False


@dataclass
class PipelineConfig:
    """训练流水线使用的顶层配置对象。"""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    stages: List[StageConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PipelineConfig":
        """从部分指定的字典构建嵌套的 PipelineConfig。"""

        raw = raw or {}
        config = cls(
            experiment=ExperimentConfig(**raw.get("experiment", {})),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            loss=LossConfig(**raw.get("loss", {})),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            scheduler=SchedulerConfig(**raw.get("scheduler", {})),
            checkpoint=CheckpointConfig(**raw.get("checkpoint", {})),
            train=TrainConfig(**raw.get("train", {})),
            stages=_build_stages(raw.get("stages", [])),
        )
        config.model.domain_names = list(config.data.domain_names)
        config.model.num_domains = len(config.data.domain_names)
        if not config.checkpoint.save_dir:
            config.checkpoint.save_dir = config.experiment.output_dir
        return config

    def to_dict(self) -> Dict[str, Any]:
        """将该 dataclass 树序列化为普通 Python 容器。"""

        return asdict(self)


def _build_stages(raw_stages: List[Dict[str, Any]]) -> List[StageConfig]:
    """将原始 stage 字典转换为带索引的 StageConfig 对象。"""

    stages: List[StageConfig] = []
    for idx, raw_stage in enumerate(raw_stages or []):
        groups = [ParamGroupConfig(**group) for group in raw_stage.get("param_groups", [])]
        stage = StageConfig(
            start_epoch=raw_stage.get("start_epoch", 0),
            end_epoch=raw_stage.get("end_epoch"),
            param_groups=groups,
            index=idx,
        )
        stages.append(stage)
    return stages


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """加载 YAML 配置文件并返回 PipelineConfig 实例。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return PipelineConfig.from_dict(raw)

