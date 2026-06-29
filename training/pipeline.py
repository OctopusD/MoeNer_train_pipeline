"""顶层 pipeline 装配逻辑。"""

from __future__ import annotations

from copy import deepcopy

import torch

from model_training.data_preparation.rasa_bo.datamodule import RasaBODataModule
from model_training.evaluation.moe_ner import DomainEvaluator, ExpertUsageEvaluator, MetricCollection, NEREvaluator
from model_training.training.checkpoint import CheckpointManager
from model_training.training.config import PipelineConfig
from model_training.training.device import resolve_device
from model_training.training.log_manager import LogManager
from model_training.training.losses import create_loss
from model_training.training.models import create_model
from model_training.training.state import TrainingState
from model_training.training.trainer import Trainer
from model_training.utils.seed import set_seed


class TrainingPipeline:
    """根据配置装配数据、模型、loss、评估器和 trainer。"""

    def __init__(self, config: PipelineConfig):
        """为当前 pipeline 实例保存一份防御性配置拷贝。"""

        self.config = deepcopy(config)

    def run(self) -> TrainingState:
        """构建全部组件并执行训练。"""

        device_context = resolve_device(self.config.train.device)
        print(f"已解析设备: {device_context.device}, GPU数量: {device_context.n_gpu}")

        set_seed(self.config.experiment.seed)
        print(f"已设置随机种子: {self.config.experiment.seed}")

        if self.config.data.name != "rasa_bo":
            raise ValueError(f"不支持的数据模块: {self.config.data.name}")
        datamodule = RasaBODataModule(self.config.data)
        print("已创建数据模块")
        datamodule.setup()
        print("已准备好训练和验证数据")

        self._inject_data_metadata(datamodule.metadata)
        print("已注入数据元信息")

        model = create_model(self.config.model)
        print("已创建模型")
        if device_context.n_gpu > 1:
            model = torch.nn.DataParallel(model)
            print("已启用 DataParallel")

        loss_computer = create_loss(self.config.loss)
        print("已创建 loss")

        log_manager = LogManager(self.config.experiment.timestamp, self.config.experiment.output_dir)
        print("已创建日志管理器")

        domain_names = self.config.model.domain_names
        evaluator = MetricCollection(
            {
                "entity_level": NEREvaluator(domain_names),
                "domain_level": DomainEvaluator(domain_names),
                "expert_frequency": ExpertUsageEvaluator(
                    domain_names,
                    self.config.model.num_experts,
                    enabled=self.config.train.compute_expert_frequency,
                ),
            }
        )
        print("已创建评估器")

        checkpoint_manager = CheckpointManager(self.config.checkpoint.save_dir)
        print("已创建 checkpoint 管理器")

        state = self._load_initial_state(model, checkpoint_manager)
        print("已加载初始训练状态")

        trainer = Trainer(
            config=self.config,
            model=model,
            datamodule=datamodule,
            loss_computer=loss_computer,
            evaluator=evaluator,
            device=device_context.device,
            n_gpu=device_context.n_gpu,
            log_manager=log_manager,
            checkpoint_manager=checkpoint_manager,
            state=state,
        )
        print("已创建 trainer")

        self._load_resume_optimizer_state(trainer, checkpoint_manager)
        print("开始训练")
        return trainer.train()

    def _inject_data_metadata(self, metadata: dict) -> None:
        """将 datamodule 派生的元数据复制到模型配置中。
        MoENER 创建领域分类头时，需要知道每个 domain 有多少个 NER 标签：
        Airplane -> 多少个 entity-B + O
        Train    -> 多少个 entity-B + O
        Hotel    -> 多少个 entity-B + O

        这些信息不是模型自己知道的，是 datamodule 读完 RASA 数据、构建 schema 后才算出来的。
        """

        self.config.model.domain_names = list(metadata.get("domain_names", self.config.data.domain_names))
        self.config.model.num_domains = len(self.config.model.domain_names)
        self.config.model.domain_tag_sizes_dict = dict(metadata.get("domain_tag_sizes_dict", {}))

    def _load_initial_state(self, model, checkpoint_manager: CheckpointManager) -> TrainingState:
        """根据配置加载仅模型参数或完整恢复 checkpoint。"""

        if self.config.checkpoint.resume_from:
            state = checkpoint_manager.load_model(model, self.config.checkpoint.resume_from)
            state.epoch += 1
            return state
        if self.config.checkpoint.init_checkpoint:
            checkpoint_manager.load_model(model, self.config.checkpoint.init_checkpoint)
        return TrainingState()

    def _load_resume_optimizer_state(self, trainer: Trainer, checkpoint_manager: CheckpointManager) -> None:
        """在 trainer 创建优化器之后恢复 checkpoint 中的优化器状态。"""

        if not self.config.checkpoint.resume_from or not self.config.checkpoint.resume_optimizer:
            return
        state = checkpoint_manager.load_full(
            model=trainer.model,
            checkpoint_path=self.config.checkpoint.resume_from,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
        )
        state.epoch += 1
        trainer.state = state
