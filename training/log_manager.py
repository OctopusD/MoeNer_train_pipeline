"""训练日志管理器。"""

from __future__ import annotations

from model_training.utils.logging import create_logger
from model_training.utils.serialization import to_plain_data


class LogManager:
    """按训练流程直接记录配置、进度、指标和完成状态。"""

    def __init__(self, timestamp: str, output_dir: str = "training_outputs", batch_log_interval: int = 50):
        """创建文件和终端双输出的 logger。"""

        self.logger = create_logger("model_training", output_dir, timestamp)
        self.batch_log_interval = batch_log_interval

    def log_config(self, config) -> None:
        """记录训练配置。"""

        self.logger.info("训练配置参数:")
        self.logger.info("=" * 50)
        for key, value in to_plain_data(config).items():
            self.logger.info("%s: %s", key, value)
        self.logger.info("=" * 50)

    def log_training_start(self, device, n_gpu: int, total_steps: int) -> None:
        """记录训练开始信息。"""

        self.logger.info("训练设备: %s, GPU数量: %s", device, n_gpu)
        self.logger.info("预计训练步数: %s", total_steps)
        self.logger.info("开始训练")

    def log_epoch_start(self, epoch: int, total_epochs: int, stage_idx: int) -> None:
        """记录 epoch 开始信息。"""

        self.logger.info("=" * 60)
        self.logger.info("Epoch %s/%s stage=%s", epoch + 1, total_epochs, stage_idx)
        self.logger.info("=" * 60)

    def log_batch_progress(self, epoch: int, batch_idx: int, loss: float, lr: float) -> None:
        """按固定间隔记录 batch 训练进度。"""

        if batch_idx % self.batch_log_interval == 0:
            self.logger.info("Epoch %s, Batch %s, Loss: %.6f, LR: %.3e", epoch + 1, batch_idx, loss, lr)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: dict,
        best_val_f1: float,
        current_lr: float,
    ) -> None:
        """记录 epoch 汇总指标。"""

        self.logger.info("训练损失: %.6f", train_loss)
        self.logger.info("验证损失: %.6f", val_loss)
        entity_metrics = metrics.get("entity_level")
        if entity_metrics:
            self._log_entity_metrics(entity_metrics)
            overall = entity_metrics.get("overall") or {}
            if overall.get("f1", 0.0) >= best_val_f1:
                self.logger.info("新的最佳Entity F1分数: %.6f", overall.get("f1", 0.0))
        domain_metrics = metrics.get("domain_level")
        if domain_metrics:
            self._log_domain_metrics(domain_metrics)
        expert_frequency = metrics.get("expert_frequency")
        if expert_frequency:
            self._log_expert_frequency(expert_frequency)
        self.logger.info("当前学习率: %.3e", current_lr)

    def log_training_complete(self, best_val_loss: float, best_val_f1: float) -> None:
        """记录训练完成信息。"""

        self.logger.info("训练完成")
        self.logger.info("最佳验证损失: %.6f", best_val_loss)
        self.logger.info("最佳验证F1: %.6f", best_val_f1)

    def _log_entity_metrics(self, metrics: dict) -> None:
        """记录实体级指标。"""

        self.logger.info("Entity级别指标:")
        overall = metrics.get("overall") or {}
        if overall:
            self.logger.info(
                "总体: precision=%.6f recall=%.6f f1=%.6f support=%s",
                overall.get("precision", 0.0),
                overall.get("recall", 0.0),
                overall.get("f1", 0.0),
                overall.get("support", 0),
            )
        for domain_name, domain_metrics in metrics.get("by_domain", {}).items():
            self.logger.info(
                "%s: precision=%.6f recall=%.6f f1=%.6f support=%s",
                domain_name,
                domain_metrics.get("precision", 0.0),
                domain_metrics.get("recall", 0.0),
                domain_metrics.get("f1", 0.0),
                domain_metrics.get("support", 0),
            )

    def _log_domain_metrics(self, metrics: dict) -> None:
        """记录领域分类指标。"""

        self.logger.info(
            "Domain级别指标: accuracy=%.6f (%s/%s)",
            metrics.get("accuracy", 0.0),
            metrics.get("correct", 0),
            metrics.get("total", 0),
        )
        for domain_name, domain_metrics in metrics.get("by_domain", {}).items():
            self.logger.info(
                "%s: precision=%.6f recall=%.6f f1=%.6f support=%s",
                domain_name,
                domain_metrics.get("precision", 0.0),
                domain_metrics.get("recall", 0.0),
                domain_metrics.get("f1", 0.0),
                domain_metrics.get("support", 0),
            )

    def _log_expert_frequency(self, metrics: dict) -> None:
        """记录专家使用频率。"""

        self.logger.info("专家使用频率:")
        self.logger.info("overall: %s", metrics.get("overall", {}))
        for domain_name, usage in metrics.get("by_domain", {}).items():
            self.logger.info("%s: %s", domain_name, usage)
