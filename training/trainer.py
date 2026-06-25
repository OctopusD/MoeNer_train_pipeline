"""通用 PyTorch trainer。"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from model_training.utils.serialization import to_plain_data
from model_training.training.state import TrainingState
from model_training.training.optim import StageResolver, build_param_groups, create_optimizer, create_scheduler

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """缺少 tqdm 时直接返回原始迭代器。"""

        return iterable


class Trainer:
    """使用注入的组件运行模型训练和验证。"""

    def __init__(
        self,
        config,
        model: nn.Module,
        datamodule,
        loss_computer: nn.Module,
        evaluator,
        device: torch.device,
        n_gpu: int = 0,
        log_manager=None,
        checkpoint_manager=None,
        state: Optional[TrainingState] = None,
    ):
        """保存训练依赖并初始化优化器状态。"""

        self.config = config
        self.model = model.to(device)
        self.datamodule = datamodule
        self.loss_computer = loss_computer
        self.evaluator = evaluator
        self.device = device
        self.n_gpu = n_gpu
        self.log_manager = log_manager
        self.checkpoint_manager = checkpoint_manager
        self.state = state or TrainingState()
        self.stage_resolver = StageResolver(config.stages, config.train.num_epochs)
        self.should_stop = False
        self.early_stop_best_value = None
        self.early_stop_bad_epochs = 0
        self.checkpoint_best_value = None
        self.last_train_loss = 0.0
        self.last_val_loss = 0.0
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.optimizer = None
        self.scheduler = None
        self.rebuild_optimizer_and_scheduler()

    def rebuild_optimizer_and_scheduler(self) -> None:
        """为当前训练阶段重新创建优化器和 scheduler。"""

        stage = self.stage_resolver.stage_for_epoch(self.state.epoch)
        param_groups = build_param_groups(
            self.model,
            stage,
            default_lr=self.config.optimizer.learning_rate,
            default_weight_decay=self.config.optimizer.weight_decay,
        )
        self.optimizer = create_optimizer(param_groups, self.config.optimizer)
        remaining_epochs = max(1, self.config.train.num_epochs - self.state.epoch)
        self.scheduler = create_scheduler(
            self.optimizer,
            self.config.scheduler,
            steps_per_epoch=len(self.datamodule.train_dataloader()),
            remaining_epochs=remaining_epochs,
        )
        print("已创建好优化器和调度器")

    def train(self) -> TrainingState:
        """运行完整训练循环并返回最终 TrainingState。"""

        if self.log_manager is not None:
            total_steps = len(self.datamodule.train_dataloader()) * max(0, self.config.train.num_epochs - self.state.epoch)
            self.log_manager.log_config(self.config)
            self.log_manager.log_training_start(self.device, self.n_gpu, total_steps)
        epoch_range = tqdm(
            range(self.state.epoch, self.config.train.num_epochs),
            desc="Training epochs",
            initial=self.state.epoch,
            total=self.config.train.num_epochs,
        )
        for epoch in epoch_range:
            self.state.epoch = epoch
            self._switch_stage_if_needed()
            if self.log_manager is not None:
                self.log_manager.log_epoch_start(epoch, self.config.train.num_epochs, self.state.stage_idx)
            self.last_train_loss = self.train_one_epoch()
            metrics = {}
            if (epoch + 1) % self.config.train.validation_interval == 0:
                self.last_val_loss, metrics = self.validate_one_epoch()
            if self.log_manager is not None:
                current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
                self.log_manager.log_epoch_metrics(
                    epoch,
                    self.last_train_loss,
                    self.last_val_loss,
                    metrics,
                    self.best_val_f1,
                    current_lr,
                )
            self._update_best_metrics(metrics)
            self._save_checkpoints(metrics)
            self._update_early_stopping(metrics)
            if self.should_stop:
                break
        if self.log_manager is not None:
            self.log_manager.log_training_complete(self.best_val_loss, self.best_val_f1)
        return self.state

    def train_one_epoch(self) -> float:
        """运行一个训练 epoch 并返回平均 loss。"""

        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for batch_idx, batch in enumerate(tqdm(self.datamodule.train_dataloader(), desc=f"Epoch {self.state.epoch}")):
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()
            outputs = self.model(**self._model_inputs(batch, training=True))
            loss_result = self.loss_computer(outputs, batch)
            loss = loss_result["loss"]
            if self.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            if self.config.train.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            self.state.global_step += 1
            if self.log_manager is not None:
                current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
                self.log_manager.log_batch_progress(self.state.epoch, batch_idx, float(loss.detach().cpu()), current_lr)
        avg_loss = total_loss / max(1, total_batches)
        self.state.train_losses.append(avg_loss)
        return avg_loss

    def validate_one_epoch(self) -> tuple[float, dict]:
        """运行验证并返回平均 loss 和计算出的指标。"""

        self.model.eval()
        self.evaluator.reset()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.datamodule.val_dataloader(), desc="Validation"):
                batch = self._move_batch(batch)
                outputs = self.model(**self._model_inputs(batch, training=False))
                loss_result = self.loss_computer(outputs, batch)
                loss = loss_result["loss"]
                if self.n_gpu > 1:
                    loss = loss.mean()
                total_loss += float(loss.detach().cpu())
                total_batches += 1
                self.evaluator.update(outputs, batch)
        avg_loss = total_loss / max(1, total_batches)
        self.state.val_losses.append(avg_loss)
        metrics = self.evaluator.compute()
        return avg_loss, metrics

    def _switch_stage_if_needed(self) -> None:
        """在 epoch 开始时按 stage 配置切换优化器和 scheduler。"""

        stage = self.stage_resolver.stage_for_epoch(self.state.epoch)
        if stage.index != self.state.stage_idx:
            self.state.stage_idx = stage.index
            self.rebuild_optimizer_and_scheduler()
            print(f"已切换到训练阶段: {stage.index}")

    def _save_checkpoints(self, metrics: dict) -> None:
        """按配置保存周期性 checkpoint 和最佳 checkpoint。"""

        if self.checkpoint_manager is None:
            return
        checkpoint_config = self.config.checkpoint
        if checkpoint_config.save_interval and (self.state.epoch + 1) % checkpoint_config.save_interval == 0:
            self._save_checkpoint(metrics, f"epoch_{self.state.epoch}.pth")
        monitor_value = self._get_nested_metric(metrics, checkpoint_config.monitor)
        if monitor_value is not None and self._is_better(monitor_value, self.checkpoint_best_value, checkpoint_config.mode):
            self.checkpoint_best_value = monitor_value
            self._save_checkpoint(metrics, "best.pth")

    def _save_checkpoint(self, metrics: dict, name: str) -> str:
        """保存一个 checkpoint。"""

        checkpoint_config = self.config.checkpoint
        optimizer = self.optimizer if checkpoint_config.save_optimizer else None
        scheduler = self.scheduler if checkpoint_config.save_optimizer else None
        path = self.checkpoint_manager.save(
            model=self.model,
            state=self.state,
            config=to_plain_data(self.config),
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            name=name,
        )
        print(f"已保存 checkpoint: {path}")
        return path

    def _update_early_stopping(self, metrics: dict) -> None:
        """根据监控指标更新早停状态。"""

        patience = self.config.train.early_stopping_patience
        if patience is None:
            return
        monitor_value = self._get_nested_metric(metrics, self.config.checkpoint.monitor)
        if monitor_value is None:
            return
        if self._is_better(monitor_value, self.early_stop_best_value, self.config.checkpoint.mode):
            self.early_stop_best_value = monitor_value
            self.early_stop_bad_epochs = 0
        else:
            self.early_stop_bad_epochs += 1
        if self.early_stop_bad_epochs >= patience:
            self.should_stop = True
            print("已触发 early stopping")

    def _update_best_metrics(self, metrics: dict) -> None:
        """更新 trainer 持有的最佳验证指标。"""

        if not metrics:
            return
        if self.last_val_loss < self.best_val_loss:
            self.best_val_loss = self.last_val_loss
        entity_overall = metrics.get("entity_level", {}).get("overall", {})
        entity_f1 = entity_overall.get("f1")
        if entity_f1 is not None and entity_f1 > self.best_val_f1:
            self.best_val_f1 = entity_f1

    def _model_inputs(self, batch: dict, training: bool) -> dict:
        """从 batch 字典中选择标准模型输入字段。"""

        use_gt_classification = (
            self.config.train.use_gt_domains_for_classification_train
            if training
            else self.config.train.use_gt_domains_for_classification_eval
        )
        use_gt_routing = (
            self.state.epoch < self.config.train.use_gt_domains_for_routing_epochs
            if training
            else self.config.train.use_gt_domains_for_routing_eval
        )
        return {
            "input_ids": batch.get("input_ids"),
            "bbox": batch.get("bbox"),
            "attention_mask": batch.get("attention_mask"),
            "domain_ids": batch.get("domain_ids"),
            "use_gt_domains_for_classification": use_gt_classification,
            "use_gt_domains_for_routing": use_gt_routing,
        }

    def _move_batch(self, batch: dict) -> dict:
        """将 batch 中的 Tensor 移动到 trainer 设备，其他元数据保持不变。"""

        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    def _get_nested_metric(self, metrics: dict, dotted_path: str):
        """使用点分路径读取嵌套指标。"""

        current = metrics
        for part in dotted_path.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _is_better(self, value: float, best_value: float | None, mode: str) -> bool:
        """根据 mode 判断 value 是否优于 best_value。"""

        if best_value is None:
            return True
        if mode == "min":
            return value < best_value
        return value > best_value
