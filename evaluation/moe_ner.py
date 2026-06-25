"""MoE NER 评估器。"""

from __future__ import annotations

from collections import defaultdict

import torch


class MetricCollection:
    """在统一的 reset/update/compute 接口后运行多个评估器。"""

    def __init__(self, evaluators: dict[str, object]):
        """按输出指标分区名称保存评估器。"""

        self.evaluators = evaluators

    def reset(self) -> None:
        """重置所有评估器。"""

        for evaluator in self.evaluators.values():
            evaluator.reset()

    def update(self, outputs: dict, batch: dict) -> None:
        """用一个 batch 更新所有评估器。"""

        for evaluator in self.evaluators.values():
            evaluator.update(outputs, batch)

    def compute(self) -> dict:
        """计算并返回所有非空指标分区。"""

        result = {}
        for name, evaluator in self.evaluators.items():
            metrics = evaluator.compute()
            if metrics:
                result[name] = metrics
        return result


class NEREvaluator:
    """通过实体 span 精确匹配计算实体级 precision、recall 和 F1。"""

    def __init__(self, domain_names: list[str]):
        """保存领域名称并初始化指标缓存。"""

        self.domain_names = domain_names
        self.reset()

    def reset(self) -> None:
        """清空实体预测和标签缓存。"""

        self.predictions = {domain: [] for domain in self.domain_names}
        self.labels = {domain: [] for domain in self.domain_names}

    def update(self, outputs: dict, batch: dict) -> None:
        """从一个验证 batch 中提取预测实体和真实实体。"""

        logits = outputs["logits"]
        predictions = logits.argmax(dim=-1).detach().cpu().tolist()
        labels = batch["labels"].detach().cpu().tolist()
        masks = batch["attention_mask"].detach().cpu().tolist()
        domains = batch["domain_ids"].detach().cpu().tolist()
        classification_domains = outputs.get("classification_domains", batch["domain_ids"]).detach().cpu().tolist()
        for pred_seq, label_seq, mask_seq, true_domain, pred_domain in zip(
            predictions, labels, masks, domains, classification_domains
        ):
            valid_pred = [value for value, mask in zip(pred_seq, mask_seq) if mask]
            valid_label = [value for value, mask in zip(label_seq, mask_seq) if mask]
            domain_name = self._name(true_domain)
            pred_domain_name = self._name(pred_domain)
            self.predictions[domain_name].append(_extract_entities(valid_pred, pred_domain_name))
            self.labels[domain_name].append(_extract_entities(valid_label, domain_name))

    def compute(self) -> dict:
        """计算分领域和整体实体指标。"""

        by_domain = {}
        total_tp = total_fp = total_fn = 0
        for domain in self.domain_names:
            tp = fp = fn = 0
            support = 0
            for pred_entities, gold_entities in zip(self.predictions[domain], self.labels[domain]):
                pred_set = set(pred_entities)
                gold_set = set(gold_entities)
                tp += len(pred_set & gold_set)
                fp += len(pred_set - gold_set)
                fn += len(gold_set - pred_set)
                support += len(gold_set)
            by_domain[domain] = _metrics(tp, fp, fn, support)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        overall = _metrics(total_tp, total_fp, total_fn, total_tp + total_fn)
        return {"by_domain": by_domain, "overall": overall}

    def _name(self, domain_id: int) -> str:
        """将领域 ID 转换为配置中的领域名称。"""

        return self.domain_names[domain_id] if 0 <= domain_id < len(self.domain_names) else f"domain_{domain_id}"


class DomainEvaluator:
    """计算领域分类准确率和简单的分领域统计。"""

    def __init__(self, domain_names: list[str]):
        """保存领域名称并初始化计数器。"""

        self.domain_names = domain_names
        self.reset()

    def reset(self) -> None:
        """清空领域计数器。"""

        self.total = 0
        self.correct = 0
        self.confusion = defaultdict(lambda: defaultdict(int))

    def update(self, outputs: dict, batch: dict) -> None:
        """根据预测领域 ID 和真实领域 ID 更新计数器。"""

        predicted = outputs.get("classification_domains", outputs.get("predicted_domains"))
        if predicted is None or "domain_ids" not in batch:
            return
        pred_ids = predicted.detach().cpu().tolist()
        true_ids = batch["domain_ids"].detach().cpu().tolist()
        for true_id, pred_id in zip(true_ids, pred_ids):
            true_name = self._name(true_id)
            pred_name = self._name(pred_id)
            self.total += 1
            self.correct += int(true_id == pred_id)
            self.confusion[true_name][pred_name] += 1

    def compute(self) -> dict:
        """返回准确率、计数、分领域指标和混淆矩阵。"""

        by_domain = {}
        for domain in self.domain_names:
            true_total = sum(self.confusion[domain].values())
            correct = self.confusion[domain][domain]
            predicted_total = sum(self.confusion[other][domain] for other in self.domain_names)
            precision = correct / predicted_total if predicted_total else 0.0
            recall = correct / true_total if true_total else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            by_domain[domain] = {"precision": precision, "recall": recall, "f1": f1, "support": true_total}
        return {
            "accuracy": self.correct / self.total if self.total else 0.0,
            "correct": self.correct,
            "total": self.total,
            "by_domain": by_domain,
            "confusion_matrix": {key: dict(value) for key, value in self.confusion.items()},
        }

    def _name(self, domain_id: int) -> str:
        """将领域 ID 转换为配置中的领域名称。"""

        return self.domain_names[domain_id] if 0 <= domain_id < len(self.domain_names) else f"domain_{domain_id}"


class ExpertUsageEvaluator:
    """可选地统计哪些专家获得了非零路由权重。"""

    def __init__(self, domain_names: list[str], num_experts: int, enabled: bool = False):
        """保存专家维度以及是否启用统计。"""

        self.domain_names = domain_names
        self.num_experts = num_experts
        self.enabled = enabled
        self.reset()

    def reset(self) -> None:
        """清空专家使用计数。"""

        self.domain_counts = torch.zeros(len(self.domain_names), self.num_experts, dtype=torch.long)
        self.overall_counts = torch.zeros(self.num_experts, dtype=torch.long)

    def update(self, outputs: dict, batch: dict) -> None:
        """统计每个样本中 gate 大于零的专家。"""

        if not self.enabled or "expert_gates" not in outputs:
            return
        gates = outputs["expert_gates"].detach().cpu()
        domains = batch["domain_ids"].detach().cpu()
        active = gates > 0
        for row, domain_id in zip(active, domains):
            for expert_id in row.nonzero(as_tuple=False).flatten().tolist():
                self.domain_counts[int(domain_id), expert_id] += 1
                self.overall_counts[expert_id] += 1

    def compute(self) -> dict:
        """按领域和整体格式化专家使用计数。"""

        if not self.enabled:
            return {}
        return {
            "by_domain": {
                domain: {f"expert_{idx}": int(self.domain_counts[didx, idx]) for idx in range(self.num_experts)}
                for didx, domain in enumerate(self.domain_names)
            },
            "overall": {f"expert_{idx}": int(self.overall_counts[idx]) for idx in range(self.num_experts)},
        }


def _extract_entities(sequence: list[int], domain_name: str) -> list[tuple]:
    """将连续非零标签片段提取为用于精确匹配的实体元组。"""

    entities = []
    current = None
    for index, label in enumerate(sequence):
        if label != 0 and label != -100:
            if current is None or current[2] != label:
                if current is not None:
                    entities.append(tuple(current))
                current = [index, index, label, domain_name]
            else:
                current[1] = index
        elif current is not None:
            entities.append(tuple(current))
            current = None
    if current is not None:
        entities.append(tuple(current))
    return entities


def _metrics(tp: int, fp: int, fn: int, support: int) -> dict:
    """根据实体计数计算 precision、recall 和 F1。"""

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}
