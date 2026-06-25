"""模型实现。"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_training.training.config import ModelConfig


class Expert(nn.Module):
    """两层前馈专家网络。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        """创建专家 MLP。"""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将专家网络应用到 token 表示上。"""

        return self.net(x)


class Router(nn.Module):
    """根据池化后的编码器特征预测领域和专家 gate。"""

    def __init__(self, input_dim: int, num_domains: int, num_experts: int, use_topk: bool, top_k: int, hidden_dim: int):
        """创建领域预测头和专家路由头。"""

        super().__init__()
        self.num_domains = num_domains
        self.num_experts = num_experts
        self.use_topk = use_topk
        self.top_k = top_k
        self.domain_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains),
        )
        self.expert_router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self,
        x: torch.Tensor,
        domain_labels: torch.Tensor | None = None,
        training: bool = True,
        use_gt_domains_for_routing: bool = False,
    ) -> dict:
        """返回一个 batch 的领域 logits 和专家 gate。"""

        pooled = x[:, 0, :] if x.dim() == 3 else x
        domain_logits = self.domain_predictor(pooled)
        domain_probs = F.softmax(domain_logits, dim=-1)
        predicted_domains = torch.argmax(domain_probs, dim=-1)
        routing_domains = predicted_domains
        if training and domain_labels is not None and use_gt_domains_for_routing:
            routing_domains = domain_labels
        expert_logits = self.expert_router(pooled)
        expert_gates = self._build_expert_gates(expert_logits)
        return {
            "domain_logits": domain_logits,
            "domain_probs": domain_probs,
            "predicted_domains": predicted_domains,
            "routing_domains": routing_domains,
            "expert_gates": expert_gates,
            "expert_logits": expert_logits,
        }

    def _build_expert_gates(self, expert_logits: torch.Tensor) -> torch.Tensor:
        """将专家 logits 转换为 dense gate 或 top-k 稀疏 gate。"""

        if self.use_topk and self.top_k < self.num_experts:
            topk_gates, topk_indices = torch.topk(expert_logits, k=self.top_k, dim=-1)
            topk_gates = F.softmax(topk_gates, dim=-1)
            gates = torch.zeros_like(expert_logits)
            return gates.scatter(dim=-1, index=topk_indices, src=topk_gates)
        return F.softmax(expert_logits, dim=-1)


class MoENER(nn.Module):
    """BERT 风格编码器，加上 MoE 专家和领域专属 NER 头。"""

    def __init__(self, model_config: ModelConfig):
        """创建编码器、router、专家和领域分类器。"""

        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.model_config = model_config
        self.encoder_config = AutoConfig.from_pretrained(model_config.pretrained_model)
        self.bert = AutoModel.from_pretrained(model_config.pretrained_model)
        self.encoder_accepts_bbox = _accepts_forward_arg(self.bert, "bbox")
        self.dropout = nn.Dropout(model_config.dropout_prob)
        self.num_domains = model_config.num_domains or len(model_config.domain_names)
        self.num_experts = model_config.num_experts
        self.router = Router(
            input_dim=self.encoder_config.hidden_size,
            num_domains=self.num_domains,
            num_experts=self.num_experts,
            use_topk=model_config.use_topk_routing,
            top_k=model_config.top_k,
            hidden_dim=model_config.router_hidden_dim,
        )
        self.experts = nn.ModuleList(
            [
                Expert(
                    input_dim=self.encoder_config.hidden_size,
                    hidden_dim=model_config.expert_hidden_size,
                    output_dim=self.encoder_config.hidden_size,
                    dropout=model_config.dropout_prob,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.domain_classifiers = self._build_domain_classifiers(model_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
        use_gt_domains_for_classification: bool = False,
        use_gt_domains_for_routing: bool = False,
    ) -> dict:
        """运行编码器、router、专家和领域专属分类器。"""

        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if self.encoder_accepts_bbox:
            encoder_inputs["bbox"] = bbox
        encoder_outputs = self.bert(**encoder_inputs)
        sequence_output = self.dropout(encoder_outputs.last_hidden_state)
        router_outputs = self.router(
            sequence_output,
            domain_labels=domain_ids,
            training=self.training,
            use_gt_domains_for_routing=use_gt_domains_for_routing,
        )
        combined_output = self._combine_experts(sequence_output, router_outputs["expert_gates"])
        classification_domains = self._classification_domains(
            domain_ids,
            router_outputs["routing_domains"],
            router_outputs["predicted_domains"],
            use_gt_domains_for_classification,
        )
        logits = self._classify_by_domain(combined_output, classification_domains)
        return {
            "logits": logits,
            "domain_logits": router_outputs["domain_logits"],
            "domain_probs": router_outputs["domain_probs"],
            "predicted_domains": router_outputs["predicted_domains"],
            "classification_domains": classification_domains,
            "routing_domains": router_outputs["routing_domains"],
            "expert_gates": router_outputs["expert_gates"],
            "hidden_states": combined_output,
        }

    def _build_domain_classifiers(self, model_config: ModelConfig) -> nn.ModuleList:
        """为每个配置的领域创建一个 token 分类器。"""

        classifiers = nn.ModuleList()
        for domain in model_config.domain_names:
            tag_size = model_config.domain_tag_sizes_dict.get(domain, 2)
            classifiers.append(nn.Linear(self.encoder_config.hidden_size, tag_size))
        return classifiers

    def _combine_experts(self, sequence_output: torch.Tensor, expert_gates: torch.Tensor) -> torch.Tensor:
        """计算每个样本的加权专家输出。"""

        if self.model_config.use_topk_routing and self.model_config.top_k < self.num_experts:
            return self._combine_topk_experts(sequence_output, expert_gates)
        expert_outputs = []
        for expert_id, expert in enumerate(self.experts):
            weight = expert_gates[:, expert_id].view(-1, 1, 1)
            expert_outputs.append(expert(sequence_output) * weight)
        return torch.stack(expert_outputs, dim=0).sum(dim=0)

    def _combine_topk_experts(self, sequence_output: torch.Tensor, expert_gates: torch.Tensor) -> torch.Tensor:
        """只计算每条样本实际激活的 top-k 专家。"""

        batch_size, seq_len, hidden_size = sequence_output.shape
        top_k = min(self.model_config.top_k, self.num_experts)
        topk_weights, topk_indices = torch.topk(expert_gates, k=top_k, dim=-1)
        flat_expert_ids = topk_indices.reshape(-1)
        expanded_input = (
            sequence_output.unsqueeze(1)
            .expand(-1, top_k, -1, -1)
            .reshape(-1, seq_len, hidden_size)
        )
        expanded_weights = topk_weights.reshape(-1, 1, 1)
        expert_outputs = torch.zeros_like(expanded_input)
        for expert_id, expert in enumerate(self.experts):
            expert_mask = flat_expert_ids == expert_id
            if expert_mask.any():
                expert_outputs[expert_mask] = expert(expanded_input[expert_mask])
        weighted_outputs = expert_outputs * expanded_weights
        combined = torch.zeros_like(sequence_output)
        batch_indices = torch.arange(batch_size, device=sequence_output.device).repeat_interleave(top_k)
        index = batch_indices.view(-1, 1, 1).expand(-1, seq_len, hidden_size)
        combined.scatter_add_(dim=0, index=index, src=weighted_outputs)
        return combined

    def _classification_domains(
        self,
        domain_ids: torch.Tensor | None,
        routing_domains: torch.Tensor,
        predicted_domains: torch.Tensor,
        use_gt_domains_for_classification: bool,
    ) -> torch.Tensor:
        """选择每个样本应该使用哪个领域分类器打分。"""

        if domain_ids is not None and use_gt_domains_for_classification:
            return domain_ids
        if domain_ids is not None:
            return routing_domains
        return predicted_domains

    def _classify_by_domain(self, hidden_states: torch.Tensor, classification_domains: torch.Tensor) -> torch.Tensor:
        """将每个样本送入选中的领域分类器，并对 logits 做 padding。"""

        batch_size, seq_len, _ = hidden_states.shape
        max_classes = max(classifier.out_features for classifier in self.domain_classifiers)
        logits = torch.zeros(batch_size, seq_len, max_classes, device=hidden_states.device, dtype=hidden_states.dtype)
        for domain_id, classifier in enumerate(self.domain_classifiers):
            mask = classification_domains == domain_id
            if mask.any():
                domain_logits = classifier(hidden_states[mask])
                logits[mask] = F.pad(domain_logits, (0, max_classes - domain_logits.size(-1)))
        return logits


def create_model(model_config: ModelConfig):
    """根据 ModelConfig 中的名称创建模型。"""

    if model_config.name == "moe_ner":
        return MoENER(model_config)
    raise ValueError(f"不支持的模型: {model_config.name}")


def _accepts_forward_arg(module: nn.Module, arg_name: str) -> bool:
    """返回模块 forward 是否声明或透传指定参数。"""

    signature = inspect.signature(module.forward)
    return arg_name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
