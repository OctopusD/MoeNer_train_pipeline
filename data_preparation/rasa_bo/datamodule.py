"""RASA NLU BO 标注数据的数据模块。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model_training.training.config import DataConfig


@dataclass
class RasaDocument:
    """单条文本样本及其标注的内部表示。"""

    doc_id: str
    text: str
    intent: str
    entities: list[dict]
    input_ids: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    offsets: list[list[int]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    char_bboxes: list[list[int]] | None = None
    token_bboxes: list[list[int]] = field(default_factory=list)


class RasaBODataset(Dataset):
    """返回用于 token 分类训练的标准化 batch item。"""

    def __init__(self, documents: list[RasaDocument], domain_to_id: dict[str, int]):
        """保存已 tokenizer 化的文档和领域 ID 映射。"""

        self.documents = documents
        self.domain_to_id = domain_to_id

    def __getitem__(self, index: int) -> dict:
        """返回一条使用标准 Tensor 字段名的样本。"""

        doc = self.documents[index]
        return {
            "input_ids": torch.tensor(doc.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(doc.attention_mask, dtype=torch.bool),
            "labels": torch.tensor(doc.labels, dtype=torch.long),
            "domain_ids": torch.tensor(self.domain_to_id.get(doc.intent, 0), dtype=torch.long),
            "offsets": torch.tensor(doc.offsets, dtype=torch.int32),
            "bbox": torch.tensor(doc.token_bboxes, dtype=torch.int32),
            "text": doc.text,
            "doc_id": doc.doc_id,
        }

    def __len__(self) -> int:
        """返回文档数量。"""

        return len(self.documents)


class RasaBOProcessor:
    """读取 RASA NLU JSON 文件，并产出训练/验证文档。"""

    def __init__(self, data_dir: str, domain_names: list[str]):
        """保存数据位置和选中的领域。"""

        self.data_dir = Path(data_dir)
        self.domain_names = domain_names
        self.schema: dict[str, set[str]] = {}

    def load(self) -> tuple[list[RasaDocument], list[RasaDocument]]:
        """从配置的领域目录加载训练和验证文档。"""

        self._validate_inputs()
        train_items = self._read_split("train")
        validation_items = self._read_split("validation")
        if not train_items:
            raise ValueError(f"没有读取到训练样本，请检查 data_dir 和 domain_names: {self.data_dir}")
        if not validation_items:
            raise ValueError(f"没有读取到验证样本，请检查 data_dir 和 domain_names: {self.data_dir}")
        self.schema = _build_full_schema(train_items + validation_items)
        if not any(self.schema.values()):
            raise ValueError("RASA 数据中没有发现任何实体标签，无法训练 NER 模型。")
        return self._to_documents(train_items), self._to_documents(validation_items)

    def _validate_inputs(self) -> None:
        """检查数据目录、领域列表和 split 文件是否完整。"""

        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir 不存在: {self.data_dir}")
        if not self.domain_names:
            raise ValueError("data.domain_names 不能为空。")
        missing = []
        for domain in self.domain_names:
            domain_dir = self.data_dir / domain
            if not domain_dir.is_dir():
                missing.append(str(domain_dir))
                continue
            for split in ("train", "validation"):
                if self._resolve_split_path(domain, split) is None:
                    missing.append(str(domain_dir / f"{split}.json or {split}.jsonl"))
        if missing:
            raise FileNotFoundError("RASA 数据文件不完整: " + ", ".join(missing))

    def _read_split(self, split: str) -> list[dict]:
        """从每个配置的领域目录读取一个 split 文件。"""

        items = []
        for domain in self.domain_names:
            path = self._resolve_split_path(domain, split)
            if path is not None:
                items.extend(self._read_rasa_file(path))
        return items

    def _read_rasa_file(self, path: Path) -> list[dict]:
        """读取一个 RASA NLU JSON/JSONL 文件并转换为标准化样本字典。"""

        with path.open("r", encoding="utf-8") as handle:
            content = handle.read().strip()
        if not content:
            return []
        try:
            raw = json.loads(content)
            if isinstance(raw, dict) and "rasa_nlu_data" in raw:
                examples = raw.get("rasa_nlu_data", {}).get("common_examples", [])
            elif isinstance(raw, list):
                examples = raw
            else:
                examples = [raw]
        except json.JSONDecodeError:
            examples = [json.loads(line) for line in content.splitlines() if line.strip()]
        normalized = []
        for example in examples:
            entities = [
                entity
                for entity in example.get("entities", [])
                if {"entity", "value", "start", "end"}.issubset(entity)
            ]
            item = {"text": example["text"], "intent": example["intent"], "entities": entities}
            if "char_bboxes" in example:
                item["char_bboxes"] = example["char_bboxes"]
            normalized.append(item)
        return normalized

    def _resolve_split_path(self, domain: str, split: str) -> Path | None:
        """返回 split 数据文件，优先使用新数据导出的 jsonl。"""

        domain_dir = self.data_dir / domain
        for suffix in (".jsonl", ".json"):
            path = domain_dir / f"{split}{suffix}"
            if path.is_file():
                return path
        return None

    def _to_documents(self, items: list[dict]) -> list[RasaDocument]:
        """将标准化字典转换为 RasaDocument 实例。"""

        return [
            RasaDocument(
                doc_id=str(index),
                text=item["text"],
                intent=item["intent"],
                entities=item["entities"],
                char_bboxes=item.get("char_bboxes"),
            )
            for index, item in enumerate(items)
        ]


class RasaBODataModule:
    """为 BO token 分类准备 RASA NLU 数据。"""

    def __init__(self, config: DataConfig):
        """保存配置，并为配置的模型路径创建 tokenizer。"""

        from transformers import AutoTokenizer

        self.config = config
        tokenizer_name = config.tokenizer_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        self.metadata = {}
        self.train_dataset = None
        self.validation_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """读取原始数据、tokenize 文档、标注 token，并创建数据集。"""

        processor = RasaBOProcessor(self.config.data_dir, self.config.domain_names)
        train_docs, validation_docs = processor.load()
        self._tokenize_and_label(train_docs, processor.schema)
        self._tokenize_and_label(validation_docs, processor.schema)
        domain_to_id = {domain: index for index, domain in enumerate(self.config.domain_names)}
        self.train_dataset = RasaBODataset(train_docs, domain_to_id)
        self.validation_dataset = RasaBODataset(validation_docs, domain_to_id)
        self.metadata = {
            "domain_names": self.config.domain_names,
            "domain_to_id": domain_to_id,
            "schema": processor.schema,
            "domain_tag_sizes_dict": {
                domain: calculate_tag_size(processor.schema.get(domain, set())) for domain in self.config.domain_names
            },
        }

    def train_dataloader(self) -> DataLoader:
        """返回训练 DataLoader。"""

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """返回验证 DataLoader。"""

        return DataLoader(
            self.validation_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def _tokenize_and_label(self, documents: list[RasaDocument], schema: dict[str, set[str]]) -> None:
        """就地为文档添加 token id、mask、offset 和 BO 标签。"""

        for doc in documents:
            encoded = self.tokenizer.encode_plus(
                doc.text,
                add_special_tokens=True,
                padding="max_length",
                max_length=self.config.max_seq_length,
                truncation=True,
                return_offsets_mapping=True,
            )
            doc.input_ids = encoded["input_ids"]
            doc.attention_mask = encoded["attention_mask"]
            doc.offsets = [[int(start), int(end)] if mask else [-1, -1] for (start, end), mask in zip(encoded["offset_mapping"], doc.attention_mask)]
            self._set_token_bboxes(doc)
            tags = self._build_bo_tags(doc)
            mapping = _create_event_specific_tag_mapping(doc.intent, schema)
            doc.labels = [
                mapping.get(tag, 0) if _is_trainable_token(mask, offset) else -100
                for tag, mask, offset in zip(tags, doc.attention_mask, doc.offsets)
            ]

    def _set_token_bboxes(self, doc: RasaDocument) -> None:
        """将字符级 bbox 对齐到 token 起始字符，缺失时使用零 bbox。"""

        if not doc.char_bboxes or len(doc.char_bboxes) != len(doc.text):
            doc.char_bboxes = [[0, 0, 0, 0] for _ in doc.text]
        doc.token_bboxes = []
        for offset, mask in zip(doc.offsets, doc.attention_mask):
            start, _ = offset
            if not mask or offset == [0, 0] or start < 0 or start >= len(doc.char_bboxes):
                doc.token_bboxes.append([0, 0, 0, 0])
            else:
                doc.token_bboxes.append(doc.char_bboxes[start])

    def _build_bo_tags(self, doc: RasaDocument) -> list[str]:
        """通过实体字符 span 匹配 token offset 来构建 BO 标签。"""

        tags = ["O" for _ in doc.input_ids]
        for entity in doc.entities:
            tag = f"{entity['entity']}-B"
            for index, (start, end) in enumerate(doc.offsets):
                if start >= 0 and end >= 0 and not (start == 0 and end == 0):
                    if entity["start"] <= start and end <= entity["end"]:
                        tags[index] = tag
        return tags


def _build_full_schema(items: list[dict]) -> dict[str, set[str]]:
    """根据样本中观察到的实体类型，构建 intent/domain 到实体类型的映射。"""

    schema: dict[str, set[str]] = {}
    for item in items:
        intent = item["intent"]
        schema.setdefault(intent, set())
        schema[intent].update(entity["entity"] for entity in item.get("entities", []))
    return schema


def calculate_tag_size(entity_types: set[str]) -> int:
    """返回实体类型集合对应的 BO 标签数量，包含 O。"""

    return len([entity for entity in entity_types if entity != "Null"]) + 1


def _create_event_specific_tag_mapping(event_name: str, schema: dict[str, set[str]]) -> dict[str, int]:
    """为一个 event/domain 创建包含 O 和 entity-B 的标签映射。"""

    entity_types = sorted(entity for entity in schema.get(event_name, set()) if entity != "Null")
    tags = ["O"] + [f"{entity}-B" for entity in entity_types]
    return {tag: index for index, tag in enumerate(tags)}


def _is_trainable_token(mask: int | bool, offset: list[int]) -> bool:
    """只让真实文本 token 参与 token classification loss。"""

    return bool(mask) and offset[0] >= 0 and offset[1] >= 0 and offset != [0, 0]
