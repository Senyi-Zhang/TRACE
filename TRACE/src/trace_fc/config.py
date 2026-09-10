"""Typed configuration with paths resolved relative to the YAML file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class DataConfig:
    dataset: str = "OTT-FV"
    train_file: str = "OTT-FV/train.jsonl"
    test_file: str = "OTT-FV/test.jsonl"
    table_file: str = "OTT-FV/tables.json"
    tree_dir: str = "artifacts/trees"
    evidence_dir: str = "artifacts/evidence"


@dataclass
class AMRConfig:
    model: str = "AMR3-structbart-L"


@dataclass
class RetrievalConfig:
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "microsoft/deberta-v3-base"
    index_dir: str = "artifacts/index"
    reranker_dir: str = "artifacts/reranker"
    coarse_top_k: int = 200
    table_top_k: int = 10
    row_top_k: int = 10
    max_table_rows: int = 50
    max_table_length: int = 512
    reranker_batch_size: int = 16
    row_ranker: str = "lexical"


@dataclass
class RetrieverTrainingConfig:
    hard_negatives: int = 8
    random_negatives: int = 2
    query_source: str = "nodes"
    dev_ratio: float = 0.1
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_ratio: float = 0.05
    random_negative_weight: float = 0.5


@dataclass
class VerifierConfig:
    backbone: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    batch_size: int = 16
    node_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 0
    dev_ratio: float = 0.1
    output_dir: str = "artifacts/verifier"


@dataclass
class TraceConfig:
    seed: int = 42
    setting: str = "open"
    data: DataConfig = field(default_factory=DataConfig)
    amr: AMRConfig = field(default_factory=AMRConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    retriever_training: RetrieverTrainingConfig = field(default_factory=RetrieverTrainingConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)

    def validate(self) -> None:
        if self.setting not in {"gold", "open"}:
            raise ValueError("setting must be 'gold' or 'open'")
        if self.retrieval.row_ranker not in {"lexical", "dense"}:
            raise ValueError("retrieval.row_ranker must be 'lexical' or 'dense'")
        if self.retriever_training.query_source not in {"claim", "nodes"}:
            raise ValueError("retriever_training.query_source must be 'claim' or 'nodes'")
        if self.retrieval.table_top_k > self.retrieval.coarse_top_k:
            raise ValueError("table_top_k cannot exceed coarse_top_k")
        if self.retrieval.row_top_k > 64:
            raise ValueError("row_top_k cannot exceed the 64 registered [RowN] markers")
        if not 0.0 < self.verifier.dev_ratio < 1.0:
            raise ValueError("verifier.dev_ratio must be in (0, 1)")
        positive_values = {
            "retrieval.coarse_top_k": self.retrieval.coarse_top_k,
            "retrieval.table_top_k": self.retrieval.table_top_k,
            "retrieval.row_top_k": self.retrieval.row_top_k,
            "retriever_training.batch_size": self.retriever_training.batch_size,
            "verifier.batch_size": self.verifier.batch_size,
            "verifier.node_batch_size": self.verifier.node_batch_size,
            "verifier.gradient_accumulation_steps": (
                self.verifier.gradient_accumulation_steps
            ),
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"These configuration values must be positive: {invalid}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    known = {item.name: item.type for item in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        value = values.get(item.name)
        default = getattr(cls(), item.name)
        nested_type = type(default)
        if value is not None and is_dataclass(default):
            kwargs[item.name] = _construct(nested_type, value)
        elif value is not None:
            kwargs[item.name] = value
    return cls(**kwargs)


def _resolve_paths(config: TraceConfig, base: Path) -> None:
    path_fields = [
        (config.data, "train_file"), (config.data, "test_file"),
        (config.data, "table_file"), (config.data, "tree_dir"),
        (config.data, "evidence_dir"), (config.retrieval, "index_dir"),
        (config.retrieval, "reranker_dir"), (config.verifier, "output_dir"),
    ]
    for obj, name in path_fields:
        path = Path(getattr(obj, name)).expanduser()
        if not path.is_absolute():
            path = (base / path).resolve()
        setattr(obj, name, str(path))


def load_config(path: str | Path) -> TraceConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    config = _construct(TraceConfig, raw)
    _resolve_paths(config, path.parent)
    config.validate()
    return config
