"""Evidence-aware node encoder and post-order TRACE verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from trace_fc.decomposition.amr import indexed_nodes, postorder_nodes
from trace_fc.models.composition import FiLM, PMA
from trace_fc.serialization import (
    NODE_SPECIAL_TOKENS,
    EvidenceRow,
    serialize_node_evidence,
)


class TraceVerifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        max_length: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        node_batch_size: int = 32,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.max_length = max_length
        self.node_batch_size = node_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        self.encoder = AutoModel.from_pretrained(backbone)
        added_tokens = self.tokenizer.add_special_tokens({
            "additional_special_tokens": NODE_SPECIAL_TOKENS,
        })
        if added_tokens:
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token_id is None:
            fallback = self.tokenizer.eos_token or self.tokenizer.unk_token
            if fallback is None:
                raise ValueError("The backbone tokenizer has no usable padding token")
            self.tokenizer.pad_token = fallback
            self.encoder.config.pad_token_id = self.tokenizer.pad_token_id
        hidden_size = self.encoder.config.hidden_size
        self.child_pma = PMA(hidden_size, num_heads, dropout)
        self.film = FiLM(hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)

    def encode_node_texts(self, texts: list[str]) -> torch.Tensor:
        batches = []
        for start in range(0, len(texts), self.node_batch_size):
            encoded = self.tokenizer(
                texts[start:start + self.node_batch_size], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            output = self.encoder(**encoded).last_hidden_state
            batches.append(self._masked_mean(output, encoded.attention_mask))
        return torch.cat(batches, dim=0)

    @staticmethod
    def _node_text(path: str, evidence: dict) -> str:
        item = evidence["nodes"].get(path)
        if item is None:
            raise KeyError(f"Missing evidence for tree node {path}")
        rows = [EvidenceRow.from_dict(value) for value in item.get("rows", [])]
        return serialize_node_evidence(item["query"], rows)

    def forward(self, items: list[dict[str, Any]]) -> torch.Tensor:
        flat_texts: list[str] = []
        offsets: list[dict[str, int]] = []
        for item in items:
            paths = [path for path, _ in indexed_nodes(item["tree"]["root"])]
            offsets.append({path: len(flat_texts) + index for index, path in enumerate(paths)})
            flat_texts.extend(self._node_text(path, item["evidence"]) for path in paths)
        self_embeddings = self.encode_node_texts(flat_texts)
        roots = []
        for item, offset in zip(items, offsets):
            composed: dict[str, torch.Tensor] = {}
            for path, _node, child_paths in postorder_nodes(item["tree"]["root"]):
                own = self_embeddings[offset[path]]
                if child_paths:
                    children = torch.stack([composed[child] for child in child_paths])
                    own = self.film(own, self.child_pma(children))
                composed[path] = own
            roots.append(composed["n0"])
        return self.classifier(self.dropout(torch.stack(roots)))

    def save_checkpoint(self, directory: str | Path, metadata: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        backbone_dir = directory / "backbone"
        self.encoder.save_pretrained(backbone_dir)
        self.tokenizer.save_pretrained(backbone_dir)
        head_state = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items() if not key.startswith("encoder.")
        }
        torch.save(head_state, directory / "trace_head.pt")
        configuration = {
            "backbone": "backbone", "max_length": self.max_length,
            "num_heads": self.child_pma.attention.num_heads,
            "dropout": self.dropout.p, "node_batch_size": self.node_batch_size,
            "metadata": metadata or {},
        }
        with (directory / "trace_config.json").open("w", encoding="utf-8") as stream:
            json.dump(configuration, stream, ensure_ascii=False, indent=2)

    @classmethod
    def from_checkpoint(cls, directory: str | Path, map_location: str = "cpu") -> "TraceVerifier":
        directory = Path(directory)
        with (directory / "trace_config.json").open("r", encoding="utf-8") as stream:
            configuration = json.load(stream)
        instance = cls(
            backbone=str(directory / configuration["backbone"]),
            max_length=configuration["max_length"], num_heads=configuration["num_heads"],
            dropout=configuration["dropout"], node_batch_size=configuration["node_batch_size"],
        )
        state = torch.load(
            directory / "trace_head.pt", map_location=map_location, weights_only=True,
        )
        missing, unexpected = instance.load_state_dict(state, strict=False)
        invalid_missing = [key for key in missing if not key.startswith("encoder.")]
        if unexpected or invalid_missing:
            raise RuntimeError(
                f"Invalid TRACE head state: missing={invalid_missing}, unexpected={unexpected}"
            )
        return instance
