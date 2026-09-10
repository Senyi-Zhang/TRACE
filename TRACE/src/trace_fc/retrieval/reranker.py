"""Cross-encoder table reranker and row-ranking strategies."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch

from trace_fc.data import Table
from trace_fc.retrieval.marked_table import serialize_marked_table
from trace_fc.serialization import EvidenceRow


class CrossEncoderReranker:
    def __init__(self, tokenizer, model, device: str | torch.device, max_length: int = 512):
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.max_length = max_length

    @classmethod
    def from_pretrained(
        cls, path: str | Path, device: str | None = None, max_length: int = 512,
    ) -> "CrossEncoderReranker":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)
        return cls(tokenizer, model, device, max_length)

    @torch.inference_mode()
    def score(
        self, query: str, table_ids: list[str], tables: dict[str, Table],
        batch_size: int = 16, max_rows: int = 50,
    ) -> list[tuple[str, float]]:
        return self.score_many(
            [(query, table_ids)], tables, batch_size=batch_size, max_rows=max_rows,
        )[0]

    @torch.inference_mode()
    def score_many(
        self,
        queries_and_tables: list[tuple[str, list[str]]],
        tables: dict[str, Table],
        batch_size: int = 16,
        max_rows: int = 50,
    ) -> list[list[tuple[str, float]]]:
        texts = []
        owners = []
        for owner, (query, table_ids) in enumerate(queries_and_tables):
            texts.extend(serialize_marked_table(query, tables[uid], max_rows) for uid in table_ids)
            owners.extend((owner, uid) for uid in table_ids)
        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start:start + batch_size], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**batch).logits.squeeze(-1)
            scores.extend(float(value) for value in logits.detach().cpu().reshape(-1))
        grouped: list[list[tuple[str, float]]] = [[] for _ in queries_and_tables]
        for (owner, uid), score in zip(owners, scores):
            grouped[owner].append((uid, score))
        return [
            sorted(items, key=lambda item: item[1], reverse=True) for items in grouped
        ]


def all_rows(table_ids: Iterable[str], tables: dict[str, Table]) -> list[EvidenceRow]:
    rows = []
    for uid in table_ids:
        table = tables[uid]
        rows.extend(
            EvidenceRow(uid, table.title, table.header, row) for row in table.data
        )
    return rows


class LexicalRowRanker:
    """The rarity-weighted row reranker used by the legacy inference code."""

    def rank(self, query: str, rows: list[EvidenceRow], top_k: int) -> list[EvidenceRow]:
        from wordfreq import zipf_frequency
        words = re.sub(r"[^\w\s]", " ", query.lower()).split()
        weights = {word: 8.0 - zipf_frequency(word, "en") for word in words}
        scored = []
        for row in rows:
            text = " ".join([row.table_title, *row.header, *row.cells]).lower()
            score = sum(weight for word, weight in weights.items() if word in text)
            scored.append(EvidenceRow(
                row.table_uid, row.table_title, row.header, row.cells, float(score),
            ))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


class DenseRowRanker:
    def __init__(self, model):
        self.model = model

    def rank(self, query: str, rows: list[EvidenceRow], top_k: int) -> list[EvidenceRow]:
        if not rows:
            return []
        texts = [
            f"[TITLE] {row.table_title} [ROW] "
            + " | ".join(f"{h}: {c}" for h, c in zip(row.header, row.cells))
            for row in rows
        ]
        query_vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
        )[0]
        row_vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        scores = row_vectors @ query_vector
        ranked = sorted(range(len(rows)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        return [EvidenceRow(
            rows[i].table_uid, rows[i].table_title, rows[i].header, rows[i].cells,
            float(scores[i]),
        ) for i in ranked]
