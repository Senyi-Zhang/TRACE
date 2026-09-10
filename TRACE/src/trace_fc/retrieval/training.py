"""Hard-negative mining and cross-encoder reranker training."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from trace_fc.data import ClaimExample, Table, read_jsonl
from trace_fc.decomposition.amr import indexed_nodes, node_query
from trace_fc.retrieval.dense import DenseTableIndex
from trace_fc.retrieval.marked_table import SPECIAL_TOKENS, serialize_marked_table

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankerPair:
    qid: str
    query: str
    table_uid: str
    label: int
    negative_source: str = ""

    def to_dict(self) -> dict:
        return {
            "qid": self.qid, "query": self.query, "table_uid": self.table_uid,
            "label": self.label, "negative_source": self.negative_source,
        }


def load_trees(path: str | Path) -> dict[str, dict]:
    return {str(item["id"]): item["tree"] for item in read_jsonl(path)}


def example_queries(
    example: ClaimExample, tree: dict | None, source: str,
) -> list[tuple[str, str]]:
    if source == "claim":
        return [(example.id, example.claim)]
    if tree is None:
        raise ValueError(f"No tree available for example {example.id}")
    result = []
    seen = set()
    for path, node in indexed_nodes(tree["root"]):
        query = node_query(node)
        if query and query not in seen:
            seen.add(query)
            result.append((f"{example.id}:{path}", query))
    return result


def mine_pairs(
    examples: list[ClaimExample],
    tables: dict[str, Table],
    dense_index: DenseTableIndex,
    trees: dict[str, dict] | None,
    query_source: str,
    hard_negatives: int,
    random_negatives: int,
    coarse_top_k: int,
    seed: int,
) -> list[RerankerPair]:
    rng = random.Random(seed)
    all_uids = list(tables)
    pairs: list[RerankerPair] = []
    for example in tqdm(examples, desc="Mining reranker pairs"):
        missing = [uid for uid in example.table_uids if uid not in tables]
        if missing:
            raise KeyError(f"Gold table(s) missing for example {example.id}: {missing}")
        positives = set(example.table_uids)
        if not positives:
            raise ValueError(f"Example {example.id} has no gold table UID")
        for qid, query in example_queries(
            example, None if trees is None else trees.get(example.id), query_source,
        ):
            candidates = [uid for uid, _ in dense_index.search(query, coarse_top_k)]
            hard = [uid for uid in candidates if uid not in positives][:hard_negatives]
            pool = [uid for uid in all_uids if uid not in positives and uid not in hard]
            random_items = rng.sample(pool, min(random_negatives, len(pool)))
            pairs.extend(RerankerPair(qid, query, uid, 1) for uid in positives)
            pairs.extend(RerankerPair(qid, query, uid, 0, "hard") for uid in hard)
            pairs.extend(RerankerPair(qid, query, uid, 0, "random") for uid in random_items)
    return pairs


def split_by_qid(
    pairs: list[RerankerPair], dev_ratio: float, seed: int,
) -> tuple[list[RerankerPair], list[RerankerPair]]:
    qids = sorted({pair.qid for pair in pairs})
    random.Random(seed).shuffle(qids)
    dev_count = max(1, round(len(qids) * dev_ratio)) if len(qids) > 1 else 0
    dev_qids = set(qids[:dev_count])
    return (
        [pair for pair in pairs if pair.qid not in dev_qids],
        [pair for pair in pairs if pair.qid in dev_qids],
    )


class PairDataset(Dataset):
    def __init__(
        self, pairs: list[RerankerPair], tables: dict[str, Table], random_weight: float,
        max_rows: int,
    ):
        self.pairs = pairs
        self.tables = tables
        self.random_weight = random_weight
        self.max_rows = max_rows

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        return {
            "qid": pair.qid,
            "text": serialize_marked_table(
                pair.query, self.tables[pair.table_uid], self.max_rows,
            ),
            "label": float(pair.label),
            "weight": self.random_weight if pair.negative_source == "random" else 1.0,
        }


class PairCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items: list[dict]) -> dict:
        encoded = self.tokenizer(
            [item["text"] for item in items], padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "inputs": encoded,
            "labels": torch.tensor([item["label"] for item in items], dtype=torch.float32),
            "weights": torch.tensor([item["weight"] for item in items], dtype=torch.float32),
            "qids": [item["qid"] for item in items],
        }


@torch.inference_mode()
def evaluate_reranker(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, list[float]] = defaultdict(list)
    for batch in loader:
        inputs = {key: value.to(device) for key, value in batch["inputs"].items()}
        scores = model(**inputs).logits.squeeze(-1).detach().cpu().reshape(-1).tolist()
        for qid, score, label in zip(batch["qids"], scores, batch["labels"].tolist()):
            grouped_scores[qid].append(score)
            grouped_labels[qid].append(label)
    values = [
        ndcg_score(np.asarray([grouped_labels[qid]]), np.asarray([scores]), k=10)
        for qid, scores in grouped_scores.items()
        if any(grouped_labels[qid])
    ]
    return float(np.mean(values)) if values else 0.0


def train_reranker(
    train_pairs: list[RerankerPair],
    dev_pairs: list[RerankerPair],
    tables: dict[str, Table],
    model_name: str,
    output_dir: str | Path,
    max_length: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    warmup_ratio: float,
    random_negative_weight: float,
    max_rows: int,
    device: str | None = None,
) -> dict[str, float]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device_obj)
    collator = PairCollator(tokenizer, max_length)
    train_loader = DataLoader(
        PairDataset(train_pairs, tables, random_negative_weight, max_rows),
        batch_size=batch_size, shuffle=True, collate_fn=collator,
    )
    dev_loader = DataLoader(
        PairDataset(dev_pairs, tables, random_negative_weight, max_rows),
        batch_size=batch_size, shuffle=False, collate_fn=collator,
    ) if dev_pairs else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * warmup_ratio), total_steps,
    )
    use_amp = device_obj.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_score = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Reranker epoch {epoch + 1}/{epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            inputs = {key: value.to(device_obj) for key, value in batch["inputs"].items()}
            labels = batch["labels"].to(device_obj)
            weights = batch["weights"].to(device_obj)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**inputs).logits.squeeze(-1)
                raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none",
                )
                loss = (raw_loss * weights).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        score = evaluate_reranker(model, dev_loader, device_obj) if dev_loader else 0.0
        LOGGER.info("epoch=%d dev_ndcg@10=%.4f", epoch + 1, score)
        if best_state is None or score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {"best_dev_ndcg@10": best_score}
