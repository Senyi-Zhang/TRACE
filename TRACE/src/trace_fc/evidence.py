"""Create and load per-node evidence caches for Gold and Open settings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from trace_fc.data import ClaimExample, Table, read_jsonl
from trace_fc.decomposition.amr import indexed_nodes, node_query
from trace_fc.retrieval.reranker import all_rows


def load_tree_store(path: str | Path) -> dict[str, dict]:
    return {str(item["id"]): item["tree"] for item in read_jsonl(path)}


def load_evidence_store(path: str | Path) -> dict[str, dict]:
    return {str(item["id"]): item for item in read_jsonl(path)}


def build_evidence_records(
    examples: Iterable[ClaimExample],
    trees: dict[str, dict],
    tables: dict[str, Table],
    setting: str,
    row_ranker,
    row_top_k: int,
    retriever=None,
) -> Iterable[dict]:
    if setting == "open" and retriever is None:
        raise ValueError("Open evidence construction requires a retriever")
    for example in tqdm(examples, desc=f"Preparing {setting} evidence"):
        tree = trees.get(example.id)
        if tree is None:
            raise KeyError(f"Missing tree for example {example.id}")
        missing = [uid for uid in example.table_uids if uid not in tables]
        if missing:
            raise KeyError(f"Missing gold table(s) for example {example.id}: {missing}")
        tree_nodes = indexed_nodes(tree["root"])
        queries = [node_query(node) or example.claim for _, node in tree_nodes]
        nodes = {}
        if setting == "open":
            results = retriever.retrieve_many(queries)
            for (path, _), result in zip(tree_nodes, results):
                nodes[path] = result
        else:
            for (path, _), query in zip(tree_nodes, queries):
                rows = row_ranker.rank(
                    query, all_rows(example.table_uids, tables), row_top_k,
                )
                nodes[path] = {
                    "query": query,
                    "coarse_tables": [],
                    "tables": [{"uid": uid, "score": 1.0} for uid in example.table_uids],
                    "rows": [row.to_dict() for row in rows],
                }
        yield {
            "id": example.id, "claim": example.claim, "label": example.label,
            "gold_table_uids": list(example.table_uids), "nodes": nodes,
        }
