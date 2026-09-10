"""End-to-end coarse-to-fine retrieval for one semantic-tree node."""

from __future__ import annotations

from trace_fc.data import Table
from trace_fc.retrieval.dense import DenseTableIndex
from trace_fc.retrieval.reranker import CrossEncoderReranker, all_rows


class TwoStageRetriever:
    def __init__(
        self,
        tables: dict[str, Table],
        dense_index: DenseTableIndex,
        table_reranker: CrossEncoderReranker,
        row_ranker,
        coarse_top_k: int = 200,
        table_top_k: int = 10,
        row_top_k: int = 10,
        batch_size: int = 16,
        max_table_rows: int = 50,
    ):
        self.tables = tables
        self.dense_index = dense_index
        self.table_reranker = table_reranker
        self.row_ranker = row_ranker
        self.coarse_top_k = coarse_top_k
        self.table_top_k = table_top_k
        self.row_top_k = row_top_k
        self.batch_size = batch_size
        self.max_table_rows = max_table_rows

    def retrieve(self, query: str) -> dict:
        return self.retrieve_many([query])[0]

    def retrieve_many(self, queries: list[str]) -> list[dict]:
        coarse_results = self.dense_index.search_many(queries, self.coarse_top_k)
        query_candidates = [
            (query, [uid for uid, _ in coarse])
            for query, coarse in zip(queries, coarse_results)
        ]
        fine_results = self.table_reranker.score_many(
            query_candidates, self.tables, self.batch_size, self.max_table_rows,
        )
        results = []
        for query, coarse, fine in zip(queries, coarse_results, fine_results):
            fine = fine[:self.table_top_k]
            table_ids = [uid for uid, _ in fine]
            rows = self.row_ranker.rank(
                query, all_rows(table_ids, self.tables), self.row_top_k,
            )
            results.append({
                "query": query,
                "coarse_tables": [
                    {"uid": uid, "score": score} for uid, score in coarse
                ],
                "tables": [{"uid": uid, "score": score} for uid, score in fine],
                "rows": [row.to_dict() for row in rows],
            })
        return results
