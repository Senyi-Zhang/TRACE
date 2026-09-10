import numpy as np

from trace_fc.data import Table
from trace_fc.retrieval.dense import DenseTableIndex
from trace_fc.retrieval.pipeline import TwoStageRetriever
from trace_fc.retrieval.reranker import LexicalRowRanker


class FakeSentenceEncoder:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            low = text.lower()
            vector = np.asarray(["mandela" in low, "cycling" in low], dtype=np.float32)
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.stack(vectors)


class FakeTableReranker:
    def score(self, query, table_ids, tables, batch_size, max_rows):
        return [(uid, float("mandela" in tables[uid].title.lower())) for uid in table_ids]

    def score_many(self, queries_and_tables, tables, batch_size, max_rows):
        return [
            self.score(query, table_ids, tables, batch_size, max_rows)
            for query, table_ids in queries_and_tables
        ]


def test_coarse_to_fine_pipeline():
    tables = {
        "politics": Table.from_dict("politics", {
            "title": "Nelson Mandela", "header": ["Year", "Role"],
            "data": [["1994", "President"]],
        }),
        "sport": Table.from_dict("sport", {
            "title": "Cycling", "header": ["Rider"], "data": [["Eros"]],
        }),
    }
    index = DenseTableIndex("fake", model=FakeSentenceEncoder())
    index.build(tables)
    retriever = TwoStageRetriever(
        tables, index, FakeTableReranker(), LexicalRowRanker(),
        coarse_top_k=2, table_top_k=1, row_top_k=1,
    )
    result = retriever.retrieve("Mandela president 1994")
    assert result["tables"][0]["uid"] == "politics"
    assert result["rows"][0]["table_uid"] == "politics"
