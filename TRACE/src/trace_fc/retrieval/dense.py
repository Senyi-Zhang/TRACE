"""SBERT coarse table index with validated on-disk persistence."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from trace_fc.data import Table
from trace_fc.serialization import serialize_table

LOGGER = logging.getLogger(__name__)


def _load_faiss():
    try:
        import faiss
    except ImportError:
        return None
    return faiss


def _fingerprint(model_name: str, ids: Sequence[str], texts: Sequence[str]) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for uid, value in zip(ids, texts):
        digest.update(uid.encode("utf-8"))
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


class DenseTableIndex:
    def __init__(self, model_name: str, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.model = model
        self.table_ids: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.index = None
        self.fingerprint = ""

    def build(self, tables: dict[str, Table], batch_size: int = 256) -> None:
        self.table_ids = list(tables)
        texts = [serialize_table(tables[uid]) for uid in self.table_ids]
        self.embeddings = np.asarray(self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ), dtype=np.float32)
        self.fingerprint = _fingerprint(self.model_name, self.table_ids, texts)
        faiss = _load_faiss()
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return self.search_many([query], top_k)[0]

    def search_many(
        self, queries: list[str], top_k: int, batch_size: int = 256,
    ) -> list[list[tuple[str, float]]]:
        if self.embeddings is None:
            raise RuntimeError("DenseTableIndex has not been built or loaded")
        top_k = min(max(0, top_k), len(self.table_ids))
        if top_k == 0:
            return [[] for _ in queries]
        vectors = np.asarray(self.model.encode(
            queries, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True,
        ), dtype=np.float32)
        if self.index is not None:
            score_matrix, index_matrix = self.index.search(vectors, top_k)
        else:
            similarities = vectors @ self.embeddings.T
            index_matrix = np.argpartition(-similarities, top_k - 1, axis=1)[:, :top_k]
            row_indices = np.arange(len(queries))[:, None]
            order = np.argsort(-similarities[row_indices, index_matrix], axis=1)
            index_matrix = index_matrix[row_indices, order]
            score_matrix = similarities[row_indices, index_matrix]
        return [
            [(self.table_ids[index], float(score)) for index, score in zip(indices, scores)]
            for indices, scores in zip(index_matrix, score_matrix)
        ]

    def save(self, directory: str | Path) -> None:
        if self.embeddings is None:
            raise RuntimeError("Cannot save an empty index")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings, allow_pickle=False)
        with (directory / "table_ids.json").open("w", encoding="utf-8") as stream:
            json.dump(self.table_ids, stream, ensure_ascii=False)
        metadata = {
            "model_name": self.model_name,
            "dimension": int(self.embeddings.shape[1]),
            "number_of_tables": len(self.table_ids),
            "fingerprint": self.fingerprint,
            "normalized": True,
        }
        with (directory / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
        faiss = _load_faiss()
        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, str(directory / "faiss.index"))

    def validate_tables(self, tables: dict[str, Table]) -> None:
        ids = list(tables)
        texts = [serialize_table(tables[uid]) for uid in ids]
        current = _fingerprint(self.model_name, ids, texts)
        if ids != self.table_ids or current != self.fingerprint:
            raise ValueError(
                "The loaded dense index does not match the configured table corpus; "
                "run build-index again"
            )

    @classmethod
    def load(cls, directory: str | Path, model=None) -> "DenseTableIndex":
        directory = Path(directory)
        with (directory / "metadata.json").open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        instance = cls(metadata["model_name"], model=model)
        with (directory / "table_ids.json").open("r", encoding="utf-8") as stream:
            instance.table_ids = json.load(stream)
        instance.embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
        if instance.embeddings.shape != (metadata["number_of_tables"], metadata["dimension"]):
            raise ValueError("Dense index files are inconsistent")
        instance.fingerprint = metadata.get("fingerprint", "")
        faiss = _load_faiss()
        faiss_path = directory / "faiss.index"
        if faiss is not None and faiss_path.exists():
            instance.index = faiss.read_index(str(faiss_path))
        return instance
