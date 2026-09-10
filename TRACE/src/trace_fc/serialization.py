"""Canonical table, row, and node-evidence serialization."""

from __future__ import annotations

from dataclasses import dataclass

from trace_fc.data import Table

NODE_SPECIAL_TOKENS = [
    "[Query]", "[Evidence]", "[Title]", *[f"[Row{index}]" for index in range(1, 65)]
]


@dataclass(frozen=True)
class EvidenceRow:
    table_uid: str
    table_title: str
    header: tuple[str, ...]
    cells: tuple[str, ...]
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "table_uid": self.table_uid, "table_title": self.table_title,
            "header": list(self.header), "cells": list(self.cells), "score": self.score,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "EvidenceRow":
        return cls(
            table_uid=str(value["table_uid"]),
            table_title=str(value.get("table_title", "")),
            header=tuple(map(str, value.get("header", []))),
            cells=tuple(map(str, value.get("cells", []))),
            score=float(value.get("score", 0.0)),
        )


def serialize_table(table: Table, max_rows: int = 32) -> str:
    title = table.title[:512]
    header = " | ".join(table.header)[:512]
    rows = " || ".join(" | ".join(row) for row in table.data[:max_rows])[:2000]
    return f"Title: {title} || Header: {header} || Rows: {rows}"


def serialize_row(table: Table, row: tuple[str, ...]) -> str:
    pairs = [f"{head}: {cell}" for head, cell in zip(table.header, row)]
    return f"[TITLE] {table.title} [ROW] " + " | ".join(pairs)


def serialize_node_evidence(query: str, rows: list[EvidenceRow]) -> str:
    parts = [f"[Query] {query}", "[Evidence]"]
    for index, row in enumerate(rows, start=1):
        pairs = [f"{head}: {cell}" for head, cell in zip(row.header, row.cells)]
        parts.append(f"[Row{index}] [Title] {row.table_title} " + " | ".join(pairs))
    return "\n".join(parts)
