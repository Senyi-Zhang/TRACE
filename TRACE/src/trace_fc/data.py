"""Dataset and table-corpus contracts used throughout TRACE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

LABEL_TO_ID = {"REFUTED": 0, "SUPPORTED": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


@dataclass(frozen=True)
class ClaimExample:
    id: str
    claim: str
    table_uids: tuple[str, ...]
    label: str
    category: str = ""

    @property
    def label_id(self) -> int:
        try:
            return LABEL_TO_ID[self.label]
        except KeyError as exc:
            raise ValueError(f"Unsupported label {self.label!r} for example {self.id}") from exc

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClaimExample":
        table_keys = (
            "table_uid_1", "table_uid_2", "table_uid1", "table_uid2",
            "gold_table1", "gold_table2",
        )
        seen: set[str] = set()
        table_uids = []
        for key in table_keys:
            uid = _clean(value.get(key))
            if uid and uid not in seen:
                seen.add(uid)
                table_uids.append(uid)
        claim = _clean(value.get("claim"))
        label = _clean(value.get("label")).upper()
        if not claim:
            raise ValueError(f"Example {value.get('id')!r} has an empty claim")
        if label not in LABEL_TO_ID:
            raise ValueError(f"Example {value.get('id')!r} has invalid label {label!r}")
        return cls(
            id=_clean(value.get("id")), claim=claim,
            table_uids=tuple(table_uids), label=label,
            category=_clean(value.get("category")),
        )


@dataclass(frozen=True)
class Table:
    uid: str
    title: str
    header: tuple[str, ...]
    data: tuple[tuple[str, ...], ...]
    url: str = ""
    section_title: str = ""
    section_text: str = ""

    @classmethod
    def from_dict(cls, uid: str, value: dict[str, Any]) -> "Table":
        # Legacy aliases are accepted only at this input boundary.
        header = value.get("header", value.get("headers", [])) or []
        data = value.get("data", value.get("rows", [])) or []
        return cls(
            uid=_clean(uid or value.get("uid")),
            title=_clean(value.get("title")),
            header=tuple(_clean(cell) for cell in header),
            data=tuple(tuple(_clean(cell) for cell in row) for row in data),
            url=_clean(value.get("url")),
            section_title=_clean(value.get("section_title")),
            section_text=_clean(value.get("section_text")),
        )


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def load_claims(path: str | Path) -> list[ClaimExample]:
    examples = [ClaimExample.from_dict(value) for value in read_jsonl(path)]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for example in examples:
        if example.id in seen:
            duplicates.add(example.id)
        seen.add(example.id)
    if duplicates:
        raise ValueError(f"Duplicate example IDs in {path}: {sorted(duplicates)}")
    return examples


def load_tables(path: str | Path) -> dict[str, Table]:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("The table corpus must be a JSON object keyed by table UID")
    tables = {str(uid).strip(): Table.from_dict(str(uid), value) for uid, value in raw.items()}
    if "" in tables:
        raise ValueError("The table corpus contains an empty UID")
    return tables


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
