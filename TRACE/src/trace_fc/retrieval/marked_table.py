"""Legacy-compatible cross-encoder table formatting."""

from __future__ import annotations

import re

from trace_fc.data import Table

SPECIAL_TOKENS = ["[KEY]", "[/KEY]", "[NUM]", "[/NUM]", "[HDR]", "[/HDR]", "[ROW]", "[/ROW]"]
_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "at", "by", "with",
    "is", "are", "was", "were", "from", "as", "that", "this", "it", "its", "be",
    "or", "if", "then", "than", "which",
}
_TOKEN_RE = re.compile(r"[a-zA-Z0-9%]+\b")
_NUMBER_RE = re.compile(
    r"\d+(?:[.,]\d+)?(?:\s*(?:%|km|m|cm|mm|h|min|s|hr|hrs|hour|hours|sec|secs|"
    r"second|seconds|kg|g|mg|\$))?",
    flags=re.IGNORECASE,
)


def query_tokens(query: str) -> list[str]:
    return [
        token for token in (item.lower() for item in _TOKEN_RE.findall(query))
        if token not in _STOPWORDS and len(token) > 1
    ]


def _mark_matches(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> tuple[str, bool]:
    """Apply non-overlapping markers without invalidating source offsets."""
    matches: list[tuple[int, int, str]] = []
    for marker, pattern in patterns:
        matches.extend((match.start(), match.end(), marker) for match in pattern.finditer(text))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, str]] = []
    end = -1
    for start, stop, marker in matches:
        if start >= end:
            selected.append((start, stop, marker))
            end = stop
    if not selected:
        return text, False
    parts: list[str] = []
    cursor = 0
    closing = {"KEY": "[/KEY]", "NUM": "[/NUM]", "HDR": "[/HDR]"}
    for start, stop, marker in selected:
        parts.extend([text[cursor:start], f"[{marker}]", text[start:stop], closing[marker]])
        cursor = stop
    parts.append(text[cursor:])
    return "".join(parts), True


def _word_patterns(words: list[str], marker: str) -> list[tuple[str, re.Pattern[str]]]:
    return [
        (marker, re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE))
        for word in words
    ]


def serialize_marked_table(query: str, table: Table, max_rows: int = 50) -> str:
    words = query_tokens(query)
    query_numbers = {match.group(0) for match in _NUMBER_RE.finditer(query)}
    title, _ = _mark_matches(table.title, _word_patterns(words, "KEY"))
    headers = [
        _mark_matches(header, _word_patterns(words, "HDR"))[0] for header in table.header
    ]
    rows = []
    for row in table.data[:max_rows]:
        cells = []
        row_hit = False
        for cell in row:
            number_patterns = [
                ("NUM", re.compile(re.escape(number), re.IGNORECASE))
                for number in query_numbers if number
            ]
            marked, hit = _mark_matches(cell, number_patterns + _word_patterns(words, "KEY"))
            cells.append(marked)
            row_hit = row_hit or hit
        line = " | ".join(cells)
        rows.append(f"- [ROW]{line}[/ROW]" if row_hit else f"- {line}")
    return "\n".join([
        f"Q: {query}", f"T: Title: {title}", f"Header: | {' | '.join(headers)} |",
        "Rows:", *rows,
    ])
