import pytest

from trace_fc.data import ClaimExample, Table


def test_claim_example_normalizes_legacy_fields():
    example = ClaimExample.from_dict({
        "id": 192,
        "claim": "A claim. ",
        "table_uid_1": " table_1  ",
        "table_uid_2": "table_2  ",
        "category": "CONJUNCTIVE  ",
        "label": "refuted ",
    })
    assert example.id == "192"
    assert example.claim == "A claim."
    assert example.table_uids == ("table_1", "table_2")
    assert example.category == "CONJUNCTIVE"
    assert example.label_id == 0


def test_table_schema_uses_header_and_data():
    table = Table.from_dict("t1", {
        "title": "Race", "header": ["Rider", "Time"],
        "data": [["Eros Capecchi", "3h 20"]],
    })
    assert table.uid == "t1"
    assert table.header == ("Rider", "Time")
    assert table.data == (("Eros Capecchi", "3h 20"),)


def test_invalid_label_fails_early():
    with pytest.raises(ValueError, match="invalid label"):
        ClaimExample.from_dict({"id": 1, "claim": "x", "label": "NEI"})
