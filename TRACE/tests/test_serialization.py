from trace_fc.data import Table
from trace_fc.retrieval.marked_table import serialize_marked_table
from trace_fc.serialization import EvidenceRow, serialize_node_evidence, serialize_table


TABLE = Table.from_dict("race", {
    "title": "2011 Giro d'Italia",
    "header": ["Rider", "Time"],
    "data": [["Eros Capecchi", "3h 20 38"], ["Marco Pinotti", "same time"]],
})


def test_table_serialization_preserves_structure():
    text = serialize_table(TABLE)
    assert text == (
        "Title: 2011 Giro d'Italia || Header: Rider | Time || Rows: "
        "Eros Capecchi | 3h 20 38 || Marco Pinotti | same time"
    )


def test_marked_serialization_does_not_corrupt_offsets():
    text = serialize_marked_table("Eros 2011", TABLE)
    assert "[KEY]Eros[/KEY]" in text
    assert "[KEY]2011[/KEY]" in text


def test_node_evidence_template():
    row = EvidenceRow("race", TABLE.title, TABLE.header, TABLE.data[0], 1.0)
    text = serialize_node_evidence("Eros time", [row])
    assert text.startswith("[Query] Eros time\n[Evidence]\n[Row1]")
    assert "Rider: Eros Capecchi" in text
