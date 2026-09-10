from trace_fc.decomposition.amr import (
    decompose_amr,
    indexed_nodes,
    node_query,
    postorder_nodes,
)


SAMPLE = '''
(a / and
  :op1 (b / become-01
    :ARG1 (p / person :name (n / name :op1 "Nelson" :op2 "Mandela"))
    :ARG2 (r / president :poss (c / country :name (n2 / name :op1 "South" :op2 "Africa")))
    :time (d / date-entity :year 1994))
  :op2 (m / move-01
    :ARG0 p
    :ARG1 (c2 / city :name (n3 / name :op1 "Pretoria"))
    :time (d2 / date-entity :year 2000)))
'''


def test_amr_projection_and_queries_are_stable():
    tree = decompose_amr(SAMPLE)
    assert tree["root"]["kind"] == "operator"
    assert tree["root"]["operator"] == "AND"
    assert len(tree["atomic_propositions"]) == 2
    assert "Nelson Mandela" in tree["atomic_propositions"][0]["retrieval_query"]
    paths = [path for path, _ in indexed_nodes(tree["root"])]
    assert paths == ["n0", "n0.0", "n0.1"]
    assert [path for path, _, _ in postorder_nodes(tree["root"])] == ["n0.0", "n0.1", "n0"]
    assert node_query(tree["root"])


def test_reentrancy_is_not_duplicated_as_a_full_proposition():
    tree = decompose_amr(SAMPLE)
    proposition_ids = [item["id"] for item in tree["atomic_propositions"]]
    assert len(proposition_ids) == len(set(proposition_ids))

