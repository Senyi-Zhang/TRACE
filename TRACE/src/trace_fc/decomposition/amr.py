"""AMR-guided claim-tree construction.

This module preserves the behavior of the original TRACE AMR implementation:
PropBank predicates become proposition nodes, coordination and polarity become
operators, non-predicate subgraphs are collapsed into arguments, and AMR
reentrancies become references or shared-entity links.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

_ALIGN_RE = re.compile(r"~(?:[a-zA-Z]+\.)?[0-9]+(?:,[0-9]+)*")
_TOKEN_RE = re.compile(
    r'''\s*("(?:\\.|[^"\\])*"|\(|\)|/|:[A-Za-z0-9_.-]+|[^\s()/]+)'''
)
_PRED_RE = re.compile(r".+-\d\d$")
_ARG_OF_RE = re.compile(r"^:ARG\d+-of$")

_INVERTIBLE_BASE_ROLES = {
    ":accompanier", ":age", ":beneficiary", ":cause", ":concession", ":condition",
    ":degree", ":destination", ":direction", ":domain", ":duration", ":example",
    ":extent", ":frequency", ":instrument", ":li", ":location", ":manner", ":medium",
    ":mod", ":mode", ":name", ":ord", ":part", ":path", ":polarity", ":poss",
    ":purpose", ":quant", ":range", ":scale", ":source", ":subevent", ":subset",
    ":superset", ":time", ":topic", ":unit", ":value",
}
_OPERATORS = {"and": "AND", "or": "OR", "either": "OR", "xor": "XOR"}
_DROP_ARGUMENT_ROLES = {":wiki", ":polarity"}
_DATE_ROLES = (":year", ":month", ":day")
_ENTITY_DETAIL_ROLES = {":mod", ":poss", ":quant", ":unit", ":value", ":age"}


def _tokens(text: str) -> list[str]:
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    tokens = _TOKEN_RE.findall(_ALIGN_RE.sub("", "\n".join(lines)))
    if not tokens:
        raise ValueError("No PENMAN tokens found")
    return tokens


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] == '"':
        try:
            return json.loads(token)
        except Exception:
            return token[1:-1]
    return token


@dataclass(frozen=True)
class Edge:
    source: str
    role: str
    target: str
    target_is_var: bool = False


@dataclass
class AMRGraph:
    top: str
    instances: dict[str, str]
    raw_edges: list[Edge]
    edges: list[Edge] = field(init=False)
    outgoing: dict[str, list[Edge]] = field(init=False)
    incoming: dict[str, list[Edge]] = field(init=False)

    def __post_init__(self) -> None:
        self.edges = self._canonicalize(self.raw_edges)
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        for edge in self.edges:
            self.outgoing[edge.source].append(edge)
            if edge.target_is_var:
                self.incoming[edge.target].append(edge)

    @classmethod
    def from_penman(cls, text: str) -> "AMRGraph":
        tokens = _tokens(text)
        instances: dict[str, str] = {}
        edges: list[Edge] = []
        cursor = 0

        def parse_node() -> str:
            nonlocal cursor
            if cursor >= len(tokens) or tokens[cursor] != "(":
                raise ValueError(f"Expected '(' at token {cursor}")
            cursor += 1
            if cursor >= len(tokens):
                raise ValueError("Unexpected end after '('")
            variable = tokens[cursor]
            cursor += 1
            if cursor >= len(tokens) or tokens[cursor] != "/":
                raise ValueError(f"Expected '/' after variable {variable!r}")
            cursor += 1
            if cursor >= len(tokens):
                raise ValueError(f"Missing concept for variable {variable!r}")
            instances[variable] = _unquote(tokens[cursor])
            cursor += 1
            while cursor < len(tokens) and tokens[cursor] != ")":
                role = tokens[cursor]
                if not role.startswith(":"):
                    raise ValueError(f"Expected role at token {cursor}, got {role!r}")
                cursor += 1
                if cursor >= len(tokens):
                    raise ValueError(f"Missing target for role {role}")
                if tokens[cursor] == "(":
                    target = parse_node()
                    edges.append(Edge(variable, role, target, True))
                else:
                    edges.append(Edge(variable, role, _unquote(tokens[cursor]), False))
                    cursor += 1
            if cursor >= len(tokens):
                raise ValueError(f"Unclosed node {variable!r}")
            cursor += 1
            return variable

        top = parse_node()
        if cursor != len(tokens):
            raise ValueError(f"Unexpected trailing PENMAN tokens at {cursor}")
        resolved = [
            Edge(e.source, e.role, e.target, e.target_is_var or e.target in instances)
            for e in edges
        ]
        return cls(top=top, instances=instances, raw_edges=resolved)

    @staticmethod
    def _canonicalize(edges: list[Edge]) -> list[Edge]:
        canonical: list[Edge] = []
        for edge in edges:
            base = edge.role[:-3] if edge.role.endswith("-of") else edge.role
            invert = edge.target_is_var and (
                bool(_ARG_OF_RE.match(edge.role))
                or (edge.role.endswith("-of") and base in _INVERTIBLE_BASE_ROLES)
            )
            canonical.append(
                Edge(edge.target, base, edge.source, True) if invert else edge
            )
        return canonical


def is_predicate(concept: str | None) -> bool:
    return bool(concept and _PRED_RE.match(concept))


def predicate_lemma(concept: str) -> str:
    return re.sub(r"-\d\d$", "", concept).replace("-", " ")


def clean_role(role: str) -> str:
    return role[1:] if role.startswith(":") else role


class ClaimTreeBuilder:
    def __init__(self, graph: AMRGraph):
        self.graph = graph
        self.entity_ids: dict[str, str] = {}
        self.proposition_ids: dict[str, str] = {}
        self.built: set[str] = set()
        self.active: set[str] = set()
        self.reserved: set[str] = set()

    def entity_id(self, variable: str) -> str:
        return self.entity_ids.setdefault(variable, f"e{len(self.entity_ids) + 1}")

    def proposition_id(self, variable: str) -> str:
        return self.proposition_ids.setdefault(variable, f"p{len(self.proposition_ids) + 1}")

    @staticmethod
    def literal(value: str) -> str:
        return {"-": "false", "+": "true"}.get(value, value)

    def entity_label(self, variable: str, seen: set[str] | None = None) -> str:
        seen = set() if seen is None else set(seen)
        if variable in seen:
            return self.graph.instances.get(variable, variable)
        seen.add(variable)
        concept = self.graph.instances.get(variable, variable)
        outgoing = self.graph.outgoing.get(variable, [])
        name_edge = next(
            (edge for edge in outgoing if edge.role == ":name" and edge.target_is_var), None
        )
        if name_edge:
            operations = []
            for edge in self.graph.outgoing.get(name_edge.target, []):
                match = re.match(r":op(\d+)$", edge.role)
                if match and not edge.target_is_var:
                    operations.append((int(match.group(1)), self.literal(edge.target)))
            if operations:
                return " ".join(value for _, value in sorted(operations))
        if concept == "date-entity":
            values = {
                edge.role: self.literal(edge.target)
                for edge in outgoing
                if edge.role in _DATE_ROLES and not edge.target_is_var
            }
            if values:
                return "-".join(values[role] for role in _DATE_ROLES if role in values)
        details = []
        for edge in outgoing:
            if edge.role not in _ENTITY_DETAIL_ROLES:
                continue
            if edge.target_is_var and not is_predicate(self.graph.instances.get(edge.target)):
                details.append(self.entity_label(edge.target, seen))
            elif not edge.target_is_var:
                details.append(self.literal(edge.target))
        if not details:
            return concept
        if concept in {"quantity", "monetary-quantity", "temporal-quantity"}:
            return " ".join([concept, *details])
        return " ".join([*details, concept])

    def argument(self, edge: Edge) -> dict[str, Any]:
        role = clean_role(edge.role)
        if not edge.target_is_var:
            return {"role": role, "value": self.literal(edge.target)}
        concept = self.graph.instances.get(edge.target, edge.target)
        if is_predicate(concept):
            return {
                "role": role,
                "proposition_ref": self.proposition_id(edge.target),
                "predicate": concept,
            }
        return {
            "role": role,
            "entity": self.entity_id(edge.target),
            "value": self.entity_label(edge.target),
            "concept": concept,
        }

    def build(self) -> dict[str, Any]:
        root = self._build_any(self.graph.top)
        return {"type": "claim_tree", "root": root, "entities": self.entity_table()}

    def _build_any(self, variable: str) -> dict[str, Any]:
        concept = self.graph.instances.get(variable, variable)
        if concept in _OPERATORS:
            return self._build_operator(variable, _OPERATORS[concept])
        if is_predicate(concept):
            return self._build_predicate(variable)
        neighbors = [
            edge.source for edge in self.graph.incoming.get(variable, [])
            if is_predicate(self.graph.instances.get(edge.source))
        ]
        if neighbors:
            children = [self._build_predicate(item) for item in neighbors]
            return children[0] if len(children) == 1 else {
                "kind": "operator", "operator": "AND", "children": children,
            }
        return {
            "kind": "entity", "entity": self.entity_id(variable),
            "value": self.entity_label(variable), "concept": concept,
        }

    def _build_operator(self, variable: str, operator: str) -> dict[str, Any]:
        operands = []
        for edge in self.graph.outgoing.get(variable, []):
            match = re.match(r":op(\d+)$", edge.role)
            if match and edge.target_is_var:
                operands.append((int(match.group(1)), edge.target))
        reserved = {
            target for _, target in operands if is_predicate(self.graph.instances.get(target))
        }
        self.reserved.update(reserved)
        children = [self._build_any(target) for _, target in sorted(operands)]
        self.reserved.difference_update(reserved)
        return {"kind": "operator", "operator": operator, "children": children}

    def _build_predicate(self, variable: str) -> dict[str, Any]:
        pid = self.proposition_id(variable)
        concept = self.graph.instances[variable]
        if variable in self.active or variable in self.built:
            return {"kind": "proposition_ref", "ref": pid, "predicate": concept}
        self.active.add(variable)
        outgoing = self.graph.outgoing.get(variable, [])
        negated = any(
            edge.role == ":polarity" and not edge.target_is_var and edge.target == "-"
            for edge in outgoing
        )
        arguments: list[dict[str, Any]] = []
        children: list[dict[str, Any]] = []
        related: list[tuple[str, str]] = []
        for edge in outgoing:
            if edge.role in _DROP_ARGUMENT_ROLES:
                continue
            if edge.target_is_var:
                target_concept = self.graph.instances.get(edge.target)
                if target_concept in _OPERATORS:
                    children.append({
                        "relation": clean_role(edge.role),
                        "node": self._build_operator(edge.target, _OPERATORS[target_concept]),
                    })
                elif is_predicate(target_concept):
                    arguments.append(self.argument(edge))
                    children.append({
                        "relation": clean_role(edge.role),
                        "node": self._build_predicate(edge.target),
                    })
                else:
                    arguments.append(self.argument(edge))
                    for incoming in self.graph.incoming.get(edge.target, []):
                        pred = incoming.source
                        if (
                            pred != variable and is_predicate(self.graph.instances.get(pred))
                            and pred not in self.built and pred not in self.active
                            and pred not in self.reserved
                        ):
                            related.append((clean_role(incoming.role), pred))
            else:
                arguments.append(self.argument(edge))
        seen_related: set[str] = set()
        for role, pred in related:
            if pred not in seen_related:
                seen_related.add(pred)
                children.append({
                    "relation": f"shared_entity:{role}", "node": self._build_predicate(pred),
                })
        node: dict[str, Any] = {
            "kind": "proposition", "id": pid, "predicate": concept,
            "lemma": predicate_lemma(concept), "arguments": arguments,
        }
        if children:
            node["children"] = children
        self.active.remove(variable)
        self.built.add(variable)
        return {"kind": "operator", "operator": "NOT", "children": [node]} if negated else node

    def entity_table(self) -> dict[str, dict[str, str]]:
        return {
            eid: {
                "amr_var": variable,
                "concept": self.graph.instances.get(variable, variable),
                "value": self.entity_label(variable),
            }
            for variable, eid in sorted(self.entity_ids.items(), key=lambda item: int(item[1][1:]))
        }


def iter_children(node: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if node.get("kind") == "operator":
        yield from node.get("children", [])
    elif node.get("kind") == "proposition":
        for child in node.get("children", []):
            yield child["node"]


def node_query(node: dict[str, Any]) -> str:
    """Derive a deterministic lexical query from any projected tree node."""
    kind = node.get("kind")
    if kind == "proposition":
        terms = [
            str(argument["value"])
            for argument in node.get("arguments", [])
            if argument.get("value")
        ]
        terms.append(node.get("lemma", predicate_lemma(node.get("predicate", ""))))
    elif kind == "entity":
        terms = [node.get("value", "")]
    elif kind == "proposition_ref":
        terms = [predicate_lemma(node.get("predicate", ""))]
    else:
        terms = [node_query(child) for child in iter_children(node)]
    unique = []
    for term in terms:
        term = str(term).strip()
        if term and term not in unique:
            unique.append(term)
    return " ".join(unique)


def indexed_nodes(root: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return stable pre-order path IDs and nodes."""
    output: list[tuple[str, dict[str, Any]]] = []

    def visit(node: dict[str, Any], path: str) -> None:
        output.append((path, node))
        for index, child in enumerate(iter_children(node)):
            visit(child, f"{path}.{index}")

    visit(root, "n0")
    return output


def postorder_nodes(root: dict[str, Any]) -> list[tuple[str, dict[str, Any], list[str]]]:
    output: list[tuple[str, dict[str, Any], list[str]]] = []

    def visit(node: dict[str, Any], path: str) -> None:
        children = list(iter_children(node))
        child_ids = [f"{path}.{index}" for index in range(len(children))]
        for child_id, child in zip(child_ids, children):
            visit(child, child_id)
        output.append((path, node, child_ids))

    visit(root, "n0")
    return output


def atomic_propositions(tree: dict[str, Any]) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    seen: set[str] = set()

    def walk(node: dict[str, Any], negated: bool = False) -> None:
        kind = node.get("kind")
        if kind == "operator":
            child_negated = not negated if node.get("operator") == "NOT" else negated
            for child in iter_children(node):
                walk(child, child_negated)
        elif kind == "proposition":
            pid = node["id"]
            if pid not in seen:
                seen.add(pid)
                atoms.append({
                    "id": pid, "predicate": node["predicate"], "lemma": node["lemma"],
                    "negated": negated, "arguments": node.get("arguments", []),
                    "retrieval_query": node_query(node),
                })
            for child in iter_children(node):
                walk(child, negated)

    walk(tree["root"])
    return atoms


def decompose_amr(penman: str) -> dict[str, Any]:
    tree = ClaimTreeBuilder(AMRGraph.from_penman(penman)).build()
    tree["atomic_propositions"] = atomic_propositions(tree)
    return tree


@lru_cache(maxsize=2)
def load_transition_parser(model: str = "AMR3-structbart-L") -> Any:
    try:
        from transition_amr_parser.parse import AMRParser
    except ImportError as exc:
        raise RuntimeError(
            "transition-amr-parser is required for raw claims; install it or provide PENMAN AMR"
        ) from exc
    return AMRParser.from_pretrained(model)


def claim_to_penman(claim: str, model: str = "AMR3-structbart-L") -> str:
    parser = load_transition_parser(model)
    tokens, _ = parser.tokenize(claim)
    annotations, machines = parser.parse_sentence(tokens)
    machine = machines[0] if isinstance(machines, (list, tuple)) else machines
    if hasattr(machine, "get_amr"):
        amr = machine.get_amr()
        if hasattr(amr, "to_penman"):
            try:
                return amr.to_penman(jamr=False, isi=False)
            except TypeError:
                return amr.to_penman(jamr=False, isi=True)
    if isinstance(annotations, str):
        return annotations
    if isinstance(annotations, (list, tuple)) and annotations:
        return str(annotations[0])
    raise RuntimeError("Could not obtain PENMAN from transition-amr-parser output")


def decompose_claim(claim: str, model: str = "AMR3-structbart-L") -> dict[str, Any]:
    penman = claim_to_penman(claim, model)
    tree = decompose_amr(penman)
    tree.update({"claim": claim, "amr": penman})
    return tree
