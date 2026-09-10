from types import SimpleNamespace

import torch
from torch import nn
from transformers import BatchEncoding

from trace_fc.models.composition import FiLM, PMA
from trace_fc.models.verifier import TraceVerifier


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "[EOS]"
    unk_token = "[UNK]"

    def add_special_tokens(self, _tokens):
        return 0

    def __call__(self, texts, **_kwargs):
        lengths = [max(1, min(4, len(text.split()))) for text in texts]
        ids = torch.zeros((len(texts), 4), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, length in enumerate(lengths):
            ids[index, :length] = torch.arange(1, length + 1)
            mask[index, :length] = 1
        return BatchEncoding({"input_ids": ids, "attention_mask": mask})


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(8, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_pma_and_film_shapes():
    values = torch.randn(3, 8)
    pooled = PMA(8, num_heads=2)(values)
    output = FiLM(8)(torch.randn(8), pooled)
    assert pooled.shape == (8,)
    assert output.shape == (8,)


def test_tree_forward_and_root_loss_backpropagate(monkeypatch):
    monkeypatch.setattr(
        "trace_fc.models.verifier.AutoTokenizer.from_pretrained",
        lambda *_a, **_k: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "trace_fc.models.verifier.AutoModel.from_pretrained",
        lambda *_a, **_k: FakeEncoder(),
    )
    model = TraceVerifier("fake", num_heads=2, node_batch_size=8)
    tree = {
        "root": {
            "kind": "operator", "operator": "AND", "children": [
                {
                    "kind": "proposition", "id": "p1", "predicate": "win-01",
                    "lemma": "win", "arguments": [],
                },
                {
                    "kind": "proposition", "id": "p2", "predicate": "hold-01",
                    "lemma": "hold", "arguments": [],
                },
            ],
        }
    }
    nodes = {
        path: {"query": path, "rows": []} for path in ("n0", "n0.0", "n0.1")
    }
    logits = model([{"tree": tree, "evidence": {"nodes": nodes}}])
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
    loss.backward()
    assert logits.shape == (1, 2)
    assert model.film.gamma_beta[0].weight.grad is not None
    assert model.encoder.embedding.weight.grad is not None
