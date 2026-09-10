"""Binary fact-checking metrics."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    if not labels:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", labels=[0, 1], zero_division=0)
        ),
    }
