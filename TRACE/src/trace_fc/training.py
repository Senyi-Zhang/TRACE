"""End-to-end verifier training and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from trace_fc.data import ClaimExample
from trace_fc.metrics import classification_metrics
from trace_fc.models.verifier import TraceVerifier
from trace_fc.utils import save_json

LOGGER = logging.getLogger(__name__)


class TraceDataset(Dataset):
    def __init__(
        self,
        examples: list[ClaimExample],
        trees: dict[str, dict],
        evidence: dict[str, dict],
    ):
        self.items = []
        for example in examples:
            if example.id not in trees:
                raise KeyError(f"Missing tree for example {example.id}")
            if example.id not in evidence:
                raise KeyError(f"Missing evidence for example {example.id}")
            self.items.append({
                "id": example.id, "claim": example.claim, "label": example.label_id,
                "tree": trees[example.id], "evidence": evidence[example.id],
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def collate_items(items: list[dict]) -> list[dict]:
    return items


@torch.inference_mode()
def evaluate(
    model: TraceVerifier, loader: DataLoader,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    records = []
    for items in tqdm(loader, desc="Evaluating"):
        logits = model(items)
        probabilities = torch.softmax(logits, dim=-1).detach().cpu()
        batch_predictions = probabilities.argmax(dim=-1).tolist()
        for item, prediction, probability in zip(items, batch_predictions, probabilities.tolist()):
            labels.append(item["label"])
            predictions.append(prediction)
            records.append({
                "id": item["id"], "claim": item["claim"], "gold": item["label"],
                "prediction": prediction, "probabilities": probability,
            })
    return classification_metrics(labels, predictions), records


def train_verifier(
    model: TraceVerifier,
    train_dataset: TraceDataset,
    dev_dataset: TraceDataset,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    mixed_precision: bool,
    num_workers: int,
) -> dict[str, Any]:
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_items,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_items,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    updates_per_epoch = (
        len(train_loader) + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    total_updates = max(1, epochs * updates_per_epoch)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(warmup_ratio * total_updates),
        num_training_steps=total_updates,
    )
    use_amp = mixed_precision and model.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history = []
    best_f1 = -1.0
    best_state = None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Verifier epoch {epoch}/{epochs}")
        for step, items in enumerate(progress, start=1):
            labels = torch.tensor([item["label"] for item in items], device=model.device)
            group_start = ((step - 1) // gradient_accumulation_steps) * (
                gradient_accumulation_steps
            ) + 1
            group_end = min(
                group_start + gradient_accumulation_steps - 1, len(train_loader)
            )
            group_size = group_end - group_start + 1
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(items)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()
            should_update = step % gradient_accumulation_steps == 0 or step == len(train_loader)
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        metrics, _ = evaluate(model, dev_loader)
        epoch_record = {
            "epoch": epoch, "train_loss": total_loss / max(1, len(train_loader)), **metrics,
        }
        history.append(epoch_record)
        LOGGER.info("epoch=%d loss=%.4f accuracy=%.4f macro_f1=%.4f", epoch,
                    epoch_record["train_loss"], metrics["accuracy"], metrics["macro_f1"])
        if best_state is None or metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir = Path(output_dir)
    model.save_checkpoint(output_dir, metadata={"best_dev_macro_f1": best_f1})
    result = {"best_dev_macro_f1": best_f1, "history": history}
    save_json(output_dir / "training_metrics.json", result)
    return result
