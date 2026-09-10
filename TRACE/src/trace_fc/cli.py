"""Command-line interface for all reproducible TRACE stages."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from trace_fc.config import TraceConfig, load_config
from trace_fc.data import ClaimExample, ID_TO_LABEL, load_claims, load_tables, write_jsonl
from trace_fc.decomposition.amr import decompose_claim
from trace_fc.evidence import build_evidence_records, load_evidence_store, load_tree_store
from trace_fc.models.verifier import TraceVerifier
from trace_fc.retrieval.dense import DenseTableIndex
from trace_fc.retrieval.pipeline import TwoStageRetriever
from trace_fc.retrieval.reranker import (
    CrossEncoderReranker,
    DenseRowRanker,
    LexicalRowRanker,
)
from trace_fc.retrieval.training import mine_pairs, split_by_qid, train_reranker
from trace_fc.training import TraceDataset, collate_items, evaluate, train_verifier
from trace_fc.utils import configure_logging, save_json, set_seed

LOGGER = logging.getLogger(__name__)


def _split_file(config: TraceConfig, split: str) -> str:
    return config.data.train_file if split == "train" else config.data.test_file


def _tree_file(config: TraceConfig, split: str) -> Path:
    return Path(config.data.tree_dir) / f"{config.data.dataset}_{split}.jsonl"


def _evidence_file(config: TraceConfig, split: str) -> Path:
    return Path(config.data.evidence_dir) / f"{config.data.dataset}_{config.setting}_{split}.jsonl"


def _device(value: str | None) -> str:
    return value or ("cuda" if torch.cuda.is_available() else "cpu")


def _row_ranker(config: TraceConfig, dense_index: DenseTableIndex | None = None):
    if config.retrieval.row_ranker == "lexical":
        return LexicalRowRanker()
    if dense_index is None:
        dense_index = DenseTableIndex.load(config.retrieval.index_dir)
    return DenseRowRanker(dense_index.model)


def _retriever(config: TraceConfig, tables, device: str):
    dense = DenseTableIndex.load(config.retrieval.index_dir)
    dense.validate_tables(tables)
    reranker = CrossEncoderReranker.from_pretrained(
        config.retrieval.reranker_dir, device=device,
        max_length=config.retrieval.max_table_length,
    )
    return TwoStageRetriever(
        tables, dense, reranker, _row_ranker(config, dense),
        coarse_top_k=config.retrieval.coarse_top_k,
        table_top_k=config.retrieval.table_top_k,
        row_top_k=config.retrieval.row_top_k,
        batch_size=config.retrieval.reranker_batch_size,
        max_table_rows=config.retrieval.max_table_rows,
    )


def build_trees_command(config: TraceConfig, args) -> None:
    examples = load_claims(_split_file(config, args.split))
    records = []
    for example in examples:
        tree = decompose_claim(example.claim, config.amr.model)
        records.append({"id": example.id, "claim": example.claim, "tree": tree})
    destination = Path(args.output) if args.output else _tree_file(config, args.split)
    write_jsonl(destination, records)
    LOGGER.info("Wrote %d trees to %s", len(records), destination)


def build_index_command(config: TraceConfig, _args) -> None:
    tables = load_tables(config.data.table_file)
    index = DenseTableIndex(config.retrieval.sbert_model)
    index.build(tables)
    index.save(config.retrieval.index_dir)
    LOGGER.info("Indexed %d tables in %s", len(tables), config.retrieval.index_dir)


def train_reranker_command(config: TraceConfig, args) -> None:
    examples = load_claims(config.data.train_file)
    tables = load_tables(config.data.table_file)
    index = DenseTableIndex.load(config.retrieval.index_dir)
    index.validate_tables(tables)
    trees = load_tree_store(_tree_file(config, "train")) if (
        config.retriever_training.query_source == "nodes"
    ) else None
    pairs = mine_pairs(
        examples, tables, index, trees, config.retriever_training.query_source,
        config.retriever_training.hard_negatives,
        config.retriever_training.random_negatives,
        config.retrieval.coarse_top_k, config.seed,
    )
    pairs_file = Path(config.retrieval.reranker_dir).parent / f"{config.data.dataset}_pairs.jsonl"
    write_jsonl(pairs_file, (pair.to_dict() for pair in pairs))
    train_pairs, dev_pairs = split_by_qid(
        pairs, config.retriever_training.dev_ratio, config.seed,
    )
    metrics = train_reranker(
        train_pairs, dev_pairs, tables, config.retrieval.reranker_model,
        config.retrieval.reranker_dir, config.retrieval.max_table_length,
        config.retriever_training.batch_size, config.retriever_training.learning_rate,
        config.retriever_training.epochs, config.retriever_training.warmup_ratio,
        config.retriever_training.random_negative_weight,
        config.retrieval.max_table_rows, _device(args.device),
    )
    save_json(Path(config.retrieval.reranker_dir) / "metrics.json", metrics)
    save_json(
        Path(config.retrieval.reranker_dir) / "resolved_config.json", config.to_dict(),
    )


def prepare_evidence_command(config: TraceConfig, args) -> None:
    examples = load_claims(_split_file(config, args.split))
    tables = load_tables(config.data.table_file)
    trees = load_tree_store(_tree_file(config, args.split))
    retriever = (
        _retriever(config, tables, _device(args.device))
        if config.setting == "open" else None
    )
    row_ranker = retriever.row_ranker if retriever is not None else _row_ranker(config)
    records = build_evidence_records(
        examples, trees, tables, config.setting, row_ranker,
        config.retrieval.row_top_k, retriever,
    )
    destination = Path(args.output) if args.output else _evidence_file(config, args.split)
    write_jsonl(destination, records)
    LOGGER.info("Wrote evidence to %s", destination)


def _dataset(config: TraceConfig, split: str) -> TraceDataset:
    return TraceDataset(
        load_claims(_split_file(config, split)),
        load_tree_store(_tree_file(config, split)),
        load_evidence_store(_evidence_file(config, split)),
    )


def train_verifier_command(config: TraceConfig, args) -> None:
    model = TraceVerifier(
        config.verifier.backbone, config.verifier.max_length, config.verifier.num_heads,
        config.verifier.dropout, config.verifier.node_batch_size,
    ).to(_device(args.device))
    full_train = _dataset(config, "train")
    if len(full_train) < 2:
        raise ValueError("At least two training examples are required for a train/dev split")
    dev_size = min(
        len(full_train) - 1,
        max(1, round(len(full_train) * config.verifier.dev_ratio)),
    )
    train_size = len(full_train) - dev_size
    generator = torch.Generator().manual_seed(config.seed)
    train_data, dev_data = random_split(full_train, [train_size, dev_size], generator=generator)
    result = train_verifier(
        model, train_data, dev_data,
        config.verifier.output_dir, config.verifier.epochs, config.verifier.batch_size,
        config.verifier.learning_rate, config.verifier.warmup_ratio,
        config.verifier.weight_decay,
        config.verifier.gradient_accumulation_steps, config.verifier.max_grad_norm,
        config.verifier.mixed_precision, config.verifier.num_workers,
    )
    save_json(
        Path(config.verifier.output_dir) / "resolved_config.json", config.to_dict(),
    )
    LOGGER.info("Best dev macro-F1: %.4f", result["best_dev_macro_f1"])


def evaluate_command(config: TraceConfig, args) -> None:
    model = TraceVerifier.from_checkpoint(config.verifier.output_dir).to(_device(args.device))
    loader = DataLoader(
        _dataset(config, args.split), batch_size=config.verifier.batch_size,
        shuffle=False, collate_fn=collate_items, num_workers=config.verifier.num_workers,
    )
    metrics, predictions = evaluate(model, loader)
    for record in predictions:
        record["gold"] = ID_TO_LABEL[record["gold"]]
        record["prediction"] = ID_TO_LABEL[record["prediction"]]
    destination = Path(
        args.output
        or (Path(config.verifier.output_dir) / f"{args.split}_predictions.jsonl")
    )
    write_jsonl(destination, predictions)
    save_json(destination.with_suffix(".metrics.json"), metrics)
    print(json.dumps(metrics, indent=2))


def validate_data_command(config: TraceConfig, _args) -> None:
    tables = load_tables(config.data.table_file)
    summary = {"tables": len(tables), "splits": {}}
    for split in ("train", "test"):
        examples = load_claims(_split_file(config, split))
        missing = sorted({uid for ex in examples for uid in ex.table_uids if uid not in tables})
        summary["splits"][split] = {
            "examples": len(examples),
            "supported": sum(ex.label == "SUPPORTED" for ex in examples),
            "refuted": sum(ex.label == "REFUTED" for ex in examples),
            "missing_table_uids": missing,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(item["missing_table_uids"] for item in summary["splits"].values()):
        raise SystemExit(1)


def predict_command(config: TraceConfig, args) -> None:
    tables = load_tables(config.data.table_file)
    tree = decompose_claim(args.claim, config.amr.model)
    example = ClaimExample(
        id="prediction", claim=args.claim, table_uids=tuple(args.table_uid or ()),
        label="SUPPORTED",
    )
    if config.setting == "gold" and not example.table_uids:
        raise ValueError("Gold prediction requires at least one --table-uid")
    retriever = (
        _retriever(config, tables, _device(args.device))
        if config.setting == "open" else None
    )
    row_ranker = retriever.row_ranker if retriever is not None else _row_ranker(config)
    evidence = next(build_evidence_records(
        [example], {example.id: tree}, tables, config.setting, row_ranker,
        config.retrieval.row_top_k, retriever,
    ))
    model = (
        TraceVerifier.from_checkpoint(config.verifier.output_dir)
        .to(_device(args.device))
        .eval()
    )
    with torch.inference_mode():
        logits = model([{"tree": tree, "evidence": evidence}])
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    prediction = int(torch.tensor(probabilities).argmax())
    print(json.dumps({
        "claim": args.claim, "prediction": ID_TO_LABEL[prediction],
        "probabilities": {ID_TO_LABEL[index]: value for index, value in enumerate(probabilities)},
        "tree": tree, "evidence": evidence,
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace", description="TRACE fact-checking pipeline")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-data")
    for name in ("build-trees", "prepare-evidence"):
        command = commands.add_parser(name)
        command.add_argument("--split", choices=("train", "test"), default="train")
        command.add_argument("--output")
        if name == "prepare-evidence":
            command.add_argument("--device")
    commands.add_parser("build-index")
    reranker = commands.add_parser("train-reranker")
    reranker.add_argument("--device")
    verifier = commands.add_parser("train-verifier")
    verifier.add_argument("--device")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--split", choices=("train", "test"), default="test")
    evaluation.add_argument("--output")
    evaluation.add_argument("--device")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--claim", required=True)
    prediction.add_argument("--table-uid", action="append")
    prediction.add_argument("--device")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    set_seed(config.seed)
    commands = {
        "build-trees": build_trees_command,
        "validate-data": validate_data_command,
        "build-index": build_index_command,
        "train-reranker": train_reranker_command,
        "prepare-evidence": prepare_evidence_command,
        "train-verifier": train_verifier_command,
        "evaluate": evaluate_command,
        "predict": predict_command,
    }
    commands[args.command](config, args)


if __name__ == "__main__":
    main()
