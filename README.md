# TRACE

Official implementation of **TRACE: Tree-Structured Reasoning for Multi-Table
Multi-Hop Fact-Checking** (EMNLP 2026).

TRACE decomposes a claim into an AMR-guided semantic tree, retrieves table and
row evidence for each tree node, encodes every node with its local evidence,
and composes the representations bottom-up with PMA and FiLM. The root
representation is used to predict `SUPPORTED` or `REFUTED`.

## What is implemented

- AMR-guided proposition/operator trees using `transition-amr-parser`.
- Open evidence: SentenceBERT table retrieval, cross-encoder table reranking,
  and row reranking.
- Gold evidence: node-specific row selection over the annotated gold tables.
- Evidence-aware node encoding with the exact `[Query]`, `[Evidence]`, and
  `[RowN]` serialization described in the paper.
- Post-order tree composition with PMA and FiLM.
- Root-level binary classification and end-to-end cross-entropy training.
- Deterministic train/dev splitting, data validation, cached trees/evidence,
  checkpointing, and standalone evaluation/prediction commands.



## Repository layout

```text
TRACE/
├── configs/                  # Dataset/setting-specific YAML files
├── src/trace_fc/
│   ├── decomposition/        # PENMAN reader and AMR-to-tree projection
│   ├── models/               # PMA, FiLM, and the TRACE verifier
│   ├── retrieval/            # Dense index, reranker, mining, row ranking
│   ├── cli.py                # Reproducible command-line stages
│   ├── config.py             # Typed configuration
│   ├── data.py               # Dataset and table schemas
│   ├── evidence.py           # Gold/Open evidence preparation
│   └── training.py           # Verifier training and evaluation
└── tests/                    # Offline unit and end-to-end smoke tests
```

Generated artifacts are kept outside the package under `artifacts/` and are
ignored by Git.

## Installation

Python 3.10 or later is required. We used NVIDIA A100 GPUs for the paper
experiments.

```bash
git clone https://github.com/Senyi-Zhang/TRACE.git
cd TRACE
python -m venv .venv
source .venv/bin/activate
pip install -e ".[faiss]"
```

Install IBM's `transition-amr-parser` and download the
`AMR3-structbart-L` checkpoint by following its official installation
instructions. It is needed only by `build-trees` and end-to-end `predict`;
precomputed tree JSONL files can be used without the parser.

For development checks:

```bash
pip install -e ".[faiss,dev]"
pytest
ruff check src tests
```

## Data layout

Place the released data next to `configs/`:

```text
TRACE/
├── OTT-FV/
│   ├── train.jsonl
│   ├── test.jsonl
│   └── tables.json
└── Hybrid-FV/
    ├── train.jsonl
    ├── test.jsonl
    └── tables.json
```

If the table corpora have different filenames, change `data.table_file` in the
corresponding YAML. All relative paths are resolved relative to the YAML file,
not the current shell directory.

Each fact-checking example is a JSON object on one line:

```json
{"id": 192, "claim": "...", "table_uid_1": "table_1", "table_uid_2": "table_2", "category": "CONJUNCTIVE", "label": "REFUTED"}
```

The loader strips surrounding whitespace from IDs, categories, and labels. The
supported labels are exactly `SUPPORTED` and `REFUTED`. Table corpora are JSON
objects keyed by UID:

```json
{
  "table_1": {
    "url": "https://en.wikipedia.org/wiki/...",
    "title": "Example title",
    "header": ["Year", "Person", "Role"],
    "data": [["1994", "Nelson Mandela", "President"]],
    "section_title": "",
    "section_text": "",
    "uid": ""
  }
}
```

Validate IDs and labels before running an experiment:

```bash
trace --config configs/ott_fv_open.yaml validate-data
```

## Reproducing an Open-evidence experiment

The stages are explicit so expensive AMR parses and retrieval results are
cached and inspectable.

### 1. Build semantic trees

```bash
trace --config configs/ott_fv_open.yaml build-trees --split train
trace --config configs/ott_fv_open.yaml build-trees --split test
```

Each output line contains the example ID, original claim, PENMAN AMR, projected
tree, entity table, and atomic propositions. Tree nodes receive stable path IDs
(`n0`, `n0.0`, `n0.1`, ...) when evidence is prepared.

### 2. Build the SentenceBERT coarse index

```bash
trace --config configs/ott_fv_open.yaml build-index
```

The normalized table embeddings, table-ID order, metadata fingerprint, and an
optional FAISS inner-product index are saved together. NumPy search is used if
FAISS is unavailable.

### 3. Mine pairs and train the table reranker

```bash
trace --config configs/ott_fv_open.yaml train-reranker
```

By default, every unique semantic-node query is paired with its example's gold
tables as positives. Hard negatives come from the SentenceBERT candidate list;
additional random negatives are down-weighted. The split is made by query ID,
so documents for a query cannot leak between training and validation.

To reproduce the older claim-level mining behavior, set:

```yaml
retriever_training:
  query_source: claim
```

### 4. Cache per-node evidence

```bash
trace --config configs/ott_fv_open.yaml prepare-evidence --split train
trace --config configs/ott_fv_open.yaml prepare-evidence --split test
```

For each node, TRACE retrieves 200 tables with SentenceBERT, reranks them with
the cross-encoder, retains 10 tables, and then retains 10 rows. These values are
configurable.

### 5. Train and evaluate the verifier

```bash
trace --config configs/ott_fv_open.yaml train-verifier
trace --config configs/ott_fv_open.yaml evaluate --split test
```

`train-verifier` deterministically reserves 10% of the training examples for
model selection. It never uses the test set to select a checkpoint. Evaluation
writes both a prediction JSONL and a metrics JSON file.

Repeat with `configs/hybrid_fv_open.yaml` for Hybrid-FV.

## Gold-evidence experiments

Gold experiments do not need a table index or a trained reranker. Trees are
still node-specific, and the row ranker selects evidence for each node from the
annotated gold tables.

```bash
trace --config configs/ott_fv_gold.yaml build-trees --split train
trace --config configs/ott_fv_gold.yaml build-trees --split test
trace --config configs/ott_fv_gold.yaml prepare-evidence --split train
trace --config configs/ott_fv_gold.yaml prepare-evidence --split test
trace --config configs/ott_fv_gold.yaml train-verifier
trace --config configs/ott_fv_gold.yaml evaluate --split test
```

## Prediction

Open-evidence prediction uses the saved index, reranker, and verifier:

```bash
trace --config configs/ott_fv_open.yaml predict \
  --claim "Nelson Mandela became president of South Africa in 1994."
```

Gold-evidence prediction additionally requires one or more known table IDs:

```bash
trace --config configs/ott_fv_gold.yaml predict \
  --claim "Nelson Mandela became president of South Africa in 1994." \
  --table-uid south_africa_presidents_by_year
```

The command prints the verdict probabilities together with the projected tree
and selected evidence for inspection.

## Core model

For each tree node `v`, the node encoder receives one sequence:

```text
[Query] <node query>
[Evidence]
[Row1] [Title] <table title> <header>: <cell> ...
...
```

The backbone produces the evidence-aware representation `h_v`. In post-order:

- A leaf uses `h_tilde_v = h_v`.
- A non-leaf pools its composed children with PMA.
- FiLM predicts `delta_gamma` and `beta` from the pooled child representation,
  applies `gamma = 1 + tanh(delta_gamma)`, and computes
  `h_tilde_v = gamma * h_v + beta`.
- A linear classifier over the root representation predicts the final label.

All encoder, PMA, FiLM, and classifier parameters are optimized jointly from
the root-level cross-entropy loss.

## Default hyperparameters

| Component | Setting |
| --- | --- |
| Coarse retriever | `all-MiniLM-L6-v2` |
| Table reranker | `microsoft/deberta-v3-base` |
| Coarse/table/row top-k | 200 / 10 / 10 |
| Verifier | `microsoft/deberta-v3-base` |
| Verifier learning rate | 2e-5 |
| Epochs | 3 |
| Claim batch size | 16 |
| Seed | 42 |
| Maximum verifier length | 512 |

The full configuration, including negative mining and mixed precision, is in
`configs/`.

To execute all stages in order, use:

```bash
bash scripts/run_pipeline.sh configs/ott_fv_open.yaml
```


## Notes on reproducibility

- Evidence is cached before verifier training. This keeps all runs on the same
  retrieval results and makes retrieval errors auditable.
- Tree and evidence records are keyed by normalized string IDs.
- Root classification is the only supervised verifier objective; intermediate
  nodes receive gradients through bottom-up composition.
- The row scorer is deterministic. Equal-score rows preserve corpus/table
  order through Python's stable sort.
- The saved verifier checkpoint contains a self-contained backbone, tokenizer,
  PMA/FiLM/classifier weights, and training metadata.

## Citation

```bibtex
@inproceedings{zhang2026trace,
  title     = {{TRACE}: Tree-Structured Reasoning for Multi-Table Multi-Hop Fact-Checking},
  author    = {Zhang, Senyi and Lee, Dongwon and Zhang, Delvin Ce},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## Intended use

The code and datasets are intended for research on table-based fact-checking.
They should not be used as fully reliable fact-checking systems in high-stakes
settings without human oversight. Please follow the licenses and terms of the
source datasets, table corpora, pretrained models, and AMR parser.
