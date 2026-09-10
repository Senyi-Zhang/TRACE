# Implementation decisions

This document records how the public implementation resolves differences
between the original research scripts and the paper-level description.

| Component | Released behavior | Rationale |
| --- | --- | --- |
| AMR tree | Proposition/operator projection from the original implementation | Preserves the trees used by the project code |
| Table retrieval | Bare SentenceBERT cosine retrieval over the full table pool | Original coarse-recall stage |
| Table reranking | Trained DeBERTa cross-encoder over marked query-table sequences | Original fine-ranking stage |
| Row reranking | Rarity-weighted lexical scorer by default; dense SBERT optional | Legacy default with a paper-aligned alternative |
| Node encoding | All selected rows are concatenated into one evidence context | Matches Section 3.4 and Appendix C.2 |
| Child aggregation | PMA over composed child representations | Matches Equation 5 |
| Fusion | `gamma = 1 + tanh(delta_gamma)` and `gamma * parent + beta` | Matches Equations 6-7 |
| Traversal | Deterministic post-order from leaves to root | Matches Section 3.5 |
| Supervision | Cross-entropy only at the root | Matches Equations 8-9 |

## Stable node identity

The AMR tree format itself is unchanged. Downstream stages address nodes with a
stable structural path: the root is `n0`; its children are `n0.0`, `n0.1`, and
so on. This avoids adding serialization-only IDs to the tree and makes cached
evidence easy to audit.

## Retriever supervision

The original reranker script mines positives and negatives from complete
claims. The release supports that behavior with `query_source: claim`. The
default is `query_source: nodes`, because TRACE performs retrieval per semantic
node. In that mode, every unique node query for an example is paired with its
annotated table(s). This is weak supervision: the datasets identify gold tables
for the claim, not an explicit table for each AMR node.

## Gold evidence

For the Gold setting, every node sees candidate rows from all gold tables for
the example. The configured row ranker selects the top rows separately for each
node query. This preserves node-specific evidence while using only annotated
tables.

## Development split

OTT-FV and Hybrid-FV contain train and test files. The code therefore creates a
deterministic example-level development subset from the training file. The test
set is not used for checkpoint selection. The split ratio and seed are stored
in the resolved run configuration.

## Caching boundary

Semantic trees and per-node retrieval results are materialized as JSONL before
verifier training. This provides three practical guarantees:

1. verifier comparisons use identical evidence;
2. retrieval misses and AMR failures can be inspected independently;
3. expensive table retrieval is not repeated every epoch.

