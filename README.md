# TRACE: Tree-Structured Reasoning for Multi-Table Multi-Hop Fact-Checking

This repository contains the materials accompanying our EMNLP 2026 paper, **“TRACE: Tree-Structured Reasoning for Multi-Table Multi-Hop Fact-Checking.”** It provides the source code, datasets, and experimental resources required to reproduce the proposed method and results.

## Abstract

Multi-table fact-checking verifies claims whose evidence is distributed across multiple tables and connected through multi-hop reasoning. This setting is challenging because claims may contain chained or nested dependencies, table retrieval is noisy, and retrieved evidence must be selected and composed before a final verdict can be made. We introduce two multi-table fact verification testbeds, OTT-FV and Hybrid-FV, and propose TRACE, a tree-structured framework that connects claim decomposition, evidence retrieval, and evidence composition. TRACE represents each claim as a dependency-aware semantic tree, retrieves locally relevant table evidence for each node, and composes evidence-aware node representations bottom-up to produce the final verdict. Experiments on TabFact, OTT-FV, and Hybrid-FV show that TRACE consistently improves over strong baselines across multiple backbones under both gold- and open-evidence settings. Further analyses show that tree construction, node-level retrieval, and bottom-up evidence composition all contribute to multi-table verification.

## Repository Structure

This material is organized as follows:

* `code/`: Source code for the proposed method and experiments.
* `datasets/`: Full datasets used in this work, including both **OTT-FV** and **Hybrid-FV**.

## Dataset Note

The `datasets/` directory contains the complete fact-verification datasets for **OTT-FV** and **Hybrid-FV**.

## Additional Note

Please refer to the paper for detailed descriptions of the task, dataset construction process, proposed method, and experimental settings.
