# Summary-Based Vector DB Construction

This directory contains the scripts we use to build the Chroma vector stores for our summary-based retrieval experiments. These scripts operate on corrected summary artifacts rather than raw benchmark chunks.

## Shared Design

The DB builders in this directory generally create two Chroma collections:

- `documents`: document-level summaries (`general_summary`)
- `chunks`: chunk-level summaries, with the original chunk text stored in metadata

This structure supports the hierarchical retrieval pattern used in the summary-based evaluation scripts:

1. retrieve the most relevant document summary
2. retrieve the most relevant chunk summaries within that document

## Files

| File | Responsibility |
| --- | --- |
| [`create_db_baseline.py`](create_db_baseline.py) | Builds the summary-based hierarchical store using the fine-tuned legal MultiQA embedding model `yuriyvnv/legal-multi-qa-mpnet-base-cos`. |
| [`create_db_baseline_finetuned_multiqa.py`](create_db_baseline_finetuned_multiqa.py) | Optimized fine-tuned MultiQA builder with cached embeddings, retry logic, and batched processing. |
| [`create_db_baseline_gtr5xxl.py`](create_db_baseline_gtr5xxl.py) | Builds the same hierarchical store using `sentence-transformers/gtr-t5-xxl`. |
| [`memory_size_checker.py`](memory_size_checker.py) | Computes the size of a Chroma persistence directory on disk. |
| [`verify_versions.py`](verify_versions.py) | Prints the local SQLite version for environment debugging. |

## Inputs

The main DB builders expect:

- [`../data/combined_summarized_corrected/final_summaries_striped.json`](../data/combined_summarized_corrected/final_summaries_striped.json) for chunk-level and document-level summary data
- a writable Chroma persistence directory under `data/benchmarks/`
- the necessary embedding model access and GPU environment

## Output Stores

| Script | Default output store |
| --- | --- |
| [`create_db_baseline.py`](create_db_baseline.py) | `data/benchmarks/fine_tuned_Embedding_multiqa_ours` |
| [`create_db_baseline_finetuned_multiqa.py`](create_db_baseline_finetuned_multiqa.py) | `data/benchmarks/fine_tuned_Embedding_multiqa_ours` |
| [`create_db_baseline_gtr5xxl.py`](create_db_baseline_gtr5xxl.py) | `data/benchmarks/baseline_ours_gtr5_xxl` |

## Metadata Conventions

The chunk collection typically stores:

- `dataset`
- `doc_id`
- `chunk_id`
- `original_text`

This is important because the evaluation scripts in [`../baseline_V1/evaluation/`](../baseline_V1/evaluation/) rely on `doc_id` to restrict chunk retrieval after document retrieval.

## Notes

- These are experimental builders, not idempotent package commands. Some of them delete and recreate collections on startup.
- Paths and GPU settings are hardcoded for our original environment.
- The fine-tuned MultiQA and GTR variants are best understood as different embedding backends applied to the same validated-summary retrieval design.
