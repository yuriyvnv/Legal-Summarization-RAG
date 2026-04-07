# Raw-Chunk Benchmark Baseline

This directory contains our raw-chunk baseline pipeline, used to compare our proposed summary-based method against a benchmark-style retrieval setup. In this branch of the workflow, we index documents as original chunks rather than validated summaries.

## Conceptual Role

The scripts here correspond to the "benchmark" side of our paper:

1. build a single Chroma collection from raw chunk text
2. evaluate retrieval on LEGALBENCH-RAG queries
3. export per-dataset results and aggregate metric tables

Unlike the summary-based evaluation in [`../baseline_V1/evaluation/`](../baseline_V1/evaluation/), these scripts query a single `documents` collection and filter results by the benchmark's target document filename.

## Files

| File | Responsibility |
| --- | --- |
| [`create_base.py`](create_base.py) | Builds the raw-chunk Chroma collection from [`../data/original_chunked/processed_documents_benchmark.json`](../data/original_chunked/processed_documents_benchmark.json) using BGE-M3. |
| [`eval_base.py`](eval_base.py) | Evaluates the raw-chunk baseline with OpenAI `text-embedding-3-large`. |
| [`eval_base_bge.py`](eval_base_bge.py) | Evaluates the raw-chunk baseline with BGE-M3. |
| [`eval_base_multiqa.py`](eval_base_multiqa.py) | Evaluates the raw-chunk baseline with `sentence-transformers/multi-qa-mpnet-base-dot-v1`. |
| [`eval_base_all_mini.py`](eval_base_all_mini.py) | Evaluates the raw-chunk baseline with `sentence-transformers/all-MiniLM-L6-v2`. |
| [`eval_base_optimized.py`](eval_base_optimized.py) | Optimized OpenAI baseline evaluation variant with batching and parallel processing. |

## Output Artifacts

| Path pattern | Responsibility |
| --- | --- |
| `evaluation_results_benchmark_openai_multi_k/` | Saved results and metric tables for the OpenAI raw-chunk baseline. |
| `evaluation_results_benchmark_bge_multi_k/` | Saved results and metric tables for the BGE-M3 raw-chunk baseline. |
| `evaluation_results_benchmark_multiqa_multi_k/` | Saved results and metric tables for the MultiQA raw-chunk baseline. |
| `evaluation_results_benchmark_allmini_multi_k/` | Saved results and metric tables for the all-MiniLM raw-chunk baseline. |
| `intermediate_*.json` | Checkpointed per-dataset results written while evaluation is running. |

## Retrieval Protocol

These scripts follow the same broad pattern:

1. load one benchmark dataset JSON from [`../data/benchmarks/`](../data/benchmarks/)
2. extract the query and the target document name
3. query the raw-chunk `documents` collection with a `where={"document": doc_name}` filter
4. mark a query as correct if any gold answer appears as a substring in any retrieved chunk
5. aggregate precision, recall, F1, and accuracy for `k = [1, 2, 4, 8]`

## Why This Directory Matters

This directory is the main reference point for our paper's benchmark comparison. It keeps the retrieval target as raw chunk text so the effect of validated summaries can be compared against a more conventional chunk-indexing approach.

## Notes

- The persistence directories are hardcoded under `data/benchmarks/`.
- These are standalone experiment runners that write local JSON and CSV/TXT outputs directly into this folder.
- The "optimized" script is not a different method; it is a performance-oriented execution variant of the same baseline idea.
