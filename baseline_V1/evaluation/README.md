# Summary-Based Evaluation

This directory contains the evaluation scripts for our proposed retrieval setup: hierarchical retrieval over validated summaries. These scripts are closer to our paper's main method than the raw benchmark baseline in [`../../baseline_from_benchmark/`](../../baseline_from_benchmark/).

## Shared Retrieval Pattern

Most scripts in this directory use the same two-stage retrieval strategy:

1. split the benchmark query into a document-level query and a chunk-level query using `;`
2. query the `documents` collection to identify the most relevant document summary
3. query the `chunks` collection restricted by `where={"doc_id": ...}`
4. evaluate success by checking whether any gold answer appears in the retrieved `original_text`

This matches our hierarchical summary-based representation design.

## Evaluation Scripts

| File | Responsibility |
| --- | --- |
| [`evaluation_script_finetuned_ours.py`](evaluation_script_finetuned_ours.py) | Evaluates the proposed method using the fine-tuned legal MultiQA embedding model and the `fine_tuned_Embedding_multiqa_ours` Chroma store. |
| [`evaluation_script_openai_multiK.py`](evaluation_script_openai_multiK.py) | Evaluates the same summary-based retrieval setup with OpenAI `text-embedding-3-large`. |
| [`evaluation_script_bge_m3_multiK.py`](evaluation_script_bge_m3_multiK.py) | Evaluates the summary-based retrieval setup with BGE-M3. |
| [`evaluation_script_multiqa_multiK.py`](evaluation_script_multiqa_multiK.py) | Evaluates the summary-based retrieval setup with `sentence-transformers/multi-qa-mpnet-base-dot-v1`. |
| [`evaluation_script_all_mini_multiK.py`](evaluation_script_all_mini_multiK.py) | Evaluates the summary-based retrieval setup with `sentence-transformers/all-MiniLM-L6-v2`. |
| [`evaluation_script_gtr5xxl_multiK.py`](evaluation_script_gtr5xxl_multiK.py) | Evaluates the summary-based retrieval setup with `sentence-transformers/gtr-t5-xxl`, using an optimized embedding cache and safer intermediate writes. |

## Result Folders

| Path | Responsibility |
| --- | --- |
| [`evaluation_results_ours_finetuned_multiK/`](evaluation_results_ours_finetuned_multiK/) | Metrics and per-dataset results for the fine-tuned legal MultiQA model. |
| [`evaluation_results_openai_multi_k/`](evaluation_results_openai_multi_k/) | Metrics and per-dataset results for the OpenAI embedding variant. |
| [`evaluation_results_bge_m3_multiK/`](evaluation_results_bge_m3_multiK/) | Metrics and per-dataset results for the BGE-M3 variant. |
| [`evaluation_results_multiqa_multiK/`](evaluation_results_multiqa_multiK/) | Metrics and per-dataset results for the MultiQA variant. |
| [`evaluation_results_all_mini_multiK/`](evaluation_results_all_mini_multiK/) | Metrics and per-dataset results for the all-MiniLM variant. |
| [`evaluation_results_gtr5_xxl_multiK/`](evaluation_results_gtr5_xxl_multiK/) | Metrics and per-dataset results for the GTR-T5-XXL variant. |
| [`intermediate_cuad.json`](intermediate_cuad.json), [`intermediate_maud.json`](intermediate_maud.json), [`intermediate_contractnli.json`](intermediate_contractnli.json), [`intermediate_privacy_qa.json`](intermediate_privacy_qa.json) | Checkpointed intermediate results written while evaluation is running. |

## Metrics

All evaluation scripts aggregate the same top-k metrics:

- precision
- recall
- F1
- accuracy

for:

- `k = 1`
- `k = 2`
- `k = 4`
- `k = 8`

This makes the result folders directly comparable across embedding models.

## How This Directory Relates to the Paper

This directory is the closest code representation of our paper's main comparison table across embedding models. The important distinction from the raw benchmark baseline is that retrieval happens over validated summary representations stored in separate `documents` and `chunks` collections.

## Notes

- The scripts here assume that the relevant Chroma stores have already been built by the scripts in [`../../create_db_baseline/`](../../create_db_baseline/).
- The OpenAI, BGE, MultiQA, all-MiniLM, and GTR scripts differ mostly by embedding backend and the associated persistence directory.
- Several scripts use aggressive batching and concurrency tuned for our original hardware environment.
