# Utils Directory

This directory contains the support scripts that turn raw chunk summaries into validated artifacts suitable for retrieval. These scripts are the operational glue of the paper's self-improvement loop.

## Main Responsibilities

| Path | Responsibility |
| --- | --- |
| [`summary_verification.py`](summary_verification.py) | Checks which documents or chunks are missing summaries and generates verification reports. |
| [`generate_missing_summaries.py`](generate_missing_summaries.py) | Regenerates missing chunk summaries using a smaller Llama model and updates summary files in place. |
| [`generate_doc_summary/final_summary_generation.py`](generate_doc_summary/final_summary_generation.py) | Produces document-level `general_summary` fields from corrected chunk summaries. |
| [`quality_evaluation/`](quality_evaluation/) | Scores chunk summaries, analyzes low-quality cases, regenerates weak summaries, and merges corrections. |
| [`verification_results.json`](verification_results.json) | Saved report from summary verification. |
| [`missing_summaries.json`](missing_summaries.json) | Machine-readable list of missing chunk summaries for recovery. |

## Suggested Workflow

1. Run [`summary_verification.py`](summary_verification.py) to determine which datasets or documents are incomplete.
2. Use [`generate_missing_summaries.py`](generate_missing_summaries.py) to fill missing chunk summaries.
3. Run the scripts under [`quality_evaluation/`](quality_evaluation/) to score summaries and identify low-quality outputs.
4. Apply regenerated summaries and merged corrections.
5. Use [`generate_doc_summary/final_summary_generation.py`](generate_doc_summary/final_summary_generation.py) to create document-level summaries for hierarchical retrieval.

## Model Assignments

- Summary generation scripts in this directory use `meta-llama/Llama-3.2-3B-Instruct`.
- This includes [`generate_missing_summaries.py`](generate_missing_summaries.py) and [`generate_doc_summary/final_summary_generation.py`](generate_doc_summary/final_summary_generation.py).
- Summary evaluation and regeneration under [`quality_evaluation/`](quality_evaluation/) use `meta-llama/Llama-3.1-8B-Instruct`.

## File-Level Notes

### `summary_verification.py`

- Compares `processed_documents.json` against per-dataset summary files.
- Produces two outputs:
  - `utils/verification_results.json`
  - `utils/missing_summaries.json`
- This is the main integrity check for the chunk-summary layer.

### `generate_missing_summaries.py`

- Loads `missing_summaries.json` and the original processed documents.
- Uses `meta-llama/Llama-3.2-3B-Instruct` to regenerate only the missing chunks.
- Writes updated summaries back to the expected dataset JSON files.

### `generate_doc_summary/final_summary_generation.py`

- Reads the combined corrected chunk summaries.
- Uses `meta-llama/Llama-3.2-3B-Instruct` to generate document-level summaries.
- Builds a longer `general_summary` per document for RAG-style relevance matching.
- Preserves the underlying chunk list, so downstream Chroma builders can index both document-level and chunk-level representations.

## Relationship to the Paper

The paper describes an iterative quality-controlled summary workflow. In this repository, that workflow is not implemented as one monolithic script. Instead, it is distributed across:

- verification and gap detection
- summary evaluation
- low-score analysis
- regeneration / correction
- merge and final document-summary generation

The scripts in this directory therefore represent the practical implementation of the "self-improvement" stage discussed in the paper.

## Caveats

- Most scripts assume hardcoded paths under `/home/yperezhohin/Legal-Summarization-RAG/...`.
- GPU selection is often set inside the scripts with `CUDA_VISIBLE_DEVICES`.
- Several scripts write large JSON artifacts directly, so they should be treated as experiment pipeline steps rather than reusable library functions.
