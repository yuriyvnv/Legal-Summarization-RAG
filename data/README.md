# Data Directory

This directory holds the repository's benchmark inputs, preprocessed corpora, summary artifacts, and small analysis helpers. Most large files here are Git LFS-managed.

## Layout

| Path | Responsibility |
| --- | --- |
| [`benchmarks/`](benchmarks/) | LEGALBENCH-RAG dataset files and small utilities for inspecting Chroma collections. |
| [`original_chunked/`](original_chunked/) | Pre-chunked legal corpora used as inputs to summarization and baseline indexing. |
| [`summarized_data_raw/`](summarized_data_raw/) | Raw chunk-level summary outputs before correction and consolidation. |
| [`combined_summarized_corrected/`](combined_summarized_corrected/) | Corrected, merged, and final summary artifacts plus analysis helpers. |

## Important Files

| File | Responsibility |
| --- | --- |
| [`benchmarks/cuad.json`](benchmarks/cuad.json) | Benchmark evaluation file for CUAD queries and gold snippets. |
| [`benchmarks/maud.json`](benchmarks/maud.json) | Benchmark evaluation file for MAUD queries and gold snippets. |
| [`benchmarks/contractnli.json`](benchmarks/contractnli.json) | Benchmark evaluation file for ContractNLI queries and gold snippets. |
| [`benchmarks/privacy_qa.json`](benchmarks/privacy_qa.json) | Benchmark evaluation file for Privacy-QA queries and gold snippets. |
| [`benchmarks/check_collections.py`](benchmarks/check_collections.py) | Inspects stored embedding dimensions and basic Chroma collection statistics. |
| [`original_chunked/processed_documents.json`](original_chunked/processed_documents.json) | Main chunked corpus used by summarization and quality-control scripts. |
| [`original_chunked/processed_documents_benchmark.json`](original_chunked/processed_documents_benchmark.json) | Chunked corpus used by the raw benchmark baseline indexer. |
| [`summarized_data_raw/summarized_docs_cuad.json`](summarized_data_raw/summarized_docs_cuad.json) | Raw chunk summaries for CUAD. |
| [`summarized_data_raw/summarized_docs_maud.json`](summarized_data_raw/summarized_docs_maud.json) | Raw chunk summaries for MAUD. |
| [`summarized_data_raw/summarized_docs_contractnli.json`](summarized_data_raw/summarized_docs_contractnli.json) | Raw chunk summaries for ContractNLI. |
| [`summarized_data_raw/summarized_docs_privacy_qa.json`](summarized_data_raw/summarized_docs_privacy_qa.json) | Raw chunk summaries for Privacy-QA. |
| [`combined_summarized_corrected/combined_updated_summaries.json`](combined_summarized_corrected/combined_updated_summaries.json) | Merged chunk-summary artifact after quality corrections are applied. |
| [`combined_summarized_corrected/final_summaries_striped.json`](combined_summarized_corrected/final_summaries_striped.json) | Final chunk-summary artifact used for summary-based vector DB creation. |
| [`combined_summarized_corrected/final_document_summaries.json`](combined_summarized_corrected/final_document_summaries.json) | Document-level summaries paired with chunk summaries for hierarchical retrieval. |

## Analysis Helpers

| File | Responsibility |
| --- | --- |
| [`combined_summarized_corrected/strip_down.py`](combined_summarized_corrected/strip_down.py) | Produces a reduced summary artifact for downstream indexing. |
| [`combined_summarized_corrected/counter_analysis.py`](combined_summarized_corrected/counter_analysis.py) | Counts and summarizes dataset-level properties of the corrected outputs. |
| [`combined_summarized_corrected/violin_plot.py`](combined_summarized_corrected/violin_plot.py) | Creates plots from summary statistics. |

## How These Files Fit the Paper

1. The benchmark JSON files in [`benchmarks/`](benchmarks/) define the evaluation tasks.
2. The chunked corpora in [`original_chunked/`](original_chunked/) serve as the repository's starting point for representation generation.
3. The raw summary files in [`summarized_data_raw/`](summarized_data_raw/) are intermediate outputs from chunk summarization.
4. The corrected and merged files in [`combined_summarized_corrected/`](combined_summarized_corrected/) correspond to the validated representation layer used for summary-based retrieval experiments.

## Notes

- Treat the large JSON files in this directory as experiment artifacts rather than hand-maintained source files.
- Many scripts outside this directory expect these files to exist at hardcoded absolute paths; update those paths before rerunning the pipeline on a different machine.
- Because most of these files are large and LFS-managed, cloning the repo without `git lfs pull` can leave pointer files instead of real data.
