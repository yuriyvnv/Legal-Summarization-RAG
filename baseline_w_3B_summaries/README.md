# Batched 3B Summarization Runs

This directory contains alternative batched summarization scripts and their saved outputs for selected datasets. These runs are related to the paper's chunk-summarization stage, but they are organized as standalone experiment scripts rather than the repository's main summarization pipeline.
The naming of the files represent only for the cuad and maud datasets, however the user can just change the position of the dataset to privacy_qa or other to summarize the chunks.

This directory is configured to use `meta-llama/Llama-3.2-3B-Instruct` for summary generation.

## Files

| File | Responsibility |
| --- | --- |
| [`summarize_docs_cuad_baseline_v2.py`](summarize_docs_cuad_baseline_v2.py) | Batched chunk summarization script for the CUAD experiment variant using `meta-llama/Llama-3.2-3B-Instruct`. |
| [`summarize_docs_maud_baseline_v2.py`](summarize_docs_maud_baseline_v2.py) | Batched chunk summarization script for the MAUD experiment variant using `meta-llama/Llama-3.2-3B-Instruct`. |
| [`summarized_docs_cuadV2.json`](summarized_docs_cuadV2.json) | Saved CUAD summary outputs from the batched v2 summarization run. |
| [`summarized_docs_maudV2.json`](summarized_docs_maudV2.json) | Saved MAUD summary outputs from the batched v2 summarization run. |

## How These Scripts Work

The v2 scripts differ from the root [`../summarize_docs.py`](../summarize_docs.py) entry point mainly in execution style:

- they batch many chunks together for generation
- they checkpoint progress as batches complete
- they are written as dataset-specific experiment scripts rather than a general reusable command

## Relationship to the Main Pipeline

- Use [`../summarize_docs.py`](../summarize_docs.py) as the simpler, root-level summarization reference.
- Use this directory to understand the author's alternative high-throughput summarization runs for specific datasets.
- The JSON outputs here are artifacts from those runs, not canonical benchmark definitions.

## Notes

- These scripts include hardcoded local paths, device settings, and model-access assumptions.
- Review them carefully before reusing them on another machine.
- Before making the repository public, sanitize this directory for embedded credentials or machine-specific settings.
