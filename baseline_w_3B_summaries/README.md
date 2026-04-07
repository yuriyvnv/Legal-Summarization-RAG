# Batched 3B Summarization Runs

This directory contains our alternative batched summarization scripts and their saved outputs for selected datasets. These runs are related to our paper's chunk-summarization stage, but we organized them as standalone experiment scripts rather than the main summarization pipeline.
The naming of the files represents only the CUAD and MAUD datasets, but you can change the dataset to privacy_qa or others to summarize those chunks instead.

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

- we batch many chunks together for generation
- we checkpoint progress as batches complete
- we wrote them as dataset-specific experiment scripts rather than a general reusable command

## Relationship to the Main Pipeline

- Use [`../summarize_docs.py`](../summarize_docs.py) as the simpler, root-level summarization reference.
- Use this directory for our alternative high-throughput summarization runs for specific datasets.
- The JSON outputs here are artifacts from those runs, not canonical benchmark definitions.

## Notes

- These scripts include hardcoded local paths, device settings, and model-access assumptions.
- Review them carefully before reusing on another machine.
