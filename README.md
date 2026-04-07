# Legal-Summarization-RAG

This repository accompanies the paper "A Summarization Framework with Self-Improvement for RAG Retrieval in the Legal Domain." The codebase focuses on improving retrieval in legal RAG systems by replacing raw chunk indexing with validated summaries, and it is organized primarily as a research repository of scripts, generated artifacts, and experiment outputs rather than as a packaged library.

## What This Repository Implements

The paper's workflow maps to the repository as follows:

| Paper stage | Responsibility in code | Main files |
| --- | --- | --- |
| Benchmark inputs | LEGALBENCH-RAG query files and pre-chunked corpora | [`data/benchmarks/`](data/benchmarks/), [`data/original_chunked/`](data/original_chunked/) |
| Chunk summarization | Generate short summaries from fixed-size document chunks | [`summarize_docs.py`](summarize_docs.py), [`baseline_w_3B_summaries/summarize_docs_cuad_baseline_v2.py`](baseline_w_3B_summaries/summarize_docs_cuad_baseline_v2.py), [`baseline_w_3B_summaries/summarize_docs_maud_baseline_v2.py`](baseline_w_3B_summaries/summarize_docs_maud_baseline_v2.py) |
| Quality control and repair | Verify missing outputs, score summary quality, regenerate weak summaries, merge corrections | [`utils/summary_verification.py`](utils/summary_verification.py), [`utils/generate_missing_summaries.py`](utils/generate_missing_summaries.py), [`utils/quality_evaluation/`](utils/quality_evaluation/) |
| Final summary artifacts | Store corrected summaries and document-level summaries used downstream | [`data/combined_summarized_corrected/`](data/combined_summarized_corrected/), [`utils/generate_doc_summary/final_summary_generation.py`](utils/generate_doc_summary/final_summary_generation.py) |
| Vector database construction | Build Chroma collections for summary-based retrieval with different embedding models | [`create_db_baseline/`](create_db_baseline/) |
| Raw-chunk baseline evaluation | Reproduce benchmark-style retrieval over original chunks | [`baseline_from_benchmark/`](baseline_from_benchmark/) |
| Summary-based evaluation | Evaluate hierarchical summary retrieval across embedding variants | [`baseline_V1/evaluation/`](baseline_V1/evaluation/) |

## Repository Guide

| Path | Responsibility |
| --- | --- |
| [`pyproject.toml`](pyproject.toml) | Project dependencies and Python version requirement. |
| [`uv.lock`](uv.lock) | Locked dependency set for `uv` users. |
| [`summarize_docs.py`](summarize_docs.py) | Root summarization entry point for chunk-level summaries. |
| [`data/`](data/) | Benchmark files, chunked inputs, summary artifacts, and analysis helpers. |
| [`utils/`](utils/) | Verification, missing-summary recovery, quality evaluation, and document-summary generation. |
| [`create_db_baseline/`](create_db_baseline/) | Chroma DB builders and small inspection utilities for summary-based retrieval stores. |
| [`baseline_from_benchmark/`](baseline_from_benchmark/) | Raw-chunk baseline indexing and evaluation scripts. |
| [`baseline_V1/evaluation/`](baseline_V1/evaluation/) | Summary-based retrieval evaluation scripts and saved metrics for multiple embeddings. |
| [`baseline_w_3B_summaries/`](baseline_w_3B_summaries/) | Alternative batched summarization runs for selected datasets. |

## Documentation Index

- [`data/README.md`](data/README.md)
- [`utils/README.md`](utils/README.md)
- [`utils/quality_evaluation/README.md`](utils/quality_evaluation/README.md)
- [`create_db_baseline/README.md`](create_db_baseline/README.md)
- [`baseline_from_benchmark/README.md`](baseline_from_benchmark/README.md)
- [`baseline_V1/evaluation/README.md`](baseline_V1/evaluation/README.md)
- [`baseline_w_3B_summaries/README.md`](baseline_w_3B_summaries/README.md)

## Typical Workflow

1. Start from benchmark inputs in [`data/benchmarks/`](data/benchmarks/) and pre-chunked corpora in [`data/original_chunked/`](data/original_chunked/).
2. Generate chunk summaries with [`summarize_docs.py`](summarize_docs.py) or the batched variants in [`baseline_w_3B_summaries/`](baseline_w_3B_summaries/).
3. Verify coverage and repair gaps with [`utils/summary_verification.py`](utils/summary_verification.py) and [`utils/generate_missing_summaries.py`](utils/generate_missing_summaries.py).
4. Score summary quality and correct weak summaries with scripts under [`utils/quality_evaluation/`](utils/quality_evaluation/).
5. Merge corrected summaries and create document-level summaries using [`utils/generate_doc_summary/final_summary_generation.py`](utils/generate_doc_summary/final_summary_generation.py).
6. Build Chroma stores for different embedding models with scripts in [`create_db_baseline/`](create_db_baseline/).
7. Evaluate retrieval either against the raw-chunk benchmark baseline in [`baseline_from_benchmark/`](baseline_from_benchmark/) or the summary-based setup in [`baseline_V1/evaluation/`](baseline_V1/evaluation/).

## Current Model Assignments

- Summary generation scripts use `meta-llama/Llama-3.2-3B-Instruct`.
- This includes chunk summarization, missing-summary recovery, and document-level summary generation.
- Summary quality evaluation and low-score regeneration use `meta-llama/Llama-3.1-8B-Instruct`.

## Setup Notes

### Dependencies

- Python `>=3.11`
- Main dependencies are declared in [`pyproject.toml`](pyproject.toml)
- If you use `uv`:

```bash
uv sync
```

### Git LFS

Large JSON artifacts and several experiment scripts are tracked with Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

### Environment variables

Many scripts assume these are available:

- `HF_TOKEN`
- `OPENAI_API_KEY`

### Runtime assumptions

Most scripts are written as standalone research scripts and commonly assume:

- a CUDA-capable machine
- manual `CUDA_VISIBLE_DEVICES` selection inside the script
- a pre-existing local Chroma persistence directory

## Reproducibility Caveats

- Many scripts still contain hardcoded absolute paths such as `/home/yperezhohin/Legal-Summarization-RAG/...`. Review and normalize them before running on a new machine.
- The repository stores both source scripts and generated experiment artifacts. Not every JSON file is an input; many are saved outputs from prior runs.
- Initial document chunking is treated as an input artifact in this repository. The code here mainly consumes pre-chunked JSON instead of exposing a single canonical chunking script.
- Before publishing or sharing the repository, review scripts for machine-specific settings and any embedded credentials.

## Recommended Reading Order

1. Start here for the high-level map.
2. Read [`data/README.md`](data/README.md) to understand the artifacts.
3. Read [`utils/README.md`](utils/README.md) and [`utils/quality_evaluation/README.md`](utils/quality_evaluation/README.md) for the self-improvement and correction workflow.
4. Read [`create_db_baseline/README.md`](create_db_baseline/README.md) for summary-based vector DB creation.
5. Compare [`baseline_from_benchmark/README.md`](baseline_from_benchmark/README.md) and [`baseline_V1/evaluation/README.md`](baseline_V1/evaluation/README.md) to understand baseline versus proposed retrieval evaluation.
