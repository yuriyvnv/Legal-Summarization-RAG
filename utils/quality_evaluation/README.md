# Quality Evaluation

This directory contains the scripts we use to evaluate, diagnose, repair, and merge chunk summaries. Together, they implement our practical version of the quality-control stage described in our paper.

## Structure

| Path | Responsibility |
| --- | --- |
| [`summary_eval/`](summary_eval/) | Dataset-specific summary scoring scripts. |
| [`analyze_low_scores.py`](analyze_low_scores.py) | Extracts low-scoring summaries and aggregates diagnostic reports. |
| [`correct_low_scores.py`](correct_low_scores.py) | Regenerates weak summaries using evaluator feedback and writes corrected outputs. |
| [`merge_correct_jsons.py`](merge_correct_jsons.py) | Applies regenerated summaries back into the original summary structures and writes a merged artifact. |
| [`results/`](results/) | Saved outputs such as corrected summaries and analysis reports. |

## Dataset-Specific Scoring Scripts

| File | Responsibility |
| --- | --- |
| [`summary_eval/summaries_quality_evaluation_cuad.py`](summary_eval/summaries_quality_evaluation_cuad.py) | Scores CUAD chunk summaries against their original chunk text. |
| [`summary_eval/summaries_quality_evaluation_maud.py`](summary_eval/summaries_quality_evaluation_maud.py) | Scores MAUD chunk summaries against their original chunk text. |
| [`summary_eval/summaries_quality_evaluation_contractnli.py`](summary_eval/summaries_quality_evaluation_contractnli.py) | Scores ContractNLI chunk summaries against their original chunk text. |
| [`summary_eval/summaries_quality_evaluation_privacy.py`](summary_eval/summaries_quality_evaluation_privacy.py) | Scores Privacy-QA chunk summaries against their original chunk text. |

These scripts use `meta-llama/Llama-3.1-8B-Instruct` evaluation prompts and write structured evaluation artifacts for downstream analysis.

## Correction Workflow

1. Generate evaluation outputs with the scripts in [`summary_eval/`](summary_eval/).
2. Run [`analyze_low_scores.py`](analyze_low_scores.py) to identify summaries with scores below the chosen threshold.
3. Run [`correct_low_scores.py`](correct_low_scores.py) to regenerate improved summaries for those weak cases.
4. Run [`merge_correct_jsons.py`](merge_correct_jsons.py) to merge regenerated summaries into a combined updated summary file.

## Important Output Files

| File | Responsibility |
| --- | --- |
| [`results/corrected_summaries.json`](results/corrected_summaries.json) | Regenerated summaries for low-scoring chunks. |
| `results/summary_analysis_results.json` | Aggregate analysis report of low-scoring cases, produced by [`analyze_low_scores.py`](analyze_low_scores.py). |

## Non-Obvious Details

- The analyzer in [`analyze_low_scores.py`](analyze_low_scores.py) treats scores below `0.6` as low-quality cases, which matches the threshold discussed in the paper.
- The regeneration step in [`correct_low_scores.py`](correct_low_scores.py) uses `meta-llama/Llama-3.1-8B-Instruct` together with the original chunk text, prior summary, and evaluator feedback to produce a new summary.
- The merge step in [`merge_correct_jsons.py`](merge_correct_jsons.py) uses `chunk_text` as the key for replacing summaries in the original data structure.

## Additional Utility

- There is also an auxiliary verification script named `utils/quality_evaluation/ verify_missing_docs.py`. Its filename currently includes a leading space, so it is best treated as an ad hoc helper rather than part of the main documented workflow.

## Caveats

- These are research utilities with hardcoded paths and large JSON dependencies.
- We designed them around saved artifacts, not a single reusable API.
- Review local path assumptions before rerunning on another machine.
