# Example 1: Basic Data Processing Pipeline

## Objective
Learn DataTrove fundamentals: read from HuggingFace datasets, apply filters, save results.

## Components
- **JsonlReader**: Stream from HuggingFace C4 dataset
- **LambdaFilter**: Custom filtering logic (length > 100 chars, keyword matching)
- **SamplerFilter**: Random sampling (50%)
- **JsonlWriter**: Save filtered results (uncompressed)

## Implementation
**File:** `spec/phase1/examples/01_basic_filtering.py`

## Data Requirements
- **Input:** `hf://datasets/allenai/c4/en/` (glob: `c4-train.00000-of-01024.json.gz`, limit: 1000)
- **Output:** `spec/phase1/output/01_filtered/filtered_${rank}.jsonl` (no compression)
- **Logs:** `spec/phase1/logs/01_basic_filtering/`

## Expected Results
- Input: 1000 documents from C4 first shard
- After length filter (>100 chars): ~900 docs
- After keyword filter (data/learning/computer/science): ~200 docs
- After 50% sampling: ~100 docs
- Actual result: 1000 → 77 docs

## Status
- [x] Implemented
- [x] Tested
- [x] Documentation updated

## Notes
- Demonstrates Document structure and pipeline composition
- Includes `inspect_results()` helper function
- Start with 1 task, scale to multiple for parallelization