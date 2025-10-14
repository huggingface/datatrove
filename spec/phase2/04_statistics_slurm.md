# Example 04: Statistics Collection with Slurm

## Objective
Collect document and language statistics using SlurmPipelineExecutor for distributed processing across multiple files.

## Components
- **JsonlReader**: Read multiple C4 files for distribution
- **DocStats**: Document-level statistics
- **LangStats**: Language detection statistics
- **SlurmPipelineExecutor**: Distributed execution with work distribution

## Implementation
**File:** `spec/phase2/examples/04_statistics_slurm.py`

## Data Requirements
- **Input:** `hf://datasets/allenai/c4/en/` (glob: `c4-train.0000[0-3]-of-01024.json.gz`, limit: 200 per file)
- **Output:** `/tmp/stats/` (or `/shared/stats/` on manual clusters)
- **Logs:** `/tmp/logs/` (or `/shared/logs/` on manual clusters)

## Expected Results
- Input: 4 C4 files, 200 docs each = 800 total documents
- Task 0 processes: c4-train.00000, c4-train.00001 (~400 docs, ~404K chars)
- Task 1 processes: c4-train.00002, c4-train.00003 (~400 docs, ~434K chars)
- Perfect load balancing: Each task gets 2 files
- Statistics collected: Document length, whitespace ratio, language confidence

## Status
- [x] Implemented
- [x] Tested on RunPod managed Slurm
- [x] Documentation updated

## Notes
- Demonstrates distributed processing (different files per task)
- Multiple input files ensure work distribution across tasks
- Stats are collected per-task and can be merged
- Requires Slurm cluster setup (see docs/guides/)
