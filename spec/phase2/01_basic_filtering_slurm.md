# Example 01: Basic Filtering with Slurm

## Objective
Apply basic filtering to C4 data using SlurmPipelineExecutor for distributed processing across cluster nodes.

## Components
- **JsonlReader**: Read from HuggingFace C4 dataset
- **LambdaFilter**: Filter by length and keywords
- **SamplerFilter**: Random sampling
- **JsonlWriter**: Save filtered results
- **SlurmPipelineExecutor**: Distributed execution across Slurm cluster

## Implementation
**File:** `spec/phase2/examples/01_basic_filtering_slurm.py`

## Data Requirements
- **Input:** `hf://datasets/allenai/c4/en/` (glob: `c4-train.00000-of-01024.json.gz`, limit: 100)
- **Output:** `/tmp/output/*.jsonl.gz` (or `/shared/output/` on manual clusters)
- **Logs:** `/tmp/logs/` (or `/shared/logs/` on manual clusters)
- **Slurm Logs:** `/tmp/slurm_logs/` (or `/shared/slurm_logs/` on manual clusters)

## Expected Results
- Input: 100 documents from C4
- After length filter (>100 chars): ~77 docs
- After keyword filter: subset with keywords
- After sampling (50%): ~5 docs
- Jobs distributed across 2 Slurm tasks

## Status
- [x] Implemented
- [x] Tested on RunPod managed Slurm
- [x] Documentation updated

## Notes
- Requires Slurm cluster setup (see docs/guides/)
- Uses SlurmPipelineExecutor instead of LocalPipelineExecutor
- Paths should use shared storage (/shared/) for manual clusters
- For managed clusters (RunPod), /tmp/ paths work due to automatic synchronization
