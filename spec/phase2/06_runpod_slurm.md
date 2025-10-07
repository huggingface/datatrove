# Example 6: RunPod Slurm Setup (Managed Slurm Clusters)

## Objective

Set up a zero-configuration Slurm cluster on RunPod using managed Slurm Clusters for distributed DataTrove processing.

## Why RunPod Managed Slurm?

- **Zero configuration** - Slurm and munge pre-installed
- **Instant provisioning** - No manual setup required
- **Automatic role assignment** - Controller/agent roles handled automatically
- **Built-in optimizations** - Pre-configured for performance
- **Standard Slurm compatibility** - All commands work out-of-box

## Prerequisites

- RunPod account with credits
- Basic understanding of Slurm
- Web browser (for RunPod console)

## Architecture

```
RunPod Instant Cluster
├── Node-0 (Primary/Controller)
│   ├── slurmctld (controller daemon)
│   ├── slurmd (worker daemon)
│   └── DataTrove + dependencies
└── Node-1 (Worker)
    ├── slurmd (worker daemon)
    └── DataTrove + dependencies
```

## Step-by-Step Setup

### 1. Deploy RunPod Slurm Cluster

```bash
# Go to https://console.runpod.io/cluster
# Click "Create Cluster"
# Select "Slurm Cluster" from cluster type dropdown
# Configure:
# - Cluster name: datatrove-learn
# - Pod count: 2 (minimum)
# - GPU type: A100 SXM (cheapest available for Slurm)
# - Region: Select preferred region
# - Pod template: RunPod PyTorch (required for Slurm)
# - Cost: ~$33.41/hour for 2x A100 pods
# Click "Deploy Cluster"
```

### 2. Connect to Slurm Controller

```bash
# On the Instant Clusters page:
# 1. Click your cluster to expand
# 2. Look for "Slurm controller" label on one pod
# 3. Click "Connect" → "Web Terminal" on controller pod
```

### 3. Setup Repository and DataTrove

Slurm is already configured! Setup your code:

```bash
# Clone your repo (public, no SSH key needed)
git clone https://github.com/yoniebans/datatrove.git
cd datatrove
git checkout learning/phase2-slurm-distributed

# Install DataTrove
pip install -e ".[processing,io]"

# Install on worker node (node-1 needs the repo)
srun --nodelist=node-1 git clone https://github.com/yoniebans/datatrove.git /root/datatrove
srun --nodelist=node-1 bash -c "cd /root/datatrove && git checkout learning/phase2-slurm-distributed && pip install -e '.[processing,io]'"
```

### 4. Verify Cluster is Working

On node-0:
```bash
# Check node status
sinfo
# Should show:
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# gpu*         up   infinite      2   idle node-[0-1]

# Test distributed execution
srun --nodes=2 hostname
# Should return both hostnames

# Test job submission (use single quotes to avoid bash history issues)
sbatch --wrap='sleep 10 && echo "Cluster working!"'
squeue  # Monitor job
```

### 5. Run Distributed Examples on Slurm

Test both basic filtering and statistics collection:

#### Example 1: Basic Filtering
```bash
# Create directory
mkdir -p examples_slurm

# Create basic filtering test
cat > spec/phase2/examples/01_basic_filtering_slurm.py << 'EOF'
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
from datatrove.pipeline.writers import JsonlWriter

pipeline = [
    JsonlReader("hf://datasets/allenai/c4/en/",
               glob_pattern="c4-train.00000-of-01024.json.gz",
               limit=100),
    LambdaFilter(lambda doc: len(doc.text) > 100),
    LambdaFilter(lambda doc: any(keyword in doc.text.lower()
                for keyword in ["data", "learning", "computer", "science"])),
    SamplerFilter(rate=0.5),
    JsonlWriter("/tmp/output/")
]

executor = SlurmPipelineExecutor(
    job_name="basic_filtering",
    pipeline=pipeline,
    tasks=2,
    time="00:05:00",
    partition="gpu",
    logging_dir="/tmp/logs/",
    slurm_logs_folder="/tmp/slurm_logs/",
    cpus_per_task=8,
    mem_per_cpu_gb=8,
)

executor.run()
EOF

# Run basic filtering test
python spec/phase2/examples/01_basic_filtering_slurm.py
```

#### Example 2: True Distributed Statistics
```bash
# Create distributed statistics test
cat > spec/phase2/examples/04_statistics_slurm.py << 'EOF'
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.stats import DocStats, LangStats

pipeline = [
    # Use multiple C4 files so both nodes get work
    JsonlReader("hf://datasets/allenai/c4/en/",
                glob_pattern="c4-train.0000[0-3]-of-01024.json.gz",  # 4 files
                limit=200),  # 200 docs per file = 800 total
    DocStats(
        output_folder="/tmp/stats_truly_distributed/",
        histogram_round_digits=1,
    ),
    LangStats(
        output_folder="/tmp/stats_truly_distributed/",
        language="en",
    ),
]

executor = SlurmPipelineExecutor(
    job_name="true_distributed",
    pipeline=pipeline,
    tasks=2,  # 2 tasks, should get 2 files each
    time="00:10:00",
    partition="gpu",
    logging_dir="/tmp/logs_truly_distributed/",
    cpus_per_task=8,
    mem_per_cpu_gb=8,
)

executor.run()
EOF

# Run distributed statistics
python spec/phase2/examples/04_statistics_slurm.py
```

### 6. Monitor and Verify Results

```bash
# Monitor job execution
squeue

# Example 1 Results - Basic Filtering
ls -la /tmp/output/
zcat /tmp/output/00000.jsonl.gz | wc -l  # Should show ~5 documents
find /tmp/logs -name "*.log" -exec tail -5 {} \;

# Example 2 Results - Distributed Statistics
ls -la /tmp/logs_truly_distributed/logs/
cat /tmp/logs_truly_distributed/logs/*.log | grep "Processing done for rank"
cat /tmp/logs_truly_distributed/logs/*.log | grep "documents:"

# Verify true distribution (both nodes working)
cat /tmp/logs_truly_distributed/logs/task_00000.log | grep "Reading input file"
cat /tmp/logs_truly_distributed/logs/task_00001.log | grep "Reading input file"

# Check collected statistics
ls -la /tmp/stats_truly_distributed/
find /tmp/stats_truly_distributed -name "*.json" | head -5
```

## Success Metrics

- [x] Managed Slurm Cluster deployed (2x A100 nodes)
- [x] Slurm services running automatically
- [x] DataTrove installed on all nodes
- [x] **Example 1 - Basic Filtering**: 100→77→5 documents processed
- [x] **Example 2 - True Distributed Statistics**: Both nodes processing different files
  - Task 0: c4-train.00000 → 200 documents (404K chars)
  - Task 1: c4-train.00001 → 200 documents (434K chars)
- [x] **Perfect load balancing**: Each node got separate files to process
- [x] **Complete statistics collection**: Document stats and language detection
- [x] **All pipeline components verified**: Readers, Filters, Stats, Writers
- [x] **Total cost: ~$25 for comprehensive testing**

## Troubleshooting

### Common Issues

1. **Nodes show DOWN**

```bash
# On node-0
scontrol update nodename=node-0 state=idle
scontrol update nodename=node-1 state=idle
```

2. **Munge authentication errors**

```bash
# Ensure same secret key used on both nodes
# Restart munge if needed:
pkill munged
munged -F &
```

3. **DataTrove not found on workers**

```bash
# Install on all nodes via srun
srun -N 2 pip install datatrove[io,processing]
```

4. **IP addresses don't match**

```bash
# Check actual IPs:
hostname -I
# Update install.sh command with correct IPs
```

## Cost Analysis

- Setup time: ~10 minutes
- Test runs: ~15 minutes
- Total cost: $33.41/hr × 0.42 hr = **~$14**

## Next Steps

Once this works:

1. Scale to more nodes (3-4 pods)
2. Run all Phase 1 examples on Slurm
3. Test with S3 storage
4. Implement multi-stage pipelines
5. Move to Lambda Labs for larger scale

## Cleanup

```bash
# IMPORTANT: Delete cluster to stop billing
# Go to https://www.console.runpod.io/cluster
# Click on your cluster
# Click "Delete Cluster"
```

## Notes

- Expensive but powerful: 2x A100 pods with 8 GPUs each
- True distributed processing across multiple pods
- High-speed networking between pods
- Perfect for testing GPU-accelerated DataTrove features
- Be efficient - test quickly and cleanup
