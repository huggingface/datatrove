# Example 7c: Running DataTrove on Manual Slurm Cluster

## Prerequisites
- Working Slurm cluster from 07b with NFS shared storage at `/shared/`
- DataTrove installed on both nodes (same paths)

## The Issue: Shared Storage Access

DataTrove's `SlurmPipelineExecutor` creates files that all nodes need to access, but by default uses `/tmp/` which is local to each node.

**Solution**: Configure shared storage permissions and modify scripts to use `/shared/` paths.

## Step 1: Fix Shared Storage Permissions

```bash
# On controller node - create group and add users
sudo groupadd shared-users
sudo usermod -aG shared-users ubuntu
sudo usermod -aG shared-users slurm

# Set up shared directories with proper permissions
sudo mkdir -p /shared/{output,logs,slurm_logs,stats}
sudo chown nobody:shared-users /shared/{output,logs,slurm_logs,stats}
sudo chmod 775 /shared/{output,logs,slurm_logs,stats}

# Exit and reconnect SSH to flush group permissions
exit
ssh ubuntu@controller_ip
```

## Step 2: Test Remote File Upload

From your local machine:

```bash
# Test SCP access to shared storage
scp spec/phase2/examples/01_basic_filtering_slurm.py ubuntu@controller_ip:/shared/
```

If this works, permissions are correct.

## Step 3: Modify Scripts for Shared Storage

Edit the uploaded script to use shared paths:

```bash
# On controller node
vim /shared/01_basic_filtering_slurm.py

# Change these lines:
# logging_dir="/tmp/logs/"           -> logging_dir="/shared/logs/"
# slurm_logs_folder="/tmp/slurm_logs/" -> slurm_logs_folder="/shared/slurm_logs/"
```

Example fixed script:

```python
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
from datatrove.pipeline.writers import JsonlWriter

pipeline = [
    JsonlReader("hf://datasets/allenai/c4/en/",
               glob_pattern="c4-train.00000-of-01024.json.gz", limit=100),
    LambdaFilter(lambda doc: len(doc.text) > 200),
    LambdaFilter(lambda doc: len(doc.text.split()) > 50),
    SamplerFilter(rate=0.5),
    JsonlWriter("/shared/output/")
]

executor = SlurmPipelineExecutor(
    job_name="basic_filtering",
    pipeline=pipeline,
    tasks=2,
    time="00:05:00",
    partition="gpu",
    logging_dir="/shared/logs/",              # Fixed path
    slurm_logs_folder="/shared/slurm_logs/",  # Fixed path
    cpus_per_task=8,
    mem_per_cpu_gb=8,
)

executor.run()
```

## Step 4: Run Distributed Processing

```bash
cd /shared
source ~/venv/bin/activate
python 01_basic_filtering_slurm.py
```

Monitor with:
```bash
squeue
sacct --format=JobID,JobName,Partition,State,NodeList
```

## Example 2: Statistics Collection

```bash
# From local machine - create stats script
cat > 04_statistics_slurm.py << 'EOF'
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.stats import DocStats, LangStats

pipeline = [
    JsonlReader("hf://datasets/allenai/c4/en/",
                glob_pattern="c4-train.0000[0-3]-of-01024.json.gz",
                limit=200),
    DocStats(output_folder="/shared/stats/"),
    LangStats(output_folder="/shared/stats/", language="en"),
]

executor = SlurmPipelineExecutor(
    job_name="stats_distributed",
    pipeline=pipeline,
    tasks=2,
    time="00:10:00",
    partition="gpu",
    logging_dir="/shared/logs/",
    slurm_logs_folder="/shared/slurm_logs/",
    cpus_per_task=8,
    mem_per_cpu_gb=8,
)

executor.run()
EOF

# Upload and run
scp 04_statistics_slurm.py ubuntu@controller_ip:/shared/
ssh ubuntu@controller_ip "cd /shared && source ~/venv/bin/activate && python 04_statistics_slurm.py"
```

## Example 3: Large Scale Processing

```bash
# From local machine - create large scale script
cat > large_scale_processing.py << 'EOF'
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LambdaFilter, LanguageFilter
from datatrove.pipeline.writers import JsonlWriter

pipeline = [
    JsonlReader("hf://datasets/allenai/c4/en/",
               glob_pattern="c4-train.00[0-4]*-of-01024.json.gz"),  # ~50 files
    LambdaFilter(lambda doc: len(doc.text) > 1000),
    LanguageFilter(languages=["en"]),
    JsonlWriter("/shared/output/")
]

executor = SlurmPipelineExecutor(
    job_name="large_scale_processing",
    pipeline=pipeline,
    tasks=20,                                 # 20 parallel tasks
    time="01:00:00",                         # 1 hour limit
    partition="gpu",
    logging_dir="/shared/logs/",
    slurm_logs_folder="/shared/slurm_logs/",
    cpus_per_task=8,
    mem_per_cpu_gb=8,
)

executor.run()
EOF

# Upload and run
scp large_scale_processing.py ubuntu@controller_ip:/shared/
ssh ubuntu@controller_ip "cd /shared && source ~/venv/bin/activate && python large_scale_processing.py"

# Monitor distributed execution
ssh ubuntu@controller_ip "watch squeue"
```

**Expected behavior**: Multiple jobs running simultaneously with job queuing when resources are full.

## Key Files Created

After successful runs, check:
- `/shared/output/*.jsonl.gz` - Processed data
- `/shared/logs/stats.json` - Processing statistics
- `/shared/slurm_logs/*.out` - Job logs from both nodes
- `/shared/logs/logs/task_*.log` - Individual task logs

## Workflow Summary

1. **SCP script** from local machine to `/shared/`
2. **Edit paths** in script to use `/shared/` directories
3. **Run script** from controller node
4. **Monitor** with `squeue` and check logs in `/shared/slurm_logs/`
5. **Results** saved to `/shared/output/`

This approach works for any DataTrove script - just change the paths to shared storage.