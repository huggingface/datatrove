# Example 6: RunPod Slurm Setup (Quick Docker Approach)

## Objective

Set up a minimal Slurm cluster on RunPod using Docker containers for quick testing of DataTrove distributed processing.

## Why RunPod First?

- Faster setup with pre-built Docker images
- Good for initial testing and validation
- Lower complexity to get started
- Can tear down and recreate easily

## Prerequisites

- RunPod account with credits
- Basic understanding of Docker
- SSH client

## Architecture

```
RunPod Pod (Single Instance)
├── Docker Network
│   ├── slurmctld (controller container)
│   ├── slurmd1 (worker container)
│   ├── slurmd2 (worker container)
│   └── slurmd3 (worker container)
└── Shared Volume (/data)
```

## Step-by-Step Setup

### 1. Create RunPod Instance

```bash
# RunPod Configuration:
# - Template: RunPod Pytorch 2.0
# - GPU: NONE (CPU only for initial testing)
# - Container Disk: 50 GB
# - Volume Disk: 20 GB
# - Cost: ~$0.10/hour for CPU instance
```

### 2. SSH into Pod

```bash
ssh root@[YOUR_POD_IP] -p [YOUR_PORT]
```

### 3. Install Docker Slurm Cluster

```bash
# Pull the Slurm Docker cluster image
docker pull giovtorres/slurm-docker-cluster

# Run the cluster (creates controller + 2 workers)
docker run -it --rm \
    --name slurm-cluster \
    -h slurm-cluster \
    -v /workspace:/data \
    --privileged \
    giovtorres/slurm-docker-cluster

# In another terminal, exec into the container
docker exec -it slurm-cluster /bin/bash
```

### 4. Verify Slurm is Working

```bash
# Inside container
sinfo  # Should show c1, c2 nodes
squeue # Should be empty
srun -N 2 hostname # Should return c1, c2

# Test job submission
sbatch --wrap="sleep 30 && echo 'Hello from Slurm'"
squeue # Should show job running
```

### 5. Install DataTrove in Container

```bash
# Inside the Slurm container
pip install datatrove[io,processing]

# Also install in worker nodes
docker exec slurmd1 pip install datatrove[io,processing]
docker exec slurmd2 pip install datatrove[io,processing]
```

### 6. Create Test Pipeline Script

```python
# /data/test_slurm.py
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter

pipeline = [
    JsonlReader(
        "hf://datasets/allenai/c4/en/",
        glob_pattern="c4-train.00000-of-01024.json.gz",
        limit=100
    ),
    LambdaFilter(lambda doc: len(doc.text) > 50),
    JsonlWriter("/data/output/")
]

executor = SlurmPipelineExecutor(
    job_name="test_datatrove",
    pipeline=pipeline,
    tasks=2,
    time="00:05:00",
    partition="normal",
    logging_dir="/data/logs/",
    slurm_logs_folder="/tmp/slurm_logs/",
    cpus_per_task=1,
    mem_per_cpu_gb=1,
)

executor.run()
```

### 7. Run the Test

```bash
cd /data
python test_slurm.py

# Monitor
squeue
tail -f /data/logs/*/logs/*.log
```

## Success Metrics

- [ ] Slurm cluster running (sinfo shows nodes)
- [ ] Can submit jobs (sbatch works)
- [ ] DataTrove installed on all nodes
- [ ] Test pipeline runs successfully
- [ ] Output files created in /data/output/
- [ ] Logs show distributed execution

## Troubleshooting

### Common Issues

1. **Nodes show DOWN**

```bash
scontrol update nodename=c1 state=idle
scontrol update nodename=c2 state=idle
```

2. **Permission issues**

```bash
chmod -R 777 /data
```

3. **DataTrove not found**

```bash
# Install on all nodes
for node in c1 c2; do
    srun -N 1 --nodelist=$node pip install datatrove
done
```

## Cost Analysis

- Setup time: ~30 minutes
- Test runs: ~30 minutes
- Total cost: $0.10/hr × 1 hr = **$0.10**

## Next Steps

Once this works:

1. Run Example 1 converted to Slurm
2. Test with more workers
3. Try with actual S3 storage
4. Move to Lambda Labs for production setup

## Cleanup

```bash
# Exit container
exit

# Stop Docker container (auto-removes with --rm flag)
docker stop slurm-cluster

# Terminate RunPod instance to stop billing
# Go to RunPod dashboard → Stop pod
```

## Notes

- This is a learning setup, not production
- All nodes run on same physical machine
- Good for understanding Slurm commands and DataTrove integration
- Real distributed processing needs separate physical nodes (see Example 7: Lambda Labs)
