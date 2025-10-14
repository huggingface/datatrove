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

### 5. Run DataTrove Examples on Slurm

The DataTrove examples are in `spec/phase2/examples/`. On RunPod managed Slurm, `/tmp/` paths work automatically across nodes.

```bash
# Run basic filtering example
python spec/phase2/examples/01_basic_filtering_slurm.py

# Run distributed statistics example
python spec/phase2/examples/04_statistics_slurm.py
```

**Note**: RunPod's managed Slurm automatically synchronizes `/tmp/` directories, so the examples work without modification.

### 6. Monitor and Verify Results

```bash
# Monitor job execution
squeue

# Check basic filtering results
ls -la /tmp/output/
zcat /tmp/output/00000.jsonl.gz | wc -l  # Should show ~5 documents

# Check statistics results
ls -la /tmp/stats/
find /tmp/stats -name "*.json" | head -5

# Check logs to verify distribution
find /tmp/logs -name "*.log" -exec tail -5 {} \;
```

## Success Metrics

- [x] Managed Slurm Cluster deployed (2x A100 nodes)
- [x] Slurm services running automatically
- [x] DataTrove installed on all nodes
- [x] Both examples run successfully with distributed processing
- [x] Jobs properly distributed across nodes
- [x] All pipeline components verified: Readers, Filters, Stats, Writers
- [x] Total cost: ~$25 for comprehensive testing

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
