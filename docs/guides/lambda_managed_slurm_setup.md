# Example 7: Lambda Labs Managed Slurm Setup

## Objective
Set up a Lambda Labs 1-Click Cluster (1CC) with Managed Slurm for production-like distributed DataTrove processing.

## Why Lambda's Managed Slurm?
- **Zero configuration** - Slurm pre-installed and managed by Lambda
- **Production-ready** - High availability, monitoring, and support
- **True distributed processing** - Separate physical nodes with high-speed networking
- **Shared filesystems** - `/home` and `/data` across all nodes
- **Container support** - Pyxis/Enroot for containerized workloads
- **User management** - LDAP-based multi-user environment

## Prerequisites
- Lambda Labs account with credits
- SSH public key for access
- Basic Slurm knowledge

## Architecture
```
Lambda 1-Click Cluster (Managed Slurm)
├── Login Node (-head-003)
│   ├── User access point
│   ├── Job submission
│   └── Cluster management
├── Compute Nodes (slurm-compute[001-N])
│   ├── GPU/CPU workers
│   ├── Slurm daemons (managed)
│   └── Container runtime
└── Shared Storage
    ├── /home (user directories)
    └── /data (datasets, shared resources)
```

## Cost Estimate
- 1CC with 2-4 nodes: Variable based on instance types
- **Managed Slurm**: No additional cost
- **Minimum commitment**: 1 week reservation required
- **Support**: Included with SLAs

**⚠️ CAVEAT: Lambda's 1-Click Managed Slurm requires minimum 1-week reservation, making it unsuitable for short learning sessions.**

## Step-by-Step Setup

### 1. Deploy Lambda 1-Click Cluster

```bash
# Go to Lambda Cloud Dashboard → 1-Click Clusters
# Click "Create cluster"
# Configure:
# - Cluster type: Select instance types (e.g., 2x GPU nodes + login node)
# - Region: Choose preferred region
# - SSH Key: Add your public key
# - Slurm: Select "Managed Slurm"
# - Name: datatrove-cluster
# Click "Launch cluster"
```

### 2. Access the Cluster

```bash
# Get login node IP from dashboard (-head-003)
LOGIN_NODE_IP=<IP_FROM_DASHBOARD>

# SSH into login node
ssh ubuntu@$LOGIN_NODE_IP

# Verify cluster status
sinfo
# Should show:
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# gpu*         up   infinite      2   idle slurm-compute[001-002]
```

### 3. Setup DataTrove Repository

```bash
# Clone your DataTrove fork (public repo, no SSH key needed)
git clone https://github.com/yoniebans/datatrove.git
cd datatrove
git checkout learning/phase2-slurm-distributed

# Install DataTrove from source
pip install -e ".[processing,io]"

# Optional: Create user accounts (if needed for team access)
sudo suser add datauser --key ~/.ssh/id_rsa.pub
```

### 5. Verify Cluster

On controller:
```bash
# Check node status
sinfo
# Should show:
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# compute*     up   infinite      2   idle slurm-worker-[1-2]

# Test job submission
srun -N2 hostname
# Should return both worker hostnames

# Submit a batch job
sbatch --wrap="sleep 10 && echo 'Cluster working!'"
squeue  # Monitor job
```

### 6. Run DataTrove Examples

Use the tested examples from our repo:

```bash
# Run existing Slurm examples (from RunPod testing)
python spec/phase2/examples/01_basic_filtering_slurm.py
python spec/phase2/examples/04_statistics_slurm.py

# Monitor execution
squeue -u ubuntu
watch squeue

# Check logs and results
tail -f /tmp/logs*/*/logs/*.log

# View final results
ls -la /tmp/output*/
ls -la /tmp/stats*/
```

## Advanced Configuration

### Add More Workers
```bash
# Launch more instances on Lambda
# Update slurm.conf NodeName line:
NodeName=slurm-worker-[1-4] CPUs=2 RealMemory=3800 State=UNKNOWN

# Restart slurmctld
sudo systemctl restart slurmctld
```

### Add GPU Node
```bash
# Launch GPU instance on Lambda
# Add to slurm.conf:
NodeName=gpu-worker-1 CPUs=8 RealMemory=30000 Gres=gpu:1 State=UNKNOWN
PartitionName=gpu Nodes=gpu-worker-1 MaxTime=INFINITE State=UP
```

### Configure S3 Access
```bash
# On all nodes
pip install boto3
aws configure  # Add your credentials
```

## Success Metrics
- [ ] All nodes show as idle in sinfo
- [ ] Can run jobs across multiple nodes
- [ ] NFS shared storage working
- [ ] DataTrove pipeline runs distributed
- [ ] Logs collected centrally
- [ ] Can scale by adding nodes

## Monitoring Commands
```bash
# Node status
sinfo -Nel

# Job queue
squeue -l

# Job history
sacct --format=JobID,JobName,Partition,State,Elapsed,AllocCPUS,ReqMem

# Node resource usage
srun -N2 htop

# Check DataTrove output
ls -la /shared/output/
wc -l /shared/output/*.jsonl
```

## Troubleshooting

### Nodes show DOWN
```bash
# On controller
sudo scontrol update nodename=slurm-worker-1 state=resume
```

### Munge authentication failures
```bash
# Ensure munge key is identical on all nodes
sudo md5sum /etc/munge/munge.key  # Should match on all

# Restart munge
sudo systemctl restart munge
```

### NFS mount issues
```bash
# Check NFS exports
showmount -e $CONTROLLER_IP

# Remount
sudo umount /shared
sudo mount -t nfs $CONTROLLER_IP:/shared /shared
```

## Cleanup
```bash
# Cancel all running jobs
scancel -u $USER

# On Lambda dashboard
# Terminate all instances to stop billing
```

## Next Steps
1. Run all Phase 1 examples in distributed mode
2. Implement multi-stage MinHash pipeline
3. Test with real S3 data
4. Benchmark scaling (2 vs 4 vs 8 workers)

## Cost Optimization Tips
- Use spot instances if available
- Start with minimum nodes, scale as needed
- Stop (don't terminate) for breaks < 1 hour
- Terminate everything when done for the day

## Notes
- This is real distributed processing
- Each node is a separate physical machine
- Network latency affects performance
- Good for understanding production setups
- Can handle much larger data than RunPod setup