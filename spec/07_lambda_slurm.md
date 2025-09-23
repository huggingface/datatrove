# Example 7: Lambda Labs Slurm Setup (Production-Like)

## Objective
Set up a proper distributed Slurm cluster on Lambda Labs with separate physical nodes for real distributed processing.

## Why Lambda Labs After RunPod?
- Real distributed processing across physical machines
- More production-like setup
- Better for testing actual scaling
- Learn real cluster administration
- Can add GPU nodes later if needed

## Prerequisites
- Lambda Labs account with credits
- SSH key added to Lambda Labs
- Basic Linux administration knowledge

## Architecture
```
Lambda Labs Infrastructure
├── slurm-controller (2 vCPU, 4GB RAM)
│   ├── slurmctld (controller daemon)
│   ├── slurmdbd (database daemon)
│   └── NFS server (/shared)
├── slurm-worker-1 (2 vCPU, 4GB RAM)
│   ├── slurmd (worker daemon)
│   └── NFS client (/shared)
└── slurm-worker-2 (2 vCPU, 4GB RAM)
    ├── slurmd (worker daemon)
    └── NFS client (/shared)
```

## Cost Estimate
- Controller: $0.10/hr
- Worker 1: $0.10/hr
- Worker 2: $0.10/hr
- **Total: $0.30/hr** (~$1.50 for 5-hour learning session)

## Step-by-Step Setup

### 1. Launch Lambda Labs Instances

```bash
# Go to Lambda Labs Cloud dashboard
# Launch 3 instances:
# - Name: slurm-controller (Ubuntu 22.04, 2 vCPU, 4GB RAM)
# - Name: slurm-worker-1 (Ubuntu 22.04, 2 vCPU, 4GB RAM)
# - Name: slurm-worker-2 (Ubuntu 22.04, 2 vCPU, 4GB RAM)

# Note the IPs:
# CONTROLLER_IP=xxx.xxx.xxx.xxx
# WORKER1_IP=xxx.xxx.xxx.xxx
# WORKER2_IP=xxx.xxx.xxx.xxx
```

### 2. Initial Setup Script (Run on ALL nodes)

Create `initial_setup.sh`:
```bash
#!/bin/bash
# Run on all nodes

# Update system
sudo apt update && sudo apt upgrade -y

# Install basic tools
sudo apt install -y build-essential git vim htop

# Create slurm user
sudo useradd -m -s /bin/bash slurm
sudo usermod -aG sudo slurm

# Install Python and pip
sudo apt install -y python3.10 python3-pip python3.10-venv

# Install DataTrove
pip install datatrove[io,processing]

# Create directories
sudo mkdir -p /var/spool/slurm/ctld
sudo mkdir -p /var/spool/slurm/d
sudo mkdir -p /var/log/slurm
sudo mkdir -p /shared

# Set permissions
sudo chown -R slurm:slurm /var/spool/slurm
sudo chown -R slurm:slurm /var/log/slurm
```

### 3. Controller-Specific Setup

SSH into controller and create `controller_setup.sh`:
```bash
#!/bin/bash

# Install Slurm controller packages
sudo apt install -y slurmctld slurmdbd mysql-server

# Install NFS server
sudo apt install -y nfs-kernel-server

# Configure NFS exports
echo "/shared *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo systemctl restart nfs-kernel-server

# Install munge for authentication
sudo apt install -y munge libmunge-dev
sudo systemctl enable munge
sudo systemctl start munge

# Create Slurm configuration
sudo tee /etc/slurm/slurm.conf << 'EOF'
# Cluster Configuration
ClusterName=datatrove
ControlMachine=slurm-controller
ControlAddr=CONTROLLER_IP
AuthType=auth/munge
CryptoType=crypto/munge

# Slurm User
SlurmUser=slurm
SlurmdUser=root
SlurmctldPort=6817
SlurmdPort=6818

# State Preservation
StateSaveLocation=/var/spool/slurm/ctld
SlurmdSpoolDir=/var/spool/slurm/d

# Logging
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
SlurmctldDebug=info
SlurmdDebug=info

# Process Tracking
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup

# Scheduling
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core

# Timing
SlurmctldTimeout=300
SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
KillWait=30
Waittime=0

# Compute Nodes
NodeName=slurm-worker-[1-2] CPUs=2 RealMemory=3800 State=UNKNOWN
PartitionName=compute Nodes=slurm-worker-[1-2] Default=YES MaxTime=INFINITE State=UP
EOF

# Replace CONTROLLER_IP with actual IP
sudo sed -i "s/CONTROLLER_IP/$CONTROLLER_IP/g" /etc/slurm/slurm.conf

# Configure cgroup
sudo tee /etc/slurm/cgroup.conf << 'EOF'
CgroupAutomount=yes
ConstrainCores=yes
ConstrainRAMSpace=yes
EOF

# Start services
sudo systemctl enable slurmctld
sudo systemctl start slurmctld
```

### 4. Worker-Specific Setup

SSH into each worker and create `worker_setup.sh`:
```bash
#!/bin/bash

# Install Slurm worker packages
sudo apt install -y slurmd

# Install NFS client
sudo apt install -y nfs-common

# Mount shared directory
sudo mount $CONTROLLER_IP:/shared /shared
echo "$CONTROLLER_IP:/shared /shared nfs defaults 0 0" | sudo tee -a /etc/fstab

# Install munge
sudo apt install -y munge libmunge-dev

# Copy munge key from controller (do this manually)
# scp controller:/etc/munge/munge.key /tmp/munge.key
sudo cp /tmp/munge.key /etc/munge/munge.key
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key

# Copy slurm.conf from controller
# scp controller:/etc/slurm/slurm.conf /etc/slurm/slurm.conf

# Start services
sudo systemctl enable munge slurmd
sudo systemctl start munge slurmd
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

### 6. Test DataTrove Pipeline

Create `/shared/test_distributed.py`:
```python
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
from datatrove.pipeline.writers import JsonlWriter

pipeline = [
    JsonlReader(
        "hf://datasets/allenai/c4/en/",
        glob_pattern="c4-train.00000-of-01024.json.gz",
        limit=1000
    ),
    LambdaFilter(lambda doc: len(doc.text) > 100),
    SamplerFilter(rate=0.5),
    JsonlWriter("/shared/output/")
]

executor = SlurmPipelineExecutor(
    job_name="datatrove_test",
    pipeline=pipeline,
    tasks=4,  # Will distribute across workers
    time="00:10:00",
    partition="compute",
    logging_dir="/shared/logs/",
    slurm_logs_folder="/shared/slurm_logs/",
    cpus_per_task=1,
    mem_per_cpu_gb=1,
)

executor.run()
```

Run it:
```bash
cd /shared
python test_distributed.py

# Monitor
watch squeue
tail -f /shared/logs/datatrove_test/logs/*.log
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