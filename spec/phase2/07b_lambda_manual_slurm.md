# Example 7b: Lambda Labs Manual Slurm Setup (H100 On-Demand)

## Objective

Set up a DIY Slurm cluster on Lambda Labs using on-demand H100 instances for learning cluster administration and distributed DataTrove processing.

## Why Manual Slurm?

- **Sysadmin learning** - Hands-on cluster setup experience
- **No minimum commitment** - Use on-demand instances for short sessions
- **Full control** - Configure Slurm exactly as needed
- **GPU acceleration** - H100s for future GPU-accelerated DataTrove features
- **Cost efficiency** - Pay only for what you use

## Prerequisites

- Lambda Labs account with credits
- SSH key configured in Lambda dashboard
- Basic Linux administration knowledge
- Understanding of Slurm concepts

## Architecture

```
Lambda Manual Slurm Cluster
├── Controller Node (1x NVIDIA H100)
│   ├── slurmctld (controller daemon)
│   ├── slurmdbd (database daemon)
│   ├── MySQL/MariaDB
│   ├── Munge authentication
│   └── NFS server (/shared)
└── Worker Node (1x NVIDIA H100)
    ├── slurmd (worker daemon)
    ├── Munge authentication
    └── NFS client (/shared)
```

## Cost Estimate

- 1x H100 instances: ~$0.80/hour each
- **Total: ~$1.60/hour**
- 2-hour session: ~$3.20
- **Much cheaper than 1-week commitment!**

## Step-by-Step Setup

### 1. Launch Lambda Instances

```bash
# Go to Lambda Cloud Console
# Launch 2 instances:
# - Name: slurm-controller
# - Instance: 1x NVIDIA H100 (on-demand)
# - SSH Key: Select your key
# - Name: slurm-worker
# - Instance: 1x NVIDIA H100 (on-demand)
# - SSH Key: Select your key

# Note the IPs from dashboard:
CONTROLLER_IP=xxx.xxx.xxx.xxx
WORKER_IP=xxx.xxx.xxx.xxx

# IMPORTANT: Lambda uses IP addresses as hostnames, which causes Slurm issues.
# We'll need to set proper hostnames during setup.
```

### 2. Base Setup (Run on BOTH nodes)

SSH into each node and run:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    build-essential \
    git \
    vim \
    htop \
    curl \
    wget \
    software-properties-common

# Create slurm user
sudo useradd -m -s /bin/bash slurm
sudo usermod -aG sudo slurm

# Install python3 dev headers, venv and python3-full first
sudo apt install -y python3-dev python3-pip python3-venv python3-full

# Create virtual environment
python3 -m venv ~/venv
source ~/venv/bin/activate

# Now pip will work
pip install --upgrade pip

# Clone our DataTrove repo
git clone https://github.com/yoniebans/datatrove.git
cd datatrove
git checkout learning/phase2-slurm-distributed
pip install -e ".[processing,io]"
```

### 3. Controller Node Setup

SSH into controller node:

```bash
# Install Slurm controller packages (including slurmd since controller is also compute node)
sudo apt install -y \
    slurmctld \
    slurmdbd \
    slurm-client \
    slurmd \
    mariadb-server \
    munge \
    libmunge-dev

# Install NFS server
sudo apt install -y nfs-kernel-server

# Create shared directory
sudo mkdir -p /shared
sudo chown nobody:nogroup /shared
sudo chmod 755 /shared

# Configure NFS exports
echo "/shared *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo systemctl restart nfs-kernel-server
sudo systemctl enable nfs-kernel-server

# Configure MySQL for Slurm accounting
sudo mysql -u root << EOF
CREATE DATABASE slurm_acct_db;
CREATE USER 'slurm'@'localhost' IDENTIFIED BY 'slurmpass';
GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';
FLUSH PRIVILEGES;
EOF

# Generate munge key
sudo systemctl stop munge

# Remove the existing key and create a new one
sudo rm /etc/munge/munge.key
sudo -u munge /usr/sbin/mungekey -v

# Start munge
sudo systemctl start munge
sudo systemctl enable munge

# Verify it's working
sudo systemctl status munge

# IMPORTANT: Set proper hostname (Lambda uses IP as hostname by default)
sudo hostnamectl set-hostname slurm-controller
echo "127.0.0.1 slurm-controller" | sudo tee -a /etc/hosts

# Create Slurm configuration (use actual IP for ControlAddr, hostname for ControlMachine)
sudo tee /etc/slurm/slurm.conf << EOF
# Cluster Configuration
ClusterName=lambda-datatrove
ControlMachine=slurm-controller
ControlAddr=$CONTROLLER_IP

# Authentication
AuthType=auth/munge
CryptoType=crypto/munge

# Users
SlurmUser=slurm
SlurmdUser=root

# Ports
SlurmctldPort=6817
SlurmdPort=6818

# Directories
StateSaveLocation=/var/spool/slurm/ctld
SlurmdSpoolDir=/var/spool/slurm/d
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log

# Process Tracking
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup

# Scheduling
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

# Accounting
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=localhost
AccountingStorageUser=slurm

# Nodes and Partitions
NodeName=slurm-controller CPUs=16 RealMemory=100000 Gres=gpu:h100:1 State=UNKNOWN
NodeName=slurm-worker CPUs=16 RealMemory=100000 Gres=gpu:h100:1 State=UNKNOWN

PartitionName=gpu Nodes=slurm-controller,slurm-worker Default=YES MaxTime=INFINITE State=UP
EOF

# Create GRES configuration
sudo tee /etc/slurm/gres.conf << EOF
NodeName=slurm-controller Name=gpu Type=h100 File=/dev/nvidia0
NodeName=slurm-worker Name=gpu Type=h100 File=/dev/nvidia0
EOF

# Configure slurmdbd
sudo tee /etc/slurm/slurmdbd.conf << EOF
AuthType=auth/munge
DbdHost=localhost
DbdPort=6819
SlurmUser=slurm
LogFile=/var/log/slurm/slurmdbd.log
PidFile=/var/run/slurmdbd.pid
StorageType=accounting_storage/mysql
StorageHost=localhost
StorageUser=slurm
StoragePass=slurmpass
StorageLoc=slurm_acct_db
EOF

sudo chown slurm:slurm /etc/slurm/slurmdbd.conf
sudo chmod 600 /etc/slurm/slurmdbd.conf

# Create directories
sudo mkdir -p /var/spool/slurm/{ctld,d}
sudo mkdir -p /var/log/slurm
sudo chown -R slurm:slurm /var/spool/slurm
sudo chown -R slurm:slurm /var/log/slurm

# Start services (controller needs slurmdbd, slurmctld, AND slurmd)
sudo systemctl enable slurmdbd slurmctld slurmd
sudo systemctl start slurmdbd
sleep 5
sudo systemctl start slurmctld
sudo systemctl start slurmd

# Add worker hostname to /etc/hosts for name resolution
echo "$WORKER_IP slurm-worker" | sudo tee -a /etc/hosts

# Copy configuration files to shared storage for worker node
sudo cp /etc/munge/munge.key /shared/
sudo cp /etc/slurm/slurm.conf /shared/
sudo cp /etc/slurm/gres.conf /shared/
```

### 4. Worker Node Setup

SSH into worker node:

```bash
# Install Slurm worker packages
sudo apt install -y slurmd slurm-client munge libmunge-dev

# Install NFS client
sudo apt install -y nfs-common

# Mount shared directory
sudo mkdir -p /shared
sudo mount $CONTROLLER_IP:/shared /shared
echo "$CONTROLLER_IP:/shared /shared nfs defaults 0 0" | sudo tee -a /etc/fstab

# IMPORTANT: Set proper hostname for worker too
sudo hostnamectl set-hostname slurm-worker
echo "127.0.0.1 slurm-worker" | sudo tee -a /etc/hosts
echo "$CONTROLLER_IP slurm-controller" | sudo tee -a /etc/hosts

# Copy configuration files from shared storage (controller should have copied them)
# NOTE: This uses NFS shared storage for simplicity in learning environments.
# SECURITY WARNING: This approach exposes sensitive files (munge key) on shared storage.
# In production, use proper SSH key distribution or configuration management tools.
sudo cp /shared/munge.key /etc/munge/munge.key
sudo cp /shared/slurm.conf /etc/slurm/
sudo cp /shared/gres.conf /etc/slurm/
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key

# Create directories
sudo mkdir -p /var/spool/slurm/d
sudo mkdir -p /var/log/slurm
sudo chown -R slurm:slurm /var/spool/slurm
sudo chown -R slurm:slurm /var/log/slurm

# Start services
sudo systemctl enable munge slurmd
sudo systemctl start munge
sleep 2  # Wait for munge to initialize
sudo systemctl start slurmd
```

### 5. Verify Cluster

On controller node:

```bash
# IMPORTANT: Clean up sensitive files from shared storage first
sudo rm /shared/munge.key /shared/slurm.conf /shared/gres.conf

# Check cluster status (both nodes should show as idle)
sinfo
# Should show:
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# gpu*         up   infinite      2   idle slurm-controller,slurm-worker

# If nodes show as UNKNOWN, check services and try:
# sudo systemctl status slurmctld slurmd  # on controller
# sudo systemctl status slurmd munge     # on worker

# Check detailed node information
scontrol show nodes

# Check GPU resources
sinfo -o "%N %G %C %m"

# Test basic job distribution
srun --nodes=2 --ntasks=2 hostname

# Test GPU job
srun --gres=gpu:1 nvidia-smi -L

# Add cluster to accounting
sudo sacctmgr -i add cluster lambda-datatrove
sudo sacctmgr -i add account lambda-account cluster=lambda-datatrove
sudo sacctmgr -i add user ubuntu account=lambda-account cluster=lambda-datatrove
```

### 6. Final Cluster Validation

```bash
# Test distributed job execution across both nodes
srun --nodes=2 --ntasks=2 hostname
# Should show both hostnames:
# slurm-controller
# slurm-worker

# Test GPU access
srun --gres=gpu:1 nvidia-smi -L
# Should show: GPU 0: NVIDIA H100 PCIe (UUID: GPU-xxx)

# Test shared storage from both nodes
echo "test from controller" | sudo tee /shared/test.txt
srun --nodelist=slurm-worker cat /shared/test.txt
# Should show: test from controller

# Clean up test file
sudo rm /shared/test.txt

# Verify cluster is ready for distributed processing
scontrol show partition
sinfo -Nel
```

## Cluster Setup Complete! 🎉

Your manual H100 Slurm cluster is now fully operational with:

- ✅ **2x H100 nodes** (controller + worker)
- ✅ **Distributed job scheduling** via Slurm
- ✅ **Shared NFS storage** at `/shared/`
- ✅ **GPU resource management**
- ✅ **Proper authentication** via Munge
- ✅ **Network connectivity** between nodes

**Next Steps**: See `07c_datatrove_slurm_execution.md` for running distributed DataTrove processing on this cluster.

**Cost**: Remember to terminate instances when done to avoid charges (~$1.60/hour).


## Success Metrics

- [ ] Both nodes show as idle in sinfo
- [ ] GPU resources visible (sinfo -o "%N %G")
- [ ] Can run distributed jobs across nodes
- [ ] NFS shared storage working
- [ ] DataTrove examples run successfully
- [ ] GPU allocation working

## Monitoring Commands

```bash
# Cluster overview
scontrol show nodes
sinfo -Nel

# Job monitoring
squeue -l
sacct --format=JobID,JobName,Partition,State,Elapsed,AllocCPUS,ReqGRES

# Resource usage
srun --gres=gpu:1 nvidia-smi
```

## Cleanup

```bash
# Stop all services
sudo systemctl stop slurmd slurmctld slurmdbd

# On Lambda dashboard: Terminate both instances
```

## Cost Analysis

- **2-hour learning session**: ~$3.20
- **4-hour deep dive**: ~$6.40
- **vs 1-week managed**: $500-1000+

Much more economical for learning!

## Key Setup Issues and Fixes

During development of this guide, we encountered several common issues:

### 1. Python Environment Issues
- **Problem**: Ubuntu 24.04 prevents system-wide pip installs
- **Solution**: Use virtual environments with `python3-dev` headers for C++ compilation

### 2. Controller Node Missing slurmd
- **Problem**: Controller defined as compute node but missing worker daemon
- **Solution**: Install `slurmd` on controller since it's configured as both controller and worker

### 3. Hostname Resolution
- **Problem**: Lambda uses IP addresses as hostnames, breaking Slurm communication
- **Solution**: Set proper hostnames and add entries to `/etc/hosts` on both nodes

### 4. Munge Authentication Timing
- **Problem**: Worker connects before munge fully initializes
- **Solution**: Add delays and restart services in correct order (munge first, then slurmd)

### 5. SSH Configuration Distribution
- **Problem**: Worker can't SCP from controller without shared SSH keys
- **Solution**: Use NFS shared storage for configuration file distribution (learning only!)

### 6. Node State Management
- **Problem**: UNKNOWN vs FUTURE vs DOWN states require different commands
- **Solution**: Use appropriate `scontrol update` commands for each state

## Troubleshooting

### Nodes show DOWN or UNKNOWN

```bash
# For nodes in DOWN state:
sudo scontrol update nodename=slurm-worker state=resume

# For nodes in UNKNOWN state:
sudo scontrol update nodename=slurm-worker state=idle

# Check if both controller and worker have slurmd running:
sudo systemctl status slurmd  # Run on both nodes
```

### Munge authentication issues

```bash
# Verify keys match (run on both nodes)
sudo md5sum /etc/munge/munge.key  # Should be identical

# Check munge permissions (run on worker node)
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key

# Restart services in correct order (worker node)
sudo systemctl stop slurmd munge
sudo systemctl start munge
sleep 2
sudo systemctl start slurmd

# Test munge authentication
munge -n | unmunge
```

### NFS mount issues

```bash
showmount -e $CONTROLLER_IP
sudo umount /shared
sudo mount -t nfs $CONTROLLER_IP:/shared /shared
```

### Name resolution issues

```bash
# Controller can't reach worker - add to /etc/hosts on controller:
echo "$WORKER_IP slurm-worker" | sudo tee -a /etc/hosts

# Worker can't reach controller - add to /etc/hosts on worker:
echo "$CONTROLLER_IP slurm-controller" | sudo tee -a /etc/hosts

# Test resolution:
ping -c 3 slurm-controller  # from worker
ping -c 3 slurm-worker     # from controller
```

### GPU not detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify GRES config
sudo slurmd -C  # Should show GPU info
```

Perfect for hands-on sysadmin learning with real H100 hardware!

## Security Notes for Learning vs Production

### Learning Environment Shortcuts
This guide uses several approaches that are convenient for learning but **NOT suitable for production**:

1. **Open NFS exports** (`*` allows any IP to mount)
2. **no_root_squash** (remote root = local root)
3. **Sensitive files in shared storage** (munge key exposed on NFS)
4. **No firewall configuration**

### How Slurm Communication Actually Works
- **NOT SSH**: Slurm uses its own TCP protocol (ports 6817/6818)
- **Munge authentication**: Shared secret key provides cryptographic auth
- **Controller → Worker**: `slurmctld` sends jobs directly to `slurmd` daemons
- **No reverse access needed**: Workers don't initiate connections back to controller

### Production Best Practices
- Use SSH key-based authentication (controller has keys to workers)
- Configuration management tools (Ansible/Puppet/Salt)
- Proper firewall rules and network segmentation
- Dedicated management networks
- Never expose munge keys on shared storage
- Use `root_squash` on NFS exports
- Principle of least privilege
