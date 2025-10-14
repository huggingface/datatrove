# Running DataTrove on Manual Slurm Cluster

## Prerequisites
- Working Slurm cluster (see lambda_manual_slurm_setup.md) with NFS shared storage at `/shared/`
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

## Step 3: Configure Scripts for Shared Storage

The examples in `spec/phase2/examples/` use `/tmp/` paths by default. For manual clusters, modify the path constants at the top of each file:

```bash
# On controller node, edit the constants at the top of each example file
vim spec/phase2/examples/01_basic_filtering_slurm.py

# Change the configuration section:
# OUTPUT_DIR = "/tmp/output"           -> OUTPUT_DIR = "/shared/output"
# LOGS_DIR = "/tmp/logs"               -> LOGS_DIR = "/shared/logs"
# SLURM_LOGS_DIR = "/tmp/slurm_logs"   -> SLURM_LOGS_DIR = "/shared/slurm_logs"
```

**Why this is needed**: `/tmp/` is local to each node. On manual clusters without automatic synchronization, all output paths must use shared storage like `/shared/` so all nodes can access the same files.

## Step 4: Run DataTrove Examples

After updating the path constants, run the examples:

```bash
# On controller node
cd ~/datatrove
source ~/venv/bin/activate

# Run basic filtering example
python spec/phase2/examples/01_basic_filtering_slurm.py

# Run distributed statistics example
python spec/phase2/examples/04_statistics_slurm.py
```

Monitor execution:
```bash
# Check job queue
squeue

# Check job history
sacct --format=JobID,JobName,Partition,State,NodeList

# Watch queue in real-time
watch squeue
```

## Key Files Created

After successful runs, check:
- `/shared/output/*.jsonl.gz` - Processed data
- `/shared/logs/stats.json` - Processing statistics
- `/shared/slurm_logs/*.out` - Job logs from both nodes
- `/shared/logs/logs/task_*.log` - Individual task logs

## Workflow Summary

1. **Clone repo** on controller node with DataTrove examples
2. **Edit path constants** in example files to use `/shared/` instead of `/tmp/`
3. **Run examples** from controller node
4. **Monitor** with `squeue` and check logs in `/shared/slurm_logs/`
5. **Results** saved to `/shared/output/`

**Key principle**: Any paths that need to be accessed by multiple nodes must use shared storage (`/shared/`), not local storage (`/tmp/`).