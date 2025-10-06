# Lambda Labs GPU Server Setup Guide
## DataTrove + Docling + RolmOCR Pipeline

This guide documents the complete setup process for a Lambda Labs GPU server to run the FinePDFs PDF processing pipeline with Docling (text extraction) and RolmOCR (OCR extraction).

---

## Prerequisites

- Lambda Labs GPU instance (with NVIDIA GPU)
- Ubuntu 22.04 or similar
- SSH access to the server

---

## 1. System Setup

### 1.1 Install System Dependencies

```bash
sudo apt update
sudo apt install -y git curl wget build-essential
```

### 1.2 Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, accept license, allow conda init
source ~/.bashrc
```

### 1.3 Install NVIDIA Drivers (if not present)

```bash
# Check if GPU is detected
nvidia-smi

# If not working, install drivers
sudo apt update
sudo apt install -y nvidia-utils-535-server nvidia-driver-535-server
sudo reboot

# After reboot, verify
nvidia-smi
```

---

## 2. Clone Repositories

### 2.1 Clone DataTrove

```bash
cd ~
git clone https://github.com/yoniebans/datatrove.git
cd datatrove
git checkout learning/phase3-routing-pipeline
```

### 2.2 Clone Docling-sync (for optimized models)

```bash
cd ~
git clone https://github.com/yoniebans/Docling-sync.git
```

**Important:** This provides the OpenVINO quantized layout model (`v2-quant.xml`) which is significantly faster than the default model.

---

## 3. Create Conda Environment

### 3.1 Accept Conda TOS (if needed)

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 3.2 Create Environment

```bash
conda create -n datatrove-docling python=3.12 -y
conda activate datatrove-docling
```

---

## 4. Install Python Dependencies

### 4.1 Install DataTrove

```bash
cd ~/datatrove
pip install -e ".[dev,all]"
```

This installs all dependencies including:
- Core datatrove packages
- Docling (official release)
- LMDeploy (for RolmOCR)
- All processing tools

### 4.2 Install Additional Dependencies

```bash
pip install openvino zstandard warcio s3fs pymupdf orjson xgboost
```

**Package purposes:**
- `openvino`: OpenVINO runtime for fast CPU inference (Docling)
- `zstandard`: Compression support
- `warcio`: WARC file processing
- `s3fs`: S3 filesystem access
- `pymupdf`: PDF manipulation
- `orjson`: Fast JSON parsing
- `xgboost`: ML model for PDF routing

### 4.3 Install LMDeploy for RolmOCR

```bash
pip install lmdeploy[all]
```

This provides GPU-accelerated inference for the RolmOCR vision model.

---

## 5. Configure Docling with Optimized Layout Model

### 5.1 Set Environment Variable

Add this to your `~/.bashrc` (or set before each run):

```bash
export LAYOUT_VINO_PATH="../Docling-sync/models/v2-quant.xml"
```

**Why this matters:**
- Default Docling uses a slower layout detection model
- The quantized OpenVINO model from Docling-sync is ~4x faster
- Must be set before running any pipeline that uses DoclingExtractor

### 5.2 Verify Setup

```bash
conda activate datatrove-docling
export LAYOUT_VINO_PATH="../Docling-sync/models/v2-quant.xml"
cd ~/datatrove

python -c "
from src.datatrove.pipeline.media.extractors.extractors import DoclingExtractor
import sys
try:
    extractor = DoclingExtractor(timeout=60)
    print('✅ DoclingExtractor initialized successfully!')
    print('✅ OpenVINO compatibility confirmed')
    sys.exit(0)
except Exception as e:
    print(f'❌ DoclingExtractor failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
```

---

## 6. Configure AWS Credentials (for CommonCrawl Access)

### 6.1 Configure Credentials

```bash
aws configure
# Enter:
#   AWS Access Key ID: <your-key>
#   AWS Secret Access Key: <your-secret>
#   Default region: us-east-1
#   Default output format: json
```

**Note:** CommonCrawl S3 data is free to access, but you need AWS credentials for S3 API access (directory listing with glob patterns).

### 6.2 Install S3 Dependencies

```bash
pip install 's3fs[boto3]' --upgrade
```

**Important:** Do NOT install `awscli` via pip if you already have `s3fs` installed - it causes version conflicts with `aiobotocore`. Use the system `aws` command or configure credentials manually via `~/.aws/credentials`.

---

## 7. Test the Setup

### 7.1 Test GPU/CUDA

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA available: True
CUDA devices: 1
GPU name: NVIDIA A10 (or similar)
```

### 7.2 Test Docling (Text Extraction)

```bash
cd ~/datatrove
conda activate datatrove-docling
export LAYOUT_VINO_PATH="../Docling-sync/models/v2-quant.xml"

python examples_local/test_local_pdfs.py
```

### 7.3 Test RolmOCR (OCR Extraction)

```bash
python examples_local/test_rolmocr.py
```

### 7.4 Test Full Routing Pipeline (Local PDFs)

```bash
python examples_local/test_finepdfs_local.py
```

### 7.5 Test WARC Processing

```bash
python examples_local/test_finepdfs_warc.py
```

### 7.6 Test HTTPS CommonCrawl Streaming (No AWS needed)

```bash
python examples_local/test_finepdfs_https.py
```

### 7.7 Test S3 CommonCrawl Access (Requires AWS credentials)

```bash
python examples/finepdfs.py
```

---

## 8. Daily Workflow

### 8.1 Start Working Session

```bash
# SSH into Lambda
ssh ubuntu@<your-lambda-ip>

# Activate environment
conda activate datatrove-docling

# Set Docling optimization
export LAYOUT_VINO_PATH="../Docling-sync/models/v2-quant.xml"

# Navigate to project
cd ~/datatrove

# Pull latest changes
git pull
```

### 8.2 Run Production Pipeline

```bash
# Edit configuration in examples/finepdfs.py:
# - DUMP_TO_PROCESS: Which CommonCrawl dump
# - PDF_LIMIT: Number of PDFs to process
# - MAIN_OUTPUT_PATH: Where to save results

python examples/finepdfs.py
```

---

## 9. Directory Structure

```
~/
├── datatrove/                          # Main repo
│   ├── src/datatrove/                  # Source code
│   ├── examples/                       # Production scripts
│   │   └── finepdfs.py                # Main production pipeline
│   ├── examples_local/                 # Test scripts
│   │   ├── test_local_pdfs.py         # Test Docling
│   │   ├── test_rolmocr.py            # Test RolmOCR
│   │   ├── test_finepdfs_local.py     # Test full pipeline (local)
│   │   ├── test_finepdfs_warc.py      # Test WARC processing
│   │   ├── test_finepdfs_https.py     # Test HTTPS streaming
│   │   ├── data/                       # Test data
│   │   ├── output/                     # Test outputs
│   │   └── logs/                       # Test logs
│   └── output/                         # Production outputs
│
└── Docling-sync/                       # Optimized Docling models
    └── models/
        └── v2-quant.xml                # Fast OpenVINO layout model
```

---

## 10. Common Commands

### Clean up test outputs
```bash
rm -rf examples_local/logs examples_local/output
```

### Clean up completion markers (to re-run failed tasks)
```bash
rm -rf examples_local/logs/*/completions/
```

### Check output files
```bash
# List outputs
ls -lh examples_local/output/finepdfs_local/*/

# Count documents
zcat examples_local/output/finepdfs_local/classified/*.jsonl.gz | wc -l
zcat examples_local/output/finepdfs_local/text_extraction/*.jsonl.gz | wc -l
zcat examples_local/output/finepdfs_local/ocr_extraction/*.jsonl.gz | wc -l

# View document IDs
zcat examples_local/output/finepdfs_local/classified/*.jsonl.gz | jq -r '.id'

# View extracted text
zcat examples_local/output/finepdfs_local/text_extraction/*.jsonl.gz | jq -r '.text' | head -50
```

### Check stats
```bash
cat examples_local/logs/finepdfs_local/classification/stats/*.json | jq '.'
```

---

## 11. Troubleshooting

### Issue: CUDA not available

**Solution:**
```bash
# Check driver installation
nvidia-smi

# Reinstall if needed
sudo apt install -y nvidia-driver-535-server
sudo reboot
```

### Issue: DoclingExtractor fails to initialize

**Solution:**
```bash
# Ensure environment variable is set
export LAYOUT_VINO_PATH="../Docling-sync/models/v2-quant.xml"

# Verify path exists
ls -la ../Docling-sync/models/v2-quant.xml
```

### Issue: S3 access denied

**Solution:**
```bash
# Reconfigure AWS credentials
aws configure

# Test S3 access
python examples_local/test_cc_s3_access.py
```

### Issue: Version conflicts with aiobotocore/botocore

**Solution:**
```bash
# Uninstall awscli if installed via pip
pip uninstall awscli -y

# Reinstall s3fs with proper dependencies
pip install 's3fs[boto3]' --upgrade
```

### Issue: LMDeploy errors during OCR

**Expected behavior:** Some LMDeploy internal errors are normal and handled by retry logic. If documents complete successfully, the pipeline is working correctly.

Check stats to verify:
```bash
cat examples_local/logs/*/ocr_extraction/stats/*.json | jq '.successful_documents'
```

---

## 12. Performance Notes

### Docling (CPU)
- With OpenVINO quantized model: ~1-2 seconds per page
- Without optimization: ~5-10 seconds per page
- **Always set `LAYOUT_VINO_PATH`**

### RolmOCR (GPU)
- With A10 GPU: ~5-10 seconds per page
- Depends on image complexity and resolution
- Some internal errors are normal and retried automatically

### WARC Processing
- ~60-90 seconds per WARC file
- Each WARC contains ~50-100 PDFs (varies)
- Use `PDF_LIMIT` for testing to avoid long runs

---

## 13. Quick Reference

### Essential Environment Setup (run every session)
```bash
conda activate datatrove-docling
export LAYOUT_VINO_PATH="../Docling-sync/models/v2-quant.xml"
cd ~/datatrove
```

### Test Suite
```bash
# Quick smoke tests
python examples_local/test_local_pdfs.py        # Docling
python examples_local/test_rolmocr.py           # RolmOCR

# Full pipeline tests
python examples_local/test_finepdfs_local.py    # Local PDFs
python examples_local/test_finepdfs_warc.py     # WARC files
python examples_local/test_finepdfs_https.py    # HTTPS streaming

# Production
python examples/finepdfs.py                     # S3 + glob patterns
```

### Key Files to Configure

**For testing:**
- `examples_local/test_finepdfs_https.py`: `NUM_WARC_FILES`, `DUMP_TO_PROCESS`

**For production:**
- `examples/finepdfs.py`: `DUMP_TO_PROCESS`, `PDF_LIMIT`, `MAIN_OUTPUT_PATH`, `NUM_TASKS`

---

## 14. Cost Optimization

### CommonCrawl Access
- **HTTPS streaming**: Free, no credentials needed
- **S3 with credentials**: Free if accessing from us-east-1 (same region as CommonCrawl)
- **Cross-region**: ~$0.09/GB data transfer

### Lambda Labs Costs
- Charged per hour for GPU instance
- Stop instance when not in use
- Consider using cheaper CPU-only instance for Docling-only workloads

---

## 15. Next Steps

Once setup is complete:

1. **Test with small batches** (`PDF_LIMIT=100`)
2. **Verify outputs** look correct
3. **Scale up gradually**
4. **Monitor costs** if processing large dumps
5. **Consider Slurm** for massive parallel processing

---

## Support

For issues or questions:
- DataTrove: https://github.com/huggingface/datatrove
- Docling: https://github.com/DS4SD/docling
- Docling-sync: https://github.com/yoniebans/Docling-sync
- RolmOCR: https://huggingface.co/Reducto/RolmOCR
