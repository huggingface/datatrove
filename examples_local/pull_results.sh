#!/bin/bash
# Pull FinePDFs pipeline results from Lambda for local review

# Configuration - set LAMBDA_HOST environment variable before running
# Example: export LAMBDA_HOST="ubuntu@your-host"
if [ -z "$LAMBDA_HOST" ]; then
    echo "Error: LAMBDA_HOST environment variable not set"
    echo "Usage: export LAMBDA_HOST='ubuntu@your-host' && ./pull_results.sh"
    exit 1
fi

REMOTE_DIR="datatrove/examples_local/output/finepdfs_local"
LOCAL_DIR="examples_local/output/finepdfs_local"

echo "=========================================="
echo "Pulling FinePDFs Results from Lambda"
echo "=========================================="
echo "Host: $LAMBDA_HOST"
echo

# Create local directory structure
mkdir -p "$LOCAL_DIR"

# Pull all outputs maintaining structure
echo "📥 Downloading all files (PDFs, PNGs, JSONL)..."
scp -r "$LAMBDA_HOST:$REMOTE_DIR"/* "$LOCAL_DIR/"

echo
echo "✅ Files downloaded to: $LOCAL_DIR"
echo
echo "Directory structure:"
ls -lR "$LOCAL_DIR"
