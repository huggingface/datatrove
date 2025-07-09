import base64
import concurrent.futures
import glob
import gzip
import hashlib
import json
import logging
import os
import posixpath
import time
from boto3.session import Session
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
from datatrove.pipeline.base import Document
import boto3
import requests  # type: ignore
import zstandard as zstd
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
import aiobotocore.client
from loguru import logger

def parse_s3_path(s3_path: str) -> tuple[str, str]:
    if not (s3_path.startswith("s3://") or s3_path.startswith("gs://") or s3_path.startswith("weka://")):
        raise ValueError("s3_path must start with s3://, gs://, or weka://")
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key

def get_s3_bytes(s3_client, s3_path: str, start_index: Optional[int] = None, end_index: Optional[int] = None) -> bytes:
    # Fall back for local files
    if os.path.exists(s3_path):
        assert start_index is None and end_index is None, "Range query not supported yet"
        with open(s3_path, "rb") as f:
            return f.read()

    bucket, key = parse_s3_path(s3_path)

    # Build the range header if start_index and/or end_index are specified
    range_header = None
    if start_index is not None and end_index is not None:
        # Range: bytes=start_index-end_index
        range_value = f"bytes={start_index}-{end_index}"
        range_header = {"Range": range_value}
    elif start_index is not None and end_index is None:
        # Range: bytes=start_index-
        range_value = f"bytes={start_index}-"
        range_header = {"Range": range_value}
    elif start_index is None and end_index is not None:
        # Range: bytes=-end_index (last end_index bytes)
        range_value = f"bytes=-{end_index}"
        range_header = {"Range": range_value}

    if range_header:
        obj = s3_client.get_object(Bucket=bucket, Key=key, Range=range_header["Range"])
    else:
        obj = s3_client.get_object(Bucket=bucket, Key=key)

    return obj["Body"].read()


def get_s3_bytes_with_backoff(s3_client, pdf_s3_path, start_index: Optional[int] = None, end_index: Optional[int] = None, max_retries: int = 8, backoff_factor: int = 2):
    attempt = 0

    while attempt < max_retries:
        try:
            return get_s3_bytes(s3_client, pdf_s3_path, start_index, end_index)
        except ClientError as e:
            # Check for some error kinds AccessDenied error and raise immediately
            if e.response["Error"]["Code"] in ("AccessDenied", "NoSuchKey"):
                logger.error(f"{e.response['Error']['Code']} error when trying to access {pdf_s3_path}: {e}")
                raise
            else:
                wait_time = backoff_factor**attempt
                logger.warning(f"Attempt {attempt+1} failed to get_s3_bytes for {pdf_s3_path}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1
        except Exception as e:
            wait_time = backoff_factor**attempt
            logger.warning(f"Attempt {attempt+1} failed to get_s3_bytes for {pdf_s3_path}: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            attempt += 1

    logger.error(f"Failed to get_s3_bytes for {pdf_s3_path} after {max_retries} retries.")
    raise Exception("Failed to get_s3_bytes after retries")



