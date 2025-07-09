import asyncio
import base64
import dataclasses
import os
import tempfile
from collections import Counter
from string import Template
from typing import List
import boto3
import orjson
from datatrove.data import Document


class AsyncS3JsonlWriter:
    """
    Async JSONL writer that batches documents and writes to S3 when thresholds are met.
    
    Args:
        s3_bucket: S3 bucket name
        s3_key_template: S3 key template, can contain placeholders like ${rank}
        max_file_size: Maximum file size in bytes before writing to S3
        max_records: Maximum number of records before writing to S3
        compression: Compression type ("gzip" or None)
        save_media_bytes: Whether to save media bytes in the output
    """
    
    def __init__(
        self,
        s3_bucket: str,
        s3_key_template: str = "${rank}.jsonl",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        max_records: int = 10000,
        compression: str = "gzip",
        save_media_bytes: bool = False,
    ):
        self.s3_bucket = s3_bucket
        self.s3_key_template = Template(s3_key_template)
        self.max_file_size = max_file_size
        self.max_records = max_records
        self.compression = compression
        self.save_media_bytes = save_media_bytes
        
        # Internal state
        self.documents_buffer: List[Document] = []
        self.current_size = 0
        self.file_id_counter = Counter()
        self.s3_client = boto3.client('s3')
        
        # Add .gz extension if using gzip compression
        if self.compression == "gzip" and not s3_key_template.endswith(".gz"):
            self.s3_key_template = Template(s3_key_template + ".gz")
    
    def _document_to_dict(self, document: Document) -> dict:
        """Convert document to dictionary format, similar to JsonlWriter._default_adapter"""
        data = {key: val for key, val in dataclasses.asdict(document).items() if val}
        
        if not self.save_media_bytes and "media" in data:
            data["media"] = [
                {
                    **media,
                    "media_bytes": None,
                }
                for media in data["media"]
            ]
        else:
            # Encode media bytes as base64 if present (like JsonlWriter does)
            for media in data.get("media", []):
                if media.get("media_bytes"):
                    media["media_bytes"] = base64.b64encode(media["media_bytes"]).decode("ascii")
        
        return data
    
    def _get_s3_key(self, rank: int = 0, **kwargs) -> str:
        """Get the S3 key for the current file, similar to DiskWriter._get_output_filename"""
        base_key = self.s3_key_template.substitute({"rank": str(rank).zfill(5), **kwargs})
        file_id = self.file_id_counter[base_key]
        
        if file_id > 0:
            # Add file counter prefix like DiskWriter does
            if "/" in base_key:
                dir_part, file_part = base_key.rsplit("/", 1)
                return f"{dir_part}/{file_id:03d}_{file_part}"
            else:
                return f"{file_id:03d}_{base_key}"
        
        return base_key
    
    def _write_temp_file(self) -> str:
        """Write buffered documents to a temporary file and return the path"""
        # Create temporary file
        suffix = ".gz" if self.compression == "gzip" else ".jsonl"
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
        
        try:
            with os.fdopen(temp_fd, 'wb') as f:
                if self.compression == "gzip":
                    import gzip
                    with gzip.open(f, 'wb') as gz_f:
                        for doc in self.documents_buffer:
                            doc_dict = self._document_to_dict(doc)
                            gz_f.write(orjson.dumps(doc_dict, option=orjson.OPT_APPEND_NEWLINE))
                else:
                    for doc in self.documents_buffer:
                        doc_dict = self._document_to_dict(doc)
                        f.write(orjson.dumps(doc_dict, option=orjson.OPT_APPEND_NEWLINE))
        except:
            # Clean up temp file if writing fails
            os.unlink(temp_path)
            raise
            
        return temp_path
    
    async def _flush_to_s3(self, rank: int = 0, **kwargs):
        """Write buffered documents to S3 using upload_file for multipart uploads"""
        if not self.documents_buffer:
            return
        
        s3_key = self._get_s3_key(rank, **kwargs)
        
        # Write to temporary file
        temp_path = await asyncio.to_thread(self._write_temp_file)
        
        try:
            # Upload using upload_file (supports multipart automatically)
            await asyncio.to_thread(
                self.s3_client.upload_file,
                temp_path,
                self.s3_bucket,
                s3_key
            )
        finally:
            # Always clean up the temporary file
            os.unlink(temp_path)
        
        # Clear buffer and reset state
        self.documents_buffer.clear()
        self.current_size = 0
        
        # Increment file counter for next file
        base_key = self.s3_key_template.substitute({"rank": str(rank).zfill(5), **kwargs})
        self.file_id_counter[base_key] += 1
    
    async def write(self, document: Document, rank: int = 0, **kwargs):
        """
        Write a document to the buffer, flushing to S3 when thresholds are met.
        
        Args:
            document: Document to write
            rank: Rank for filename templating
            **kwargs: Additional template variables
        """
        # Add document to buffer
        self.documents_buffer.append(document)
        
        # Estimate size (approximate, like the size checking in DiskWriter)
        doc_dict = self._document_to_dict(document)
        doc_size = len(orjson.dumps(doc_dict))
        self.current_size += doc_size
        
        # Check if we need to flush (similar to DiskWriter logic)
        should_flush = (
            (self.max_records > 0 and len(self.documents_buffer) >= self.max_records) or
            (self.max_file_size > 0 and self.current_size >= self.max_file_size)
        )
        
        if should_flush:
            await self._flush_to_s3(rank, **kwargs)
    
    async def flush(self, rank: int = 0, **kwargs):
        """Explicitly flush any remaining documents to S3"""
        await self._flush_to_s3(rank, **kwargs)
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - flush any remaining documents"""
        await self.flush()