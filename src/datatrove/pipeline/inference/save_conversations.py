import asyncio
import base64
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import boto3
from botocore.config import Config
from loguru import logger

from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.pipeline.base import Document, PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.inference.preprocessing.query_preparator import QueryPreparator
from datatrove.pipeline.inference.preprocessing.readers import read_warc_bytes, read_zstd_bytes


@dataclass
class ConversationConfig:
    max_concurrent_tasks: int = 50
    resize_longest_side_pixels: Optional[int] = None
    max_image_tokens: int = 4096
    query_text: str = "Return the plain text representation of this document as if you were reading it naturally.\n"


class ConversationSaver(PipelineStep):
    name = "Conversation Saver 💾"
    type = "Data Collection 📂"

    def __init__(self,
                 records_readers: BaseDiskReader,
                 media_path: str,
                 config: ConversationConfig,
                 output_file: str,
                 media_reader: str = "warc",
                 ):
        super().__init__()
        self.records_readers = records_readers
        self.config = config
        self.media_path = media_path
        self.output_file = output_file
        self.media_reader = media_reader
        self.conversations: List[Dict[str, Any]] = []

    def create_conversation_entry(self, document_id: str, page_images_b64: List[str]) -> Dict[str, Any]:
        """Create a conversation entry in the specified JSON format"""
        conversations = []
        
        # Create human message with images and text
        for page_image_b64 in page_images_b64:
            conversations.append(
                {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{page_image_b64}", "max_pixels": self.config.max_image_tokens * 28 * 28, "min_pixels": 56 * 28 * 28},
                            },
                            {
                                "type": "text",
                                "text": self.config.query_text
                            }
                        ]
                    },
                ]
                }
            )
        
        return {
            "id": document_id,
            "conversations": conversations
        }

    async def process_pdf(self, document: Document, s3_client, query_preparator: QueryPreparator):
        """Process PDF and extract page images for conversation saving"""
        try:
            if self.media_reader == "warc":
                pdf_data, length = await asyncio.to_thread(
                    read_warc_bytes, 
                    s3_client, 
                    f"{self.media_path}/{document.metadata['warc_filename']}", 
                    document.metadata["warc_record_offset"]
                )
            elif self.media_reader == "zstd":
                pdf_data, length = await asyncio.to_thread(
                    read_zstd_bytes, 
                    s3_client, 
                    f"{self.media_path}/{document.media[0].path}", 
                    document.media[0].offset
                )
            else:
                raise ValueError(f"Unsupported media reader: {self.media_reader}")

            num_pages, language, pdf_query_iterator = await query_preparator.process(
                pdf_data, length, image_rotation=0, id=document.id
            )
            
            page_images = []
            async for page_num, page_image_b64, page_text in pdf_query_iterator:
                page_images.append(page_image_b64)
            
            return page_images
            
        except Exception as e:
            logger.exception(f"Exception occurred while processing document {document.id}")
            return []

    async def process_document(self, document: Document, s3_client, query_preparator: QueryPreparator):
        """Process a single document and create conversation entry"""
        try:
            page_images = await self.process_pdf(document, s3_client, query_preparator)
            if page_images:
                conversation_entry = self.create_conversation_entry(document.id, page_images)
                self.conversations.append(conversation_entry)
                self.stat_update("processed_documents", value=1, unit="documents")
                self.stat_update("processed_pages", value=len(page_images), unit="pages")
            else:
                self.stat_update("failed_documents", value=1, unit="documents")
                logger.warning(f"No pages extracted for document {document.id}")
        except Exception as e:
            logger.exception(f"Exception occurred while processing document {document.id}")
            self.stat_update("failed_documents", value=1, unit="documents")

    def save_conversations_to_file(self):
        """Save all conversations to JSON file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.conversations)} conversations to {self.output_file}")
        except Exception as e:
            logger.exception(f"Failed to save conversations to {self.output_file}")

    async def run_async(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        client_config = Config(
            max_pool_connections=self.config.max_concurrent_tasks + 10,
        )
        workspace_session = boto3.Session()
        workspace_s3 = workspace_session.client("s3", config=client_config)
        
        logger.info(f"Starting conversation saving pipeline with PID {os.getpid()}")
        
        tasks_pool = set()
        
        with self.track_time("total_time"):
            async with QueryPreparator(
                resize_longest_side_pixels=self.config.resize_longest_side_pixels, 
                max_visual_tokens=self.config.max_image_tokens, 
                is_zstd=self.media_reader == "zstd"
            ) as query_preparator:
                
                for record in self.records_readers.run(rank=rank, world_size=world_size):
                    # Limit concurrent tasks
                    while len(tasks_pool) >= self.config.max_concurrent_tasks:
                        results, tasks_pool = await asyncio.wait(tasks_pool, return_when=asyncio.FIRST_COMPLETED)
                        for result in results:
                            try:
                                await result  # Just wait for completion, no return value needed
                            except Exception as e:
                                logger.exception(f"Task failed: {e}")

                    new_future = asyncio.create_task(
                        self.process_document(record, workspace_s3, query_preparator)
                    )
                    tasks_pool.add(new_future)
        
                # Wait for remaining tasks
                while tasks_pool:
                    results, tasks_pool = await asyncio.wait(tasks_pool, return_when=asyncio.FIRST_COMPLETED)
                    for result in results:
                        try:
                            await result
                        except Exception as e:
                            logger.exception(f"Task failed: {e}")

        # Save all conversations to file
        self.save_conversations_to_file()
        
        logger.info(f"Conversation saving pipeline completed. Total conversations: {len(self.conversations)}")

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        asyncio.run(self.run_async(data, rank, world_size))
        return data



