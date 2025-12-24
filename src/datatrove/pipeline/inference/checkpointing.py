from __future__ import annotations

import asyncio
import dataclasses
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import orjson
from loguru import logger

from datatrove.data import Document
from datatrove.io import get_datafolder
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils._import_utils import check_required_dependencies


if TYPE_CHECKING:
    import aiosqlite


try:
    import xxhash  # type: ignore
except ImportError:
    xxhash = None


def _ensure_xxhash(checkpoint_dir: str | None) -> None:
    if checkpoint_dir is None or xxhash is not None:
        return
    check_required_dependencies("Inference request cache", ["xxhash"])
    import xxhash as _xxhash  # type: ignore

    globals()["xxhash"] = _xxhash


class RequestCache:
    """
    Lightweight sqlite-backed cache used to store individual request results for replay.
    """

    def __init__(self, checkpoints_local_dir: str | None):
        self.base_dir = checkpoints_local_dir
        self.enabled = checkpoints_local_dir is not None
        _ensure_xxhash(checkpoints_local_dir)
        self.db_path: str | None = None
        self._conn: aiosqlite.Connection | None = None
        self._doc_ids_in_cache: set[str] = set()
        self._queue: asyncio.Queue | None = None
        self._writer_task: asyncio.Task | None = None

    async def initialize(self, rank: int) -> None:
        if not self.enabled:
            return
        check_required_dependencies("Inference request cache", ["aiosqlite"])
        import aiosqlite

        os.makedirs(os.path.join(self.base_dir, f"{rank:05d}"), exist_ok=True)
        self.db_path = os.path.join(self.base_dir, f"{rank:05d}", "replay.sqlite3")
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS request_cache (
                chunk_index INTEGER NOT NULL,
                doc_id TEXT NOT NULL,
                rollout_idx INTEGER NOT NULL,
                payload_hash TEXT NOT NULL,
                result BLOB,
                error_message TEXT,
                PRIMARY KEY (doc_id, rollout_idx, payload_hash)
            )
        """
        )
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_request_cache_chunk ON request_cache(chunk_index)")
        await self._conn.commit()
        self._queue = asyncio.Queue()
        self._writer_task = asyncio.create_task(self._writer_loop())
        self._doc_ids_in_cache = await self._load_cached_doc_ids()

    async def close(self, delete_file: bool = False) -> None:
        if self._conn is None:
            return
        if self._queue is not None and self._writer_task is not None:
            await self.flush()
            await self._queue.put(None)
            await self._writer_task
            self._writer_task = None
            self._queue = None
        await self._conn.close()
        self._conn = None
        if delete_file and self.db_path and os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db_path = None
        self._doc_ids_in_cache.clear()

    async def _load_cached_doc_ids(self) -> set[str]:
        if self._conn is None:
            return set()
        cursor = await self._conn.execute("SELECT DISTINCT doc_id FROM request_cache")
        rows = await cursor.fetchall()
        await cursor.close()
        return {row[0] for row in rows}

    def prepare_payload(self, payload: dict[str, Any]) -> str:
        payload_bytes = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
        if xxhash is None:
            raise RuntimeError("xxhash is required for request caching")
        return xxhash.xxh128_hexdigest(payload_bytes)

    async def _writer_loop(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                op_type, data = item
                if self._conn is None:
                    continue
                if op_type == "result":
                    chunk_index, doc_id, rollout_idx, payload_hash, result = data
                    result_blob = await asyncio.to_thread(orjson.dumps, result)
                    await self._conn.execute(
                        "INSERT OR REPLACE INTO request_cache (chunk_index, doc_id, rollout_idx, payload_hash, result, error_message) VALUES (?, ?, ?, ?, ?, NULL)",
                        (chunk_index, doc_id, rollout_idx, payload_hash, result_blob),
                    )
                elif op_type == "error":
                    chunk_index, doc_id, rollout_idx, payload_hash, error_message = data
                    await self._conn.execute(
                        "INSERT OR REPLACE INTO request_cache (chunk_index, doc_id, rollout_idx, payload_hash, result, error_message) VALUES (?, ?, ?, ?, NULL, ?)",
                        (chunk_index, doc_id, rollout_idx, payload_hash, error_message),
                    )
                await self._conn.commit()
            except Exception as exc:
                logger.error(f"Failed to write request cache entry: {exc}")
            finally:
                self._queue.task_done()

    async def flush(self) -> None:
        if self._queue is None:
            return
        await self._queue.join()

    async def get_cached_response(
        self,
        doc_id: str,
        rollout_idx: int,
        *,
        payload_hash: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not self.enabled or self._conn is None or doc_id not in self._doc_ids_in_cache:
            return None, None

        cursor = await self._conn.execute(
            "SELECT result, error_message FROM request_cache WHERE doc_id = ? AND rollout_idx = ? AND payload_hash = ?",
            (doc_id, rollout_idx, payload_hash),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None, None
        result_blob, error_message = row
        return (orjson.loads(result_blob) if result_blob is not None else None, error_message)

    async def store_result(
        self,
        chunk_index: int,
        doc_id: str,
        rollout_idx: int,
        result: dict[str, Any],
        *,
        payload_hash: str,
    ) -> None:
        if not self.enabled or self._conn is None or self._queue is None:
            return
        await self._queue.put(("result", (chunk_index, doc_id, rollout_idx, payload_hash, result)))

    async def store_error(
        self,
        chunk_index: int,
        doc_id: str,
        rollout_idx: int,
        error_message: str,
        *,
        payload_hash: str,
    ) -> None:
        if not self.enabled or self._conn is None or self._queue is None:
            return
        await self._queue.put(("error", (chunk_index, doc_id, rollout_idx, payload_hash, error_message)))

    async def drop_chunk(self, chunk_index: int) -> None:
        if not self.enabled or self._conn is None:
            return
        await self.flush()
        await self._conn.execute("DELETE FROM request_cache WHERE chunk_index = ?", (chunk_index,))
        await self._conn.commit()

    async def mark_document_complete(self, doc_id: str) -> None:
        if not self.enabled:
            return
        self._doc_ids_in_cache.discard(doc_id)


class CheckpointManager:
    def __init__(
        self,
        checkpoints_local_dir: str | None = None,
        records_per_chunk: int = 6000,
        request_cache: RequestCache | None = None,
    ):
        """
        Manages checkpointing and chunking of documents.
        If checkpoints_local_dir is provided, it will save documents to it in chunks of records_per_chunk documents.
        If it's not provided, it will only write to the main output writer.
        """
        self.checkpoints_local_dir = checkpoints_local_dir if checkpoints_local_dir is not None else None
        self.checkpoints_local_dir_df = (
            get_datafolder(checkpoints_local_dir) if checkpoints_local_dir is not None else None
        )
        if self.checkpoints_local_dir_df is not None and not self.checkpoints_local_dir_df.is_local():
            raise ValueError("checkpoints_local_dir must be a local directory")
        if records_per_chunk <= 0:
            raise ValueError("records_per_chunk must be positive")
        self.records_per_chunk = records_per_chunk
        self.request_cache = request_cache

        self.file_locks = defaultdict(asyncio.Lock)
        self.checkpoint_file_lock = asyncio.Lock()
        self.per_chunk_counts = Counter()
        self.new_completed_chunks = set()
        self.last_chunk_index = -1

    async def write_document(self, document: Document, rank: int, chunk_index: int, output_writer_context: DiskWriter):
        """
        Write a document to the checkpoint and main output writer. Potentially closes the main file if the chunk is complete.
        """
        import aiofiles

        should_update_last_chunk_index = False
        async with self.file_locks[chunk_index]:
            # write to main output writer
            if "__no_rollouts_remove" not in document.metadata:
                output_writer_context.write(document, rank=rank, chunk_index=chunk_index)
            self.per_chunk_counts[chunk_index] += 1

            if self.checkpoints_local_dir is not None:
                # save to checkpoint/chunk
                save_path = os.path.join(self.checkpoints_local_dir, f"{rank:05d}/chunk_{chunk_index:05d}.jsonl")
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                if not os.path.exists(save_path):
                    logger.info(f"Creating checkpoint file {save_path}")
                async with aiofiles.open(save_path, "ab") as f:
                    await f.write(orjson.dumps(dataclasses.asdict(document), option=orjson.OPT_APPEND_NEWLINE))
                # see if we have to close the file
                if self.per_chunk_counts[chunk_index] == self.records_per_chunk:
                    # we gotta close the main file
                    output_writer_context.close_file(
                        output_writer_context._get_output_filename(document, rank, chunk_index=chunk_index)
                    )
                    self.new_completed_chunks.add(chunk_index)
                    should_update_last_chunk_index = True
        # can not be within the chunk lock
        if should_update_last_chunk_index:
            await self.update_last_chunk_index(rank)

    async def parse_existing_checkpoints(self, rank: int, output_writer_context: DiskWriter) -> tuple[int, set[str]]:
        """
        Load all checkpoints for a given rank and write them to the output writer.
        Returns:
        - documents to skip: number of documents from completed chunks that were already finished
        - set of ids of documents that were already processed in the unfinished chunks
        """
        all_ids = set()
        if not self.checkpoints_local_dir:
            return 0, all_ids

        should_update_last_chunk_index = False

        async with self.checkpoint_file_lock:
            if self.checkpoints_local_dir_df.exists(f"last_chunk/{rank:05d}.txt"):
                with self.checkpoints_local_dir_df.open(f"last_chunk/{rank:05d}.txt", "r") as f:
                    self.last_chunk_index = int(f.read().strip())

            reader = JsonlReader(self.checkpoints_local_dir, compression=None)
            # find existing chunk files and read from them
            for filename in self.checkpoints_local_dir_df.glob(f"{rank:05d}/*.jsonl"):
                chunk_index = int(filename.removeprefix(f"{rank:05d}/chunk_").removesuffix(".jsonl"))
                # not strictly needed but just to be safe for the future
                async with self.file_locks[chunk_index]:
                    for document in reader.read_file(filename):
                        if "__no_rollouts_remove" not in document.metadata:
                            output_writer_context.write(document, rank=rank, chunk_index=chunk_index)
                        all_ids.add(document.id)
                        self.per_chunk_counts[chunk_index] += 1
                        if self.per_chunk_counts[chunk_index] == self.records_per_chunk:
                            # close the file
                            output_writer_context.close_file(
                                output_writer_context._get_output_filename(document, rank, chunk_index=chunk_index)
                            )
                            self.new_completed_chunks.add(chunk_index)
                            # update the last chunk index/delete local file etc
                            should_update_last_chunk_index = True
        # can not be within the chunk lock
        if should_update_last_chunk_index:
            await self.update_last_chunk_index(rank)
        return (self.last_chunk_index + 1) * self.records_per_chunk if self.last_chunk_index >= 0 else 0, all_ids

    async def cleanup_last_chunk(self, rank: int, chunk_index: int):
        import shutil

        if self.checkpoints_local_dir is not None:
            self.new_completed_chunks.add(chunk_index)
            await self.update_last_chunk_index(rank)
            rank_dir = os.path.join(self.checkpoints_local_dir, f"{rank:05d}")
            # second part should be redundant as we technically only call this after everything completes but seems buggy for now
            if os.path.exists(rank_dir) and self.last_chunk_index == chunk_index:
                shutil.rmtree(rank_dir)

    async def update_last_chunk_index(self, rank: int):
        """
        Update the last chunk index and delete the local file if it's complete.
        """

        async with self.checkpoint_file_lock:
            # possibly multiple ones, in case file +2 finished before +1
            while self.last_chunk_index + 1 in self.new_completed_chunks:
                self.last_chunk_index += 1
                chunk_index = self.last_chunk_index
                async with self.file_locks[chunk_index]:
                    chunk_file = os.path.join(self.checkpoints_local_dir, f"{rank:05d}/chunk_{chunk_index:05d}.jsonl")
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                logger.info(f"Finished chunk {chunk_index}")
                # clean up
                self.file_locks.pop(chunk_index)
                self.per_chunk_counts.pop(chunk_index)
                self.new_completed_chunks.remove(chunk_index)
                if self.request_cache is not None:
                    await self.request_cache.drop_chunk(chunk_index)
                # save new last chunk index
                with self.checkpoints_local_dir_df.open(f"last_chunk/{rank:05d}.txt", "wt") as f:
                    f.write(str(self.last_chunk_index))

    def chunk_index_gen(self):
        ci = 0
        while True:
            for _ in range(self.records_per_chunk):
                yield ci
            ci += 1
