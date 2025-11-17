import asyncio

from datatrove.pipeline.inference.checkpointing import RequestCache


def test_request_cache_store_and_fetch(tmp_path):
    async def _run():
        cache = RequestCache(str(tmp_path))
        await cache.initialize(rank=0)

        payload = {"prompt": "hello"}
        result = {"text": "world", "finish_reason": "stop", "usage": {"prompt_tokens": 1}}

        # no cached entry for doc-1 initially
        payload_hash = cache.prepare_payload(payload)
        assert await cache.get_cached_response("doc-1", 0, payload_hash=payload_hash) == (None, None)

        await cache.store_result(0, "doc-1", 0, result, payload_hash=payload_hash)
        await cache.store_error(0, "doc-2", 0, "boom", payload_hash=payload_hash)
        await cache.flush()

        resume_cache = RequestCache(str(tmp_path))
        await resume_cache.initialize(rank=0)

        cached_result, cached_error = await resume_cache.get_cached_response("doc-1", 0, payload_hash=payload_hash)
        assert cached_error is None
        assert cached_result == result
        cached_result, cached_error = await resume_cache.get_cached_response("doc-2", 0, payload_hash=payload_hash)
        assert cached_result is None
        assert cached_error == "boom"

        await resume_cache.mark_document_complete("doc-1")
        assert await resume_cache.get_cached_response("doc-1", 0, payload_hash=payload_hash) == (None, None)

        await resume_cache.drop_chunk(0)
        await resume_cache.close(delete_file=True)
        await cache.close()
        assert not list(tmp_path.glob("*_replay.sqlite3"))

    asyncio.run(_run())
