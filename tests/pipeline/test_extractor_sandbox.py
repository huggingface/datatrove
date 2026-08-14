import os
import time
import unittest

from datatrove.data import Document
from datatrove.pipeline.extractors.base import BaseExtractor


# extractors must be defined at module level to be picklable under the "spawn" start method (e.g. macOS)


class AlwaysFailExtractor(BaseExtractor):
    """Raises on every input, including the warmup text (https://github.com/huggingface/datatrove/issues/339)."""

    def __init__(self, timeout: float = 10):
        super().__init__(timeout)

    def extract(self, text: str) -> str:
        raise ValueError("extraction failed")


class MixedExtractor(BaseExtractor):
    """Fails only on documents containing "FAIL". Returns the worker pid to check worker reuse."""

    def __init__(self, timeout: float = 10):
        super().__init__(timeout)

    def extract(self, text: str) -> str:
        if "FAIL" in text:
            raise ValueError("doc-specific failure")
        return f"pid:{os.getpid()}"


class SlowExtractor(BaseExtractor):
    """Sleeps longer than the timeout on documents containing "SLOW"."""

    def __init__(self, timeout: float = 1):
        super().__init__(timeout)

    def extract(self, text: str) -> str:
        if "SLOW" in text:
            time.sleep(30)
        return "fast"


class CrashExtractor(BaseExtractor):
    """Hard-crashes the worker process on every document (simulates a segfault/OOM kill)."""

    def __init__(self, timeout: float = 10):
        super().__init__(timeout)

    def extract(self, text: str) -> str:
        if text == "":  # survive warmup
            return ""
        os._exit(1)


class WarmupCrashExtractor(BaseExtractor):
    """Hard-crashes the worker process during warmup."""

    def __init__(self, timeout: float = 10):
        super().__init__(timeout)

    def extract(self, text: str) -> str:
        os._exit(1)


class WarmupOnlyBrokenExtractor(BaseExtractor):
    """Raises on the empty warmup text but works on real documents."""

    def __init__(self, timeout: float = 10):
        super().__init__(timeout)

    def extract(self, text: str) -> str:
        if not text:
            raise ValueError("cannot handle empty html")
        return "extracted"


def get_docs(*texts):
    return [Document(text=text, id=str(i)) for i, text in enumerate(texts)]


class TestExtractorSandbox(unittest.TestCase):
    def get_stats(self, extractor):
        return {key: stat.total for key, stat in extractor.stats.stats.items()}

    def test_extraction_error_is_reported_per_document(self):
        """An extractor raising on every document (incl. warmup) must not burn the full timeout per document."""
        extractor = AlwaysFailExtractor(timeout=10)
        start = time.monotonic()
        docs = list(extractor.run(iter(get_docs("a", "b", "c"))))
        elapsed = time.monotonic() - start
        stats = self.get_stats(extractor)
        self.assertEqual(len(docs), 0)
        self.assertEqual(stats.get("clean_error"), 3)
        self.assertNotIn("timeout", stats)
        self.assertLess(elapsed, 8)  # before the fix: full 10s timeout per document

    def test_worker_reused_after_extraction_error(self):
        extractor = MixedExtractor(timeout=10)
        docs = list(extractor.run(iter(get_docs("a", "FAIL", "b"))))
        stats = self.get_stats(extractor)
        self.assertEqual(stats.get("extracted"), 2)
        self.assertEqual(stats.get("clean_error"), 1)
        # the error must not kill the worker: both successful documents are processed by the same process
        self.assertEqual(len({doc.text for doc in docs}), 1)

    def test_timeout_is_still_enforced(self):
        extractor = SlowExtractor(timeout=1)
        docs = list(extractor.run(iter(get_docs("a", "SLOW", "b"))))
        stats = self.get_stats(extractor)
        self.assertEqual(len(docs), 2)
        self.assertEqual(stats.get("timeout"), 1)
        self.assertEqual(stats.get("extracted"), 2)

    def test_worker_crash_detected_quickly(self):
        extractor = CrashExtractor(timeout=10)
        start = time.monotonic()
        docs = list(extractor.run(iter(get_docs("a", "b"))))
        elapsed = time.monotonic() - start
        stats = self.get_stats(extractor)
        self.assertEqual(len(docs), 0)
        self.assertEqual(stats.get("broken_process"), 2)
        self.assertLess(elapsed, 8)  # pipe EOF should report the death immediately, not after poll timeouts

    def test_worker_crash_during_warmup(self):
        extractor = WarmupCrashExtractor(timeout=10)
        start = time.monotonic()
        docs = list(extractor.run(iter(get_docs("a", "b"))))
        elapsed = time.monotonic() - start
        stats = self.get_stats(extractor)
        self.assertEqual(len(docs), 0)
        self.assertEqual(stats.get("broken_process"), 2)
        self.assertNotIn("timeout", stats)  # death during warmup is not a timeout
        self.assertLess(elapsed, 8)

    def test_warmup_error_does_not_break_extraction(self):
        """Extractors that raise on the empty warmup text must still work on real documents."""
        extractor = WarmupOnlyBrokenExtractor(timeout=10)
        docs = list(extractor.run(iter(get_docs("a", "b", "c"))))
        stats = self.get_stats(extractor)
        self.assertEqual(len(docs), 3)
        self.assertEqual(stats.get("extracted"), 3)
