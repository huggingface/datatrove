"""Tests for vLLM compilation locking mechanism.

This tests the file-based locking that prevents concurrent torch.compile cache
corruption when multiple SLURM jobs with the same config start simultaneously.
"""

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from datatrove.pipeline.inference.servers.compile_lock import CompileLockManager, compute_config_hash


class TestComputeConfigHash(unittest.TestCase):
    """Tests for the compute_config_hash function."""

    def test_deterministic(self):
        """Test that the same inputs produce the same hash."""
        hash1 = compute_config_hash("model-a", tp=2, pp=1, dp=1, model_max_context=4096)
        hash2 = compute_config_hash("model-a", tp=2, pp=1, dp=1, model_max_context=4096)

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 12)

    def test_different_configs_produce_different_hashes(self):
        """Test that different configs produce different hashes."""
        hashes = {
            compute_config_hash("model-a", 1, 1, 1, 2048),
            compute_config_hash("model-b", 1, 1, 1, 2048),  # different model
            compute_config_hash("model-a", 2, 1, 1, 2048),  # different tp
            compute_config_hash("model-a", 1, 2, 1, 2048),  # different pp
            compute_config_hash("model-a", 1, 1, 2, 2048),  # different dp
            compute_config_hash("model-a", 1, 1, 1, 4096),  # different context
            compute_config_hash("model-a", 1, 1, 1, 2048, {"quant": "fp8"}),  # with kwargs
        }
        self.assertEqual(len(hashes), 7)

    def test_model_kwargs_order_independent(self):
        """Test that model_kwargs hash is order-independent (sorted)."""
        hash1 = compute_config_hash("model", 1, 1, 1, 2048, model_kwargs={"a": 1, "b": 2})
        hash2 = compute_config_hash("model", 1, 1, 1, 2048, model_kwargs={"b": 2, "a": 1})

        self.assertEqual(hash1, hash2)

    def test_empty_vs_none_model_kwargs(self):
        """Test that empty dict and None produce the same hash."""
        hash1 = compute_config_hash("model", 1, 1, 1, 2048, model_kwargs={})
        hash2 = compute_config_hash("model", 1, 1, 1, 2048, model_kwargs=None)

        self.assertEqual(hash1, hash2)


class TestCompileLockManager(unittest.TestCase):
    """Tests for the CompileLockManager class."""

    def setUp(self):
        """Set up test fixtures with a temporary lock directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.lock_dir = Path(self.temp_dir) / "locks"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_lock(self, config_hash: str = "test123") -> CompileLockManager:
        """Create a CompileLockManager instance."""
        return CompileLockManager(config_hash, self.lock_dir)

    def test_cache_exists(self):
        """Test cache_exists returns correct value based on marker file."""
        lock = self._create_lock()

        self.assertFalse(lock.cache_exists())

        self.lock_dir.mkdir(parents=True, exist_ok=True)
        lock.marker_path.touch()

        self.assertTrue(lock.cache_exists())

    def test_acquire_creates_lock_dir_and_file(self):
        """Test that acquiring a lock creates the directory and lock file."""
        lock = self._create_lock()
        self.assertFalse(self.lock_dir.exists())

        result = lock.acquire()

        self.assertTrue(result)
        self.assertTrue(self.lock_dir.exists())
        self.assertTrue(lock.lock_path.exists())
        lock.release()

    def test_acquire_skips_when_cache_exists(self):
        """Test that acquisition is skipped when cache marker exists."""
        lock = self._create_lock()
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        lock.marker_path.touch()

        result = lock.acquire()

        self.assertFalse(result)

    def test_mark_complete_creates_marker(self):
        """Test that mark_complete creates the .done file."""
        lock = self._create_lock()
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        lock.mark_complete()

        self.assertTrue(lock.marker_path.exists())

    def test_release_is_idempotent(self):
        """Test that releasing multiple times doesn't raise."""
        lock = self._create_lock()
        lock.acquire()
        lock.release()
        lock.release()  # Should not raise

    def test_full_workflow(self):
        """Test the complete acquire -> mark_complete -> release workflow."""
        lock1 = self._create_lock()

        # First job acquires lock
        self.assertTrue(lock1.acquire())
        self.assertFalse(lock1.cache_exists())

        # Mark complete and release
        lock1.mark_complete()
        lock1.release()
        self.assertTrue(lock1.cache_exists())

        # Second job should skip
        lock2 = self._create_lock()
        self.assertFalse(lock2.acquire())

    def test_concurrent_acquisition(self):
        """Test that concurrent lock acquisition serializes correctly."""
        results = []
        lock = threading.Lock()

        def acquire_and_release(job_id: int):
            mgr = self._create_lock("shared")
            acquired = mgr.acquire()
            if acquired:
                time.sleep(0.1)  # Hold lock briefly to simulate compilation
                mgr.mark_complete()
                mgr.release()
            with lock:
                results.append((job_id, acquired))

        threads = [threading.Thread(target=acquire_and_release, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one thread should have acquired the lock
        acquired_count = sum(1 for _, acquired in results if acquired)
        self.assertEqual(acquired_count, 1)
