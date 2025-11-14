"""Tests for io.py changes"""

import shutil
import tempfile
import unittest
from pathlib import Path


class TestIoChanges(unittest.TestCase):
    """Test io.py changes for kwargs passing"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_open_file_with_kwargs(self):
        """Test that open_file passes kwargs to fs.open"""
        from datatrove.io import open_file

        # Create a test file
        test_file = Path(self.tmp_dir) / "test.txt"
        test_file.write_text("test content")

        # Open with additional kwargs (like block_size)
        # This should not raise an error
        with open_file(str(test_file), mode="rt", block_size=1024) as f:
            content = f.read()
            self.assertEqual(content, "test content")

    def test_get_shard_from_paths_file_with_compression(self):
        """Test that get_shard_from_paths_file supports compression='infer'"""
        import gzip

        from datatrove.io import get_shard_from_paths_file

        # Create a compressed paths file
        paths_file = Path(self.tmp_dir) / "paths.txt.gz"
        test_paths = ["path1.txt", "path2.txt", "path3.txt"]

        with gzip.open(paths_file, "wt") as f:
            for path in test_paths:
                f.write(path + "\n")

        # Read with compression='infer' (automatically added in the diff)
        shard_paths = list(get_shard_from_paths_file(str(paths_file), rank=0, world_size=1))

        # Should successfully read all paths
        self.assertEqual(shard_paths, test_paths)

    def test_get_shard_from_paths_file_sharding(self):
        """Test that get_shard_from_paths_file correctly shards paths"""
        from datatrove.io import get_shard_from_paths_file

        # Create a paths file
        paths_file = Path(self.tmp_dir) / "paths.txt"
        test_paths = [f"path{i}.txt" for i in range(10)]

        paths_file.write_text("\n".join(test_paths))

        # Test rank 0 with world_size 2
        shard_0 = list(get_shard_from_paths_file(str(paths_file), rank=0, world_size=2))
        # Test rank 1 with world_size 2
        shard_1 = list(get_shard_from_paths_file(str(paths_file), rank=1, world_size=2))

        # Verify sharding
        # rank 0 should get indices 0, 2, 4, 6, 8
        # rank 1 should get indices 1, 3, 5, 7, 9
        self.assertEqual(shard_0, ["path0.txt", "path2.txt", "path4.txt", "path6.txt", "path8.txt"])
        self.assertEqual(shard_1, ["path1.txt", "path3.txt", "path5.txt", "path7.txt", "path9.txt"])

        # Verify no overlap
        self.assertEqual(set(shard_0) & set(shard_1), set())

        # Verify complete coverage
        self.assertEqual(set(shard_0) | set(shard_1), set(test_paths))


if __name__ == "__main__":
    unittest.main()
