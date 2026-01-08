import gzip
import multiprocessing
import os
import shutil
import tempfile
import time
import unittest
from functools import partial
from pathlib import Path

import boto3
import moto

from datatrove.io import get_datafolder, get_shard_from_paths_file, safely_create_file


EXAMPLE_DIRS = ("/home/testuser/somedir", "file:///home/testuser2/somedir", "s3://test-bucket/somedir")
FULL_PATHS = (
    "/home/testuser/somedir/file.txt",
    "/home/testuser2/somedir/file.txt",
    "s3://test-bucket/somedir/file.txt",
)


def fake_do_download(cc, ll):
    time.sleep(0.5)
    with ll:
        cc.value += 1


@moto.mock_aws
class TestIO(unittest.TestCase):
    def setUp(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_resolve_paths(self):
        for example, full in zip(EXAMPLE_DIRS, FULL_PATHS):
            self.assertEqual(get_datafolder(example).resolve_paths("file.txt"), full)

    def test_make_dirs(self):
        df = get_datafolder(self.tmp_dir)
        with df.open("subdir1/subdir2/some_path.txt", "wt") as f:
            f.write("hello")
        assert df.isdir("subdir1/subdir2")

    def test_safely_create_file_locking(self):
        for runi, (completed_exists, lock_exists, expec_calls) in enumerate(
            (
                (True, True, 0),
                (False, True, 1),
                (False, False, 1),
            )
        ):
            manager = multiprocessing.Manager()
            counter = manager.Value("i", 0)
            lock = manager.Lock()

            file_path = os.path.join(self.tmp_dir, str(runi), "myfile")
            os.makedirs(os.path.join(self.tmp_dir, str(runi)))

            with manager.Pool(2) as pool:
                if completed_exists:
                    open(file_path + ".completed", "a").close()
                if lock_exists:
                    open(file_path + ".lock", "a").close()

                pool.starmap(
                    partial(safely_create_file, do_processing=partial(fake_do_download, cc=counter, ll=lock)),
                    [(file_path,) for _ in range(2)],
                )

                self.assertEqual(counter.value, expec_calls)

    def test_get_shard_from_paths_file_with_compression(self):
        """Test that get_shard_from_paths_file supports compression='infer'"""
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
