import multiprocessing
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import boto3
import moto

from datatrove.io import cached_asset_path_or_download, get_datafolder


EXAMPLE_DIRS = ("/home/testuser/somedir", "file:///home/testuser2/somedir", "s3://test-bucket/somedir")
FULL_PATHS = (
    "/home/testuser/somedir/file.txt",
    "/home/testuser2/somedir/file.txt",
    "s3://test-bucket/somedir/file.txt",
)


def cached_asset_path_or_download_wait_mock(remote_path, local_path):
    # We have to define mocking here, as multiprocessing can't pickle non top-level fc
    with patch("datatrove.io.download_file") as mock:
        mock.side_effect = lambda *args, **kwargs: time.sleep(3)
        cached_asset_path_or_download(remote_path, local_path)


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

    def test_cached_asset_path_or_download_locking(self):
        # This could be a bit flaky test due to time.sleep, but it's a good way to test the locking

        start = time.time()
        with multiprocessing.Pool(2) as pool:
            pool.starmap(
                cached_asset_path_or_download_wait_mock, [("dummy_remote_path", "dummy_local_path") for _ in range(2)]
            )

        # if locking works this should NOT be faster than 6 seconds, as the wait times must be sequential
        self.assertGreater(time.time() - start, 6)
