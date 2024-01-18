import shutil
import tempfile
import unittest

import boto3
import moto

from datatrove.io import get_datafolder


EXAMPLE_DIRS = ("/home/testuser/somedir", "file:///home/testuser2/somedir", "s3://test-bucket/somedir")
FULL_PATHS = (
    "/home/testuser/somedir/file.txt",
    "/home/testuser2/somedir/file.txt",
    "s3://test-bucket/somedir/file.txt",
)


@moto.mock_s3
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
