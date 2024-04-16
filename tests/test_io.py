import multiprocessing
import os
import shutil
import tempfile
import time
import unittest
from functools import partial

import boto3
import moto

from datatrove.io import get_datafolder, safely_create_file


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
