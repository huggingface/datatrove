import os
import shutil
import struct
import tempfile
import unittest

from datatrove.data import Document
from datatrove.io import get_datafolder
from datatrove.pipeline.dedup.minhash import (
    SENTINEL,
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.utils.hashing import HashConfig


class TestMinhashIndexRepresentative(unittest.TestCase):
    def test_later_new_docs_with_the_same_index_band_are_removed(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        sig_dir = os.path.join(tmp, "sigs", "bucket_000")
        index_dir = os.path.join(tmp, "index", "bucket_000")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(sig_dir)
        os.makedirs(index_dir)

        sig = 42
        with open(os.path.join(sig_dir, "00000.minhash.sig"), "wb") as handle:
            handle.write(struct.pack("<QI", sig, 1))
            handle.write(struct.pack("<QI", sig, 2))
        with open(os.path.join(index_dir, "historical.minhash.index"), "wb") as handle:
            handle.write(struct.pack("<Q", sig))

        config = MinhashConfig(
            num_buckets=1,
            hashes_per_bucket=1,
            hash_config=HashConfig(precision=64, hash_fc="sha1"),
        )
        MinhashDedupBuckets(
            input_folder=os.path.join(tmp, "sigs"),
            output_folder=out_dir,
            index_folder=os.path.join(tmp, "index"),
            config=config,
            only_dedup_in_index=True,
        ).run(None, rank=0, world_size=1)

        folder = get_datafolder(out_dir)
        pairs = []
        for name in folder.list_files(glob_pattern="*.dups"):
            with folder.open(name, "rb") as handle:
                while data := handle.read(struct.calcsize("<4I")):
                    pairs.append(struct.unpack("<4I", data))

        self.assertEqual(
            pairs,
            [
                (SENTINEL, SENTINEL, 0, 1),
                (SENTINEL, SENTINEL, 0, 2),
            ],
        )

    def test_only_dedup_in_index_false_with_ignore_index_matches_dedups_new_copies(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        sig_dir = os.path.join(tmp, "sigs", "bucket_000")
        index_dir = os.path.join(tmp, "index", "bucket_000")
        buckets_dir = os.path.join(tmp, "buckets")
        cluster_dir = os.path.join(tmp, "cluster")
        os.makedirs(sig_dir)
        os.makedirs(index_dir)

        sig = 42
        with open(os.path.join(sig_dir, "00000.minhash.sig"), "wb") as handle:
            handle.write(struct.pack("<QI", sig, 0))
            handle.write(struct.pack("<QI", sig, 1))
        with open(os.path.join(index_dir, "historical.minhash.index"), "wb") as handle:
            handle.write(struct.pack("<Q", sig))

        config = MinhashConfig(
            num_buckets=1,
            hashes_per_bucket=1,
            hash_config=HashConfig(precision=64, hash_fc="sha1"),
        )
        MinhashDedupBuckets(
            input_folder=os.path.join(tmp, "sigs"),
            output_folder=buckets_dir,
            index_folder=os.path.join(tmp, "index"),
            config=config,
            only_dedup_in_index=False,
        ).run(None, rank=0, world_size=1)

        MinhashDedupCluster(
            input_folder=buckets_dir,
            output_folder=cluster_dir,
            config=config,
            ignore_index_matches=True,
        ).run(None)

        docs = [
            Document(text="copy a", id="a"),
            Document(text="copy b", id="b"),
        ]
        kept = list(MinhashDedupFilter(input_folder=cluster_dir).run(iter(docs), rank=0, world_size=1))
        self.assertEqual([doc.id for doc in kept], ["a"])
