import heapq
import os
import shutil
import struct
import tempfile
import unittest
from collections import defaultdict, deque
from math import floor

import numpy as np

from datatrove.data import Document
from datatrove.io import get_datafolder
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
    MinhashDedupSignature,
    read_sigs,
)

from ..utils import require_nltk, require_xxhash, use_hash_configs


lorem_ipsum = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam euismod vel ante vitae rhoncus. Curabitur eu lectus et magna maximus facilisis eu non magna. Maecenas sed velit vitae est ornare placerat. Vestibulum quis consectetur nunc, a feugiat lorem. Cras in ipsum fringilla, vestibulum urna sit amet, viverra tortor. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Morbi euismod vestibulum elit id placerat. Fusce malesuada ultricies condimentum. Cras tincidunt eget lorem nec hendrerit. Aenean mattis arcu dolor, id semper velit ullamcorper malesuada. Aliquam non ipsum et eros venenatis aliquet. Proin eleifend interdum scelerisque. Interdum et malesuada fames ac ante ipsum primis in faucibus. Mauris nunc sapien, molestie eget convallis at, maximus nec ipsum. Morbi quam diam, blandit ut mollis at, varius eu tellus. Maecenas sem justo, porttitor at odio nec, interdum posuere ex.

Aliquam pretium ac nulla et porttitor. Nunc quis felis posuere, lobortis magna quis, imperdiet nulla. Maecenas tempor, mi vel vestibulum tempus, arcu elit scelerisque erat, eu molestie velit eros id metus. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Proin fringilla viverra urna eu dictum. Suspendisse interdum, leo non sagittis pulvinar, mauris est euismod ante, et luctus nisi est id odio. Proin purus nunc, feugiat id condimentum eu, efficitur at diam. Quisque aliquet felis non risus rutrum, eu porttitor justo venenatis. Donec ut est felis. Proin risus augue, gravida fermentum elementum eu, varius eget tortor. Sed a sem quis tortor ultrices elementum. Nullam tempor rutrum ipsum id eleifend. Donec a elit tincidunt, sagittis magna a, pretium felis.

Sed ut faucibus dui, a sodales nunc. Praesent fermentum diam quis augue porttitor vulputate. Mauris pretium ipsum ut erat hendrerit eleifend. Maecenas consectetur elit a ligula auctor porta. Duis vitae diam ac velit porttitor tempor. Quisque at arcu a augue dictum molestie et sed erat. Donec fermentum pulvinar elit. Morbi orci nisi, condimentum non tincidunt sit amet, hendrerit non justo. Ut fringilla dolor neque, ut porttitor ante congue vitae. In aliquam augue a sapien sodales ornare. In in maximus nunc. Quisque turpis nibh, commodo non semper non, hendrerit a massa.

Suspendisse potenti. Ut feugiat nibh ex. Nunc eget ligula ut massa tempus pretium vitae et mauris. Suspendisse potenti. Vivamus euismod ipsum est, id consectetur lorem suscipit non. Integer ac felis egestas risus ornare luctus nec vel massa. Donec scelerisque enim eu nulla commodo fringilla. Donec et pulvinar dolor, sit amet tristique risus. Cras et est id leo malesuada sollicitudin.

Quisque et aliquet diam. Aenean euismod efficitur enim, non semper eros. Nullam molestie vehicula eros, nec porttitor justo feugiat nec. Maecenas fringilla eleifend augue, eu mollis arcu vulputate ac. Quisque ullamcorper turpis sed tristique dapibus. Etiam imperdiet pulvinar fringilla. Nulla sed est eget odio dictum pretium. Cras ultricies nibh libero, efficitur consequat neque semper id. Donec porttitor lacus nunc, vitae gravida lorem consectetur sit amet. Pellentesque mollis, dui nec molestie consectetur, massa enim tempus ipsum, quis pretium felis massa congue felis. Donec efficitur pretium diam, quis elementum felis eleifend quis. Nullam vehicula tortor et quam eleifend, maximus dignissim nisi feugiat. """


@require_nltk
@require_xxhash
class TestMinhash(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    @use_hash_configs()
    def test_signatures(self, hash_config):
        config = MinhashConfig(hash_config=hash_config)
        minhash = MinhashDedupSignature(output_folder=os.path.join(self.tmp_dir, "signatures1"), config=config)
        shingles = minhash.get_shingles(lorem_ipsum)
        sig = minhash.get_signature(shingles)
        minhash2 = MinhashDedupSignature(output_folder=os.path.join(self.tmp_dir, "signatures2"), config=config)
        # check consistency
        assert sig == minhash2.get_signature(shingles)

        # check correct number of outputs
        assert len(sig) == minhash.config.num_buckets
        assert all((len(x) == minhash.config.hashes_per_bucket for x in sig))

        # check similarity approximation
        for pctd in range(0, 100, 5):
            dec = pctd / 100
            endp = floor(len(lorem_ipsum) * dec)
            textd = lorem_ipsum[:endp] + lorem_ipsum[len(lorem_ipsum) - 1 : endp : -1]
            sigd = minhash.get_signature(minhash.get_shingles(textd))
            simil = sum([1 if a == b else 0 for ba, bb in zip(sig, sigd) for a, b in zip(ba, bb)]) / minhash.num_hashes
            assert dec - 0.21 < simil < dec + 0.21

        # check output file format and order
        samples = [Document(f"sample {i}, {lorem_ipsum[i:: 10]}", id="test") for i in range(100)]
        minhash(samples)
        for bi in range(config.num_buckets):
            with minhash.output_folder.open(f"bucket_{bi:03d}/00000.minhash.sig", "rb") as f:
                prev = None
                doc_ids = set()
                S = np.dtype(config.hash_config.np_dtype).itemsize
                for di in range(100):
                    data = struct.unpack(
                        f"<%s{config.hash_config.struct_format}" % config.hashes_per_bucket,
                        f.read(config.hashes_per_bucket * S),
                    )
                    doc_id = struct.unpack("<I", f.read(struct.calcsize("I")))[0]
                    # ensure sorted order
                    assert prev is None or data >= prev
                    prev = data
                    assert 0 <= doc_id < 100
                    doc_ids.add(doc_id)
                assert len(doc_ids) == 100

    @use_hash_configs()
    def test_buckets_and_cluster(self, hash_config):
        sigs_folder = os.path.join(self.tmp_dir, "b_signatures")
        buckets_folder = os.path.join(self.tmp_dir, "b_buckets")
        clusters_folder = os.path.join(self.tmp_dir, "b_clusters")
        config = MinhashConfig(hash_config=hash_config)

        signatures_block = MinhashDedupSignature(output_folder=sigs_folder, config=config)
        buckets_block = MinhashDedupBuckets(
            input_folder=sigs_folder,
            output_folder=buckets_folder,
            config=config,
        )

        clusters = [[0, 20, 50], [400, 420], [800, 810, 820, 840, 860], [1205, 1215, 1225, 1245], [1600], [2000]]

        cluster_samples = [
            Document(text=lorem_ipsum[x : x + 400], id=f"{ci}_{xi}", metadata={"ci": ci, "xi": xi})
            for ci, cluster in enumerate(clusters)
            for xi, x in enumerate(cluster)
        ]

        signatures_block(cluster_samples)
        # test file read
        for fi, file in enumerate(buckets_block.input_folder.list_files()):
            last = None
            for sig in read_sigs(buckets_block.input_folder.open(file, "rb"), fi, config):
                assert 0 <= sig.doc_id < 100
                assert last is None or sig.sig >= last
                assert len(sig.sig) == config.hashes_per_bucket
                last = sig.sig

        # test duplicate pairs
        for b in range(config.num_buckets * 10):
            buckets_block(None, rank=b, world_size=config.num_buckets * 10)
        bucket_results_folder = get_datafolder(buckets_folder)
        dup_files = bucket_results_folder.list_files(glob_pattern="*.dups")
        pairs = defaultdict(set)
        for dup_file in dup_files:
            with bucket_results_folder.open(dup_file, "rb") as df:
                while data := df.read(4 * struct.calcsize("I")):
                    f1, d1, f2, d2 = struct.unpack("<4I", data)
                    assert f1 == f2 == 0
                    assert cluster_samples[d1].metadata["ci"] == cluster_samples[d2].metadata["ci"]
                    pairs[d1].add(d2)
                    pairs[d2].add(d1)
        doc_id = 0
        for cluster in clusters:
            print(cluster)
            print(pairs)
            for a in range(doc_id, doc_id + len(cluster)):
                assert len(cluster) < 2 or any(a in pairs[b] for b in range(doc_id, doc_id + len(cluster)) if a != b)
            doc_id += len(cluster)

        # clustering
        cluster_block = MinhashDedupCluster(bucket_results_folder, clusters_folder, config=config)
        cluster_block(None)

        cluster_results_folder = get_datafolder(clusters_folder)
        remove_ids = set()
        with cluster_results_folder.open(cluster_results_folder.list_files()[0], "rb") as df:
            while data := df.read(struct.calcsize("I")):
                remove_ids.add(struct.unpack("<I", data)[0])
        doc_id = 0
        kept = set()
        for ci, cluster in enumerate(clusters):
            to_remove = 0
            for xi, a in enumerate(range(doc_id, doc_id + len(cluster))):
                if a in remove_ids:
                    to_remove += 1
                else:
                    kept.add(f"{ci}_{xi}")
            doc_id += len(cluster)
            assert to_remove == len(cluster) - 1

        # filtering
        filter_block = MinhashDedupFilter(cluster_results_folder)
        filtered = filter_block(cluster_samples)
        filtered_ids = {x.id for x in filtered}
        assert filtered_ids == kept

    @use_hash_configs()
    def test_multiprocess_s2(self, hash_config):
        sigs_folder = os.path.join(self.tmp_dir, "b_signatures")
        buckets_folder1 = os.path.join(self.tmp_dir, "b_buckets")
        buckets_folder2 = os.path.join(self.tmp_dir, "b_buckets2")
        config = MinhashConfig(hash_config=hash_config)

        signatures_block = MinhashDedupSignature(output_folder=sigs_folder, config=config)
        buckets_block1 = MinhashDedupBuckets(
            input_folder=sigs_folder,
            output_folder=buckets_folder1,
            config=config,
        )
        buckets_block2 = MinhashDedupBuckets(
            input_folder=sigs_folder,
            output_folder=buckets_folder2,
            config=config,
        )

        samples = [Document(text=lorem_ipsum[x : x + 200], id="?") for x in range(0, len(lorem_ipsum) - 200, 10)]

        signatures_block(samples)
        # test duplicate pairs
        for b in range(config.num_buckets):
            buckets_block1(None, rank=b, world_size=config.num_buckets)
        for b in range(config.num_buckets * 10):
            buckets_block2(None, rank=b, world_size=config.num_buckets * 10)

        bucket_results_folder1 = get_datafolder(buckets_folder1)
        bucket_results_folder2 = get_datafolder(buckets_folder2)
        dup_files1 = bucket_results_folder1.list_files(glob_pattern="*.dups")
        dup_files2 = bucket_results_folder2.list_files(glob_pattern="*.dups")
        for bucket, dup_file1 in enumerate(dup_files1):
            alllines = []
            with bucket_results_folder1.open(dup_file1, mode="rb") as df1:
                while data := df1.read(4 * struct.calcsize("I")):
                    alllines.append(struct.unpack("<4I", data))
            pi = 0
            for filei in range(bucket * 10, (bucket + 1) * 10):
                pidelta = 0
                with bucket_results_folder2.open(dup_files2[filei], mode="rb") as df2:
                    while data := df2.read(4 * struct.calcsize("I")):
                        assert alllines[pi + pidelta] == struct.unpack("<4I", data)
                        pidelta += 1
                pi += pidelta

            assert pi == len(alllines)

            # check if actually read the same hashes, when using 1 worker or 10
            for bucket in range(config.num_buckets):
                sigs_df = get_datafolder(sigs_folder)
                sig_files = sigs_df.list_files(subdirectory=f"bucket_{bucket:03d}")
                sig_readers = [
                    read_sigs(sigs_df.open(file, mode="rb"), file_i, config) for file_i, file in enumerate(sig_files)
                ]
                pq = [x for x in [next(sig_reader, None) for sig_reader in sig_readers] if x is not None]
                heapq.heapify(pq)
                ordered = deque()
                while pq:
                    v = heapq.heappop(pq)
                    ordered.append(v)
                    next_sig = next(sig_readers[v.reader_id], None)
                    if next_sig:
                        heapq.heappush(pq, next_sig)
                for bucket_worker in range(10):
                    hash_min, hash_max = buckets_block2.get_worker_hash_range(
                        sig_files, bucket * 10 + bucket_worker, config.num_buckets * 10
                    )
                    sig_readers = [
                        read_sigs(sigs_df.open(file, mode="rb"), file_i, config, min_hash=hash_min, max_hash=hash_max)
                        for file_i, file in enumerate(sig_files)
                    ]
                    pq = [x for x in [next(sig_reader, None) for sig_reader in sig_readers] if x is not None]
                    while pq:
                        v = heapq.heappop(pq)
                        nextv = ordered.popleft()
                        assert v == nextv
                        next_sig = next(sig_readers[v.reader_id], None)
                        if next_sig:
                            heapq.heappush(pq, next_sig)
