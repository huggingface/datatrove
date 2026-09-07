This is a fast and memory efficient implementation of MinHash step 3 written in Rust.

Build and run with
```
cargo build --release
./target/release/s3 --help
```

Two versions are available:
- `s3` reads and writes the data directly to s3
- `local` reads and writes the data from/to the local filesystem

Here's an example of a config with the python version and the equivalent rust command:
```python
BASE_PATH = "s3://some-bucket/minhash"
s3 = SlurmPipelineExecutor(
    job_name=f"mh3",
    pipeline=[
        MinhashDedupCluster(
            input_folder=f"{BASE_PATH}/buckets",
            output_folder=f"{BASE_PATH}/remove_ids",
            save_cluster_size=True
        ),
    ],
    tasks=1,
    cpus_per_task=2,
    mem_per_cpu_gb=450,
    logging_dir=f"logs/clusters",
    partition="hopper-cpu",
    time="100:00:00"
).run()
```

Assuming step 2 was run with `minhash_config.num_buckets * 50 = 700` tasks

```
./target/release/s3 --input-folder s3://some-bucket/minhash/buckets/ --output-folder s3://some-bucket/minhash/remove_ids/ --total-files 700 --downloads 20
```
Or if running locally:
```
./target/release/local --input-folder /fsx/some-path/minhash/buckets/ --output-folder /fsx/some-path/minhash/remove_ids/ --total-files 700 --concurrent-ops 20
```

Both binaries always write `.remove` and `.sizes`. Add `--save-cluster-id` to also
write `.clusters`, equivalent to Python's `save_cluster_id=True`:

```sh
./target/release/local --input-folder /path/to/buckets --output-folder /path/to/remove_ids --total-files 700 --save-cluster-id
```

Each `{rank:06d}.clusters` file contains sorted `(document_position, cluster_id)`
pairs of little-endian unsigned 32-bit integers (8 bytes per record, no header).
IDs start at zero and are shared across ranks, ordered by the smallest
`(rank, document_position)` in each component, as in Python. Representatives and
removed documents both receive IDs; documents absent from the duplicate graph
have no record. Matches to a historical index retain Python's shared sentinel
component. IDs identify components within this input, not across different runs
with different documents.

Load them in stage 4 with the same document order and ranks as stage 1:

```python
MinhashDedupFilter("/path/to/remove_ids", load_cluster_ids=True, load_cluster_sizes=True)
```

Saving IDs adds 8 bytes of output per node and a shared lookup table proportional
to the number of components. This work is skipped when the flag is omitted.

To run the Rust tests and Python interoperability tests from the repository root:

```sh
cargo test --locked --manifest-path src/datatrove/tools/fast_mh3/Cargo.toml
cargo build --locked --bins --manifest-path src/datatrove/tools/fast_mh3/Cargo.toml
DATATROVE_FAST_MH3_BIN_DIR="$PWD/src/datatrove/tools/fast_mh3/target/debug" python -m pytest -q tests/pipeline/dedup/test_fast_mh3.py
```

The Python tests use the testing dependencies, including `moto[s3,server]` for
a local S3 server. They skip when `DATATROVE_FAST_MH3_BIN_DIR` is not set.
