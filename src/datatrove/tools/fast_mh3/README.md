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
./target/release/local --input-folder /fsx/some-path/minhash/buckets/ --output-folder /fsx/some-path/minhash/remove_ids/ --total-files 700 --downloads 20
```