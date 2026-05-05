# Manual tests

Scripts in this folder hit the real Hugging Face Hub (and optionally Slurm) and
are therefore **not** run by `pytest` in CI.

Common requirements:

- `HF_TOKEN` environment variable set to a token with bucket-create / bucket-write
  scope for the org you target.
- Override the bucket id with `--bucket myorg/test-bucket` (default uses
  `$USER` and a UUID suffix).
- Slurm scripts additionally require a usable Slurm cluster and the right
  `--partition` / `--qos` flags.

Each script prints a clear `PASS` / `FAIL` line at the end and cleans up the
test bucket on success.
