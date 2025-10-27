# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataTrove is a library for processing, filtering, and deduplicating text data at very large scale. It provides prebuilt processing blocks and a framework for custom functionality. Pipelines are platform-agnostic, running locally or on slurm/ray clusters with relatively low memory usage.

## Development Commands

### Setup
```bash
pip install -e ".[dev]"  # Install with all dev dependencies
pre-commit install       # Install pre-commit hooks
```

### Testing
```bash
# Run all tests
make test
# or
python -m pytest -sv ./tests/

# Run specific test file
python -m pytest -sv ./tests/pipeline/test_filters.py

# Run specific test
python -m pytest -sv ./tests/pipeline/test_filters.py::test_filter_name
```

### Code Quality
```bash
# Check code quality (linter + formatter)
make quality

# Auto-fix code style issues
make style
```

### CLI Tools
After installation, several command-line tools are available:
- `merge_stats` - Merge statistics from multiple tasks
- `check_dataset` - Validate dataset integrity
- `failed_logs` - View logs from failed tasks
- `inspect_data` - Inspect processed data
- `jobs_status` - Check status of pipeline jobs
- `track_jobs` - Track multiple pipeline jobs
- `launch_pickled_pipeline` - Execute pickled pipeline configurations

## Architecture

### Core Concepts

**Document**: The fundamental data unit (`src/datatrove/data.py`)
- `text`: The actual text content
- `id`: Unique identifier (string)
- `metadata`: Dictionary for additional information
- `media`: List of associated media (future use)

**Pipeline**: A list of processing steps that transform documents
- Each step takes a generator of `Document` and yields `Document`
- Steps can be `PipelineStep` instances, custom callables, or sequences
- Data flows through pipeline via generator pattern (memory efficient)

**PipelineStep**: Base class for all processing blocks (`src/datatrove/pipeline/base.py`)
- `run(data, rank, world_size)`: Main processing method
- `stat_update()`: Track statistics during processing
- `track_time()`: Context manager for timing code blocks
- Automatically checks dependencies via `_requires_dependencies`

**Executor**: Manages pipeline execution across different platforms (`src/datatrove/executor/`)
- `LocalPipelineExecutor`: Multi-process execution on local machine
- `SlurmPipelineExecutor`: Distributed execution on slurm clusters
- `RayPipelineExecutor`: Distributed execution using Ray
- All executors share common interface: `run()`, `world_size`, task completion tracking

**Task & Sharding**: Parallelization is achieved by dividing work into tasks
- Each task processes a non-overlapping shard of input files
- Files are distributed: task `i` processes files `i, i+N, i+2N, ...` where N = world_size
- Completion tracking via empty marker files in `${logging_dir}/completions/`
- Failed tasks can be rerun by relaunching the same executor (don't change task count)

**DataFolder**: Abstraction over filesystem operations (`src/datatrove/io.py`)
- Wraps fsspec's `DirFileSystem` for local/remote file operations
- `get_shard(rank, world_size)`: Deterministic file sharding
- `list_files()`: List files with optional glob patterns
- `open()`: Open files with automatic parent directory creation
- Supports local, S3, HuggingFace Hub, and other fsspec backends

### Pipeline Block Types

All blocks in `src/datatrove/pipeline/`:

**Readers** (`readers/`): Read data and yield Documents
- `WarcReader`, `JsonlReader`, `CSVReader`, `ParquetReader`, `HuggingFaceReader`
- Common args: `data_folder`, `text_key`, `id_key`, `default_metadata`, `limit`, `glob_pattern`
- Implement `_get_document_from_dict()` to transform raw data to Documents

**Writers** (`writers/`): Save Documents to disk/cloud
- `JsonlWriter`, `ParquetWriter`, `HuggingFaceWriter`
- Use `output_filename` templates: `${rank}`, `${id}`, `${metadata_key}`
- Inherit from `DiskWriter` base class

**Extractors** (`extractors/`): Extract text from raw formats
- `Trafilatura`: HTML text extraction (most common)
- Transform document text in-place

**Filters** (`filters/`): Remove documents based on criteria
- Return `True` to keep, `False` to remove
- Can save removed docs via `exclusion_writer` parameter
- Examples: `LanguageFilter`, `GopherQualityFilter`, `URLFilter`, `C4QualityFilter`
- Inherit from `BaseFilter`

**Formatters** (`formatters/`): Modify document content
- `PIIFormatter`: Remove personally identifiable information
- `FTFYFormatter`: Fix text encoding issues

**Dedup** (`dedup/`): Deduplication algorithms
- `MinhashDedup*`: Multi-stage minhash deduplication (signature → buckets → cluster → filter)
- `SentenceDedup`: Sentence-level exact deduplication
- `ExactSubstrings`: Substring deduplication
- Typically runs as multi-stage dependent pipelines

**Stats** (`stats/`): Collect dataset statistics
- Two-stage process: collect per-shard → merge across shards
- Groupings: `summary`, `fqdn`, `suffix`, `histogram`
- Results saved as `MetricStatsDict` JSON files

**Tokens** (`tokens/`): Tokenization and token operations
- `TokensCounter`: Count tokens in documents
- `DocumentTokenizer`: Tokenize and save tokens

**Inference** (`inference/`): Run LLM inference for synthetic data
- `InferenceRunner`: Supports vLLM, SGLang, and remote vLLM endpoints
  - **Local servers** (`server_type="vllm"` or `"sglang"`): Automatically spawns and manages server processes
  - **Remote servers** (`server_type="vllm-remote"`): Connects to existing external vLLM endpoints
- Automatic checkpointing via `checkpoints_local_dir` and `records_per_chunk`
- Server architecture:
  - `LocalInferenceServer`: Base for local server management (process spawning, port finding, logging)
  - `RemoteInferenceServer`: Base for external endpoint connections (health checks, no process management)

**Using External vLLM Server:**
```python
from datatrove.pipeline.inference import InferenceRunner, InferenceConfig
from datatrove.pipeline.writers import JsonlWriter

# Connect to an existing vLLM server instead of spawning a local one
config = InferenceConfig(
    server_type="vllm-remote",
    model_name_or_path="meta-llama/Llama-3-8B",
    external_endpoint="http://my-vllm-server.com:8000",  # Required for vllm-remote
    temperature=0.7,
    max_concurrent_requests=100,
)

runner = InferenceRunner(
    query_builder=my_query_builder,
    config=config,
    output_writer=JsonlWriter("output/synthetic_data"),
    checkpoints_local_dir="checkpoints/",
)
```

### Key Implementation Patterns

**Custom Pipeline Blocks**: Three approaches
1. List of Documents (for testing): `[Document(...), Document(...)]`
2. Custom function: `def process(data, rank, world_size) -> DocumentsPipeline`
3. Custom class inheriting from `PipelineStep` or subclass (`BaseFilter`, `BaseExtractor`, etc.)

**Statistics Tracking**:
```python
with self.track_time():
    # processing code
    self.stat_update("metric_name", value=count, unit="doc")
```

**Dependency Pipeline Execution**:
```python
stage2 = SlurmPipelineExecutor(..., depends=stage1)
stage3 = SlurmPipelineExecutor(..., depends=stage2)
stage3.run()  # Automatically runs stage1 → stage2 → stage3
```

**Multi-stage Deduplication**: See `examples/minhash_deduplication.py`
- Stage 1: Compute signatures (`MinhashDedupSignature`)
- Stage 2: Create buckets (`MinhashDedupBuckets`)
- Stage 3: Cluster duplicates (`MinhashDedupCluster`)
- Stage 4: Filter documents (`MinhashDedupFilter`)

## File Locations

- Core data structures: `src/datatrove/data.py`
- Pipeline base: `src/datatrove/pipeline/base.py`
- Executor base: `src/datatrove/executor/base.py`
- I/O utilities: `src/datatrove/io.py`
- Logging utilities: `src/datatrove/utils/logging.py`
- Stats handling: `src/datatrove/utils/stats.py`

## Testing Strategy

Tests are organized by component in `tests/`:
- `tests/pipeline/` - Tests for readers, filters, extractors, dedup, etc.
- `tests/executor/` - Tests for executors
- `tests/test_io.py` - Tests for I/O operations
- Use `tests/utils.py` for test fixtures and helpers

## Important Implementation Details

**Logging Structure**: Each pipeline execution creates:
```
${logging_dir}/
├── executor.json          # Serialized executor config
├── ranks_to_run.json      # List of tasks being run
├── logs/
│   └── task_00000.log     # Individual task logs
├── completions/
│   └── 00000              # Empty marker files for completed tasks
├── stats/
│   └── 00000.json         # Per-task statistics
└── stats.json             # Merged global statistics
```

**Generator Pattern**: Pipelines use generators for memory efficiency
- Documents flow through pipeline without loading entire dataset into memory
- Use `deque(pipelined_data, maxlen=0)` to exhaust generator at pipeline end

**Sharding Guarantees**: File distribution is deterministic
- Same `world_size` always produces same sharding
- Never change `world_size` when re-running failed tasks
- Each file processed by exactly one task

**Dependency Checking**: `PipelineStep.__new__` checks `_requires_dependencies`
- Add `_requires_dependencies = ["package_name"]` to custom blocks
- Checked via `check_required_dependencies()` from `utils/_import_utils.py`

## Common Patterns in Examples

All examples are in `examples/`:
- `fineweb.py`: Full reproduction of FineWeb dataset (filtering + minhash dedup)
- `process_common_crawl_dump.py`: CommonCrawl WARC processing pipeline
- `minhash_deduplication.py`: Complete minhash deduplication workflow
- `sentence_deduplication.py`: Sentence-level deduplication
- `tokenize_c4.py`: Tokenization from HuggingFace datasets
- `summary_stats.py`: Collecting and merging statistics

## Notes for Development

- DataFolder paths support local, S3, and HuggingFace Hub via fsspec
- Use `get_datafolder()` to parse various path formats: str, tuple, or DataFolder
- Executors save pickled versions of themselves for slurm job arrays
- Color logging can be controlled via `DATATROVE_COLORIZE_LOGS` env var
- All pipeline blocks should yield Documents, never return lists (memory efficiency)
