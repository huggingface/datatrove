# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

DataTrove is a large-scale data processing library from HuggingFace for processing, filtering, and deduplicating text data. It provides platform-agnostic pipelines that work locally, on Slurm clusters, or with Ray for distributed processing.

## Common Development Commands

### Building and Dependencies
```bash
# Install for development with all dependencies
pip install -e ".[dev]"

# Install specific features
pip install -e ".[processing,s3]"  # Processing with S3 support
pip install -e ".[all]"             # All dependencies
```

### Code Quality and Testing
```bash
# Run linter and formatter checks
make quality
# Or directly:
ruff check src tests examples
ruff format --check src tests examples

# Auto-fix linting issues and format code
make style
# Or directly:
ruff check --fix src tests examples
ruff format src tests examples

# Run tests
make test
# Or directly:
python -m pytest -sv ./tests/

# Run specific test file
python -m pytest -sv ./tests/pipeline/test_filters.py
```

### Pre-commit Setup
```bash
pre-commit install  # Set up code style hooks
```

## Architecture Overview

### Core Concepts

1. **Pipeline**: A list of processing steps (read → filter → extract → write)
2. **Executor**: Runs pipelines on different platforms (LocalPipelineExecutor, SlurmPipelineExecutor, RayPipelineExecutor)
3. **Task**: Unit of parallelization - each processes a shard of data
4. **Document**: Core data structure with `text`, `id`, and `metadata` fields

### Project Structure

```
src/datatrove/
├── executor/          # Pipeline executors (local, slurm, ray)
├── pipeline/          # Processing blocks
│   ├── readers/       # Read various formats (WARC, JSON, CSV, etc.)
│   ├── extractors/    # Extract text (e.g., Trafilatura for HTML)
│   ├── filters/       # Filter documents (language, quality, etc.)
│   ├── writers/       # Save to disk/cloud
│   ├── dedup/         # Deduplication algorithms
│   ├── tokens/        # Tokenization
│   ├── stats/         # Statistics collection
│   └── inference/     # LLM inference support
├── io.py              # DataFolder abstraction for filesystem operations
├── data.py            # Document class definition
└── tools/             # CLI utilities
```

### Pipeline Design Pattern

All pipeline blocks follow this pattern:
1. Take a generator of `Document` as input
2. Process/filter documents
3. Yield transformed `Document` objects
4. Track statistics via `self.stat_update()`
5. Use `self.track_time()` for performance monitoring

### Key Abstractions

- **PipelineStep**: Base class for all pipeline blocks
- **DataFolder**: Unified interface for local/S3/HF Hub file systems via fsspec
- **PipelineStats**: Collects and merges statistics across tasks
- **Document**: `text` (content), `id` (unique identifier), `metadata` (dict)

## Creating Custom Pipeline Blocks

When implementing custom blocks:
1. Inherit from appropriate base class (`PipelineStep`, `BaseFilter`, `BaseReader`, etc.)
2. Implement `run()` method that accepts and yields `DocumentsPipeline`
3. Use `self.stat_update()` to track metrics
4. Wrap processing in `self.track_time()` context manager
5. Handle sharding via `rank` and `world_size` parameters

## Testing Patterns

- Tests use `unittest` framework
- Mock S3 operations with `moto`
- Use decorators like `@require_nltk` for optional dependencies
- Test individual pipeline blocks with synthetic `Document` objects
- Integration tests compose multiple pipeline steps

## Important Notes

- Each task processes complete files (no automatic file splitting)
- Use absolute paths for `file_path` parameters
- Statistics are saved to `logging_dir/stats/`
- Completed tasks create markers in `logging_dir/completions/`
- Rerunning jobs automatically skips completed tasks (unless `skip_completed=False`)