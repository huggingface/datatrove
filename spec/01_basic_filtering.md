# Example 1: Basic Data Processing Pipeline

## Objective
Learn the fundamentals of DataTrove by creating a simple pipeline that reads data, applies basic filtering, and writes the results.

## Learning Goals
- Understand the Document structure (text, id, metadata)
- Learn pipeline composition basics
- Master basic I/O operations (JsonlReader, JsonlWriter)
- Implement simple filtering logic with LambdaFilter

## Implementation Details

### Pipeline Components
1. **JsonlReader**: Read from HuggingFace datasets
2. **LambdaFilter**: Apply custom filtering logic
3. **JsonlWriter**: Save filtered results

### Data Source
- Use HuggingFace C4 dataset (small subset)
- Direct reading: `hf://datasets/allenai/c4/en/`
- DataTrove can read directly from HF Hub
- No need to download - streams data as needed

### Example Access Pattern
```python
JsonlReader(
    "hf://datasets/allenai/c4/en/",
    glob_pattern="c4-train.00000-of-01024.json.gz",  # Just one shard
    limit=1000  # Only read 1000 documents for testing
)
```

### Filtering Criteria
- Filter 1: Keep documents with text length > 50 characters
- Filter 2: Keep documents containing specific keywords
- Filter 3: Keep documents based on metadata values

### Expected Pipeline Flow
```
JsonlReader("data/sample.jsonl")
    ↓
LambdaFilter(lambda doc: len(doc.text) > 50)
    ↓
LambdaFilter(lambda doc: "machine learning" in doc.text.lower())
    ↓
JsonlWriter("output/filtered/")
```

## Files to Create
1. `examples_local/01_basic_filtering.py` - Main pipeline implementation
2. `examples_local/data/sample_data_generator.py` - Script to generate sample data
3. `examples_local/data/sample.jsonl` - Sample input data

## Execution Plan
1. Generate sample data
2. Run pipeline with LocalPipelineExecutor
3. Verify output
4. Experiment with different filter combinations
5. Analyze filtered vs. original data statistics

## Success Metrics
- [x] Pipeline runs without errors
- [x] Correct number of documents filtered
- [x] Output files created in expected location
- [x] Can modify filters and see different results
- [x] Understanding of Document flow through pipeline

## Variations to Try
1. Chain multiple filters
2. Use SamplerFilter for random sampling
3. Add exclusion_writer to save filtered-out documents
4. Experiment with different reader/writer formats

## Key Concepts Demonstrated
- Document as the core data structure
- Generator-based processing (memory efficient)
- Pipeline composition pattern
- Task-based parallelization basics

## Notes
- Start with 1 task, then try with multiple tasks
- Observe how files are sharded across tasks
- Check logging output to understand execution flow

## Implementation Notes (Completed)
- Used C4 dataset directly from HuggingFace (1000 docs)
- Applied 3 filters: length > 100, keyword matching, 50% sampling
- Results: 1000 → 77 documents after filtering
- Learned about LambdaFilter parameter issues (no 'name' param)