# Example 4: Statistics Collection

## Objective
Analyze dataset characteristics using DataTrove's statistics blocks.

## Learning Goals
- Collect document, word, and line statistics
- Understand statistics aggregation
- Use StatsMerger for combining results
- Analyze data quality metrics

## Implementation Details

### Pipeline Components
1. **JsonlReader**: Read processed data
2. **DocStats**: Document-level statistics
3. **WordStats**: Word-level analysis
4. **LineStats**: Line structure metrics
5. **StatsMerger**: Combine statistics from multiple tasks

### Data Source
- Use output from Example 2 (clean Common Crawl data)
- Or fallback to C4 subset if Example 2 not completed
- Small enough for local processing but meaningful stats

### Pipeline Flow
```
JsonlReader("output/clean/" or "hf://datasets/allenai/c4/en/")
    ↓
DocStats(output_folder="stats/")
    ↓
WordStats(output_folder="stats/")
    ↓
LineStats(output_folder="stats/")
```

Then merge:
```
StatsMerger(
    input_folder="stats/",
    output_folder="stats_merged/"
)
```

## Statistics Collected

### DocStats
- Document length
- Whitespace ratio
- Digit ratio
- Punctuation ratio

### WordStats
- Word count
- Average word length
- Stop word ratio
- Type-token ratio

### LineStats
- Number of lines
- Average line length
- Short/long line ratios

## Files to Create
1. `spec/phase1/examples/04_statistics.py` - Main pipeline
2. `spec/phase1/examples/04_merge_stats.py` - Stats merger

## Execution Plan
1. Run stats collection pipeline (can use multiple tasks)
2. Run stats merger
3. Load and analyze merged statistics
4. Create simple visualizations (optional)

## Success Metrics
- [x] Statistics collected for all documents
- [x] Stats files created in output folder
- [x] Merger combines stats correctly (understood format issues)
- [x] Can interpret statistics results

## Notes
- Stats are saved as JSON for easy loading
- Multiple tasks create separate stats files
- Merger combines all task outputs
- Useful for data quality assessment

## Implementation Notes (Completed)
- Analyzed 922 tokenized documents from Example 3
- Collected doc stats (length, whitespace), word stats, line stats, language stats
- Key findings: avg 1875 chars/doc, 16.71% whitespace, 0.929 English confidence
- Discovered StatsMerger format issues - stats collectors need separate output_folder
- Simplified to single-task execution for local examples
- Each stat type requires output_folder parameter, not just config