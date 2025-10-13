# Example 2: Text Extraction and Quality Filtering

## Objective
Extract clean text from Common Crawl WARC files using Trafilatura and apply quality filters.

## Components
- **WarcReader**: Read Common Crawl WARC files
- **Trafilatura**: Extract text from HTML
- **LanguageFilter**: Keep only English (with exclusion writer)
- **GopherRepetitionFilter**: Remove repetitive content (with exclusion writer)
- **GopherQualityFilter**: Apply quality heuristics (with exclusion writer)
- **JsonlWriter**: Save clean results

## Implementation
**File:** `spec/phase1/examples/02_text_extraction.py`

## Data Requirements
- **Input:** `spec/phase1/data/CC-MAIN-20180420081400-20180420101400-00000.warc.gz` (limit: 200)
- **Output:**
  - Clean: `spec/phase1/output/02_clean/clean_${rank}.jsonl`
  - Non-English: `spec/phase1/output/02_non_english/`
  - Repetitive: `spec/phase1/output/02_repetitive/`
  - Low Quality: `spec/phase1/output/02_low_quality/`
- **Logs:** `spec/phase1/logs/02_text_extraction/`

## Expected Results
- Input: 200 WARC records
- After Trafilatura extraction: ~100 docs (some records fail)
- After language filter: ~80 docs (remove non-English)
- After repetition filter: ~40 docs (aggressive repetition removal)
- After quality filter: ~8 docs (strict quality requirements)
- Result: 200 → 8 clean documents (96% filtered)

## Status
- [x] Implemented
- [x] Tested
- [x] Documentation updated

## Notes
- Demonstrates exclusion writers to analyze what gets filtered
- Includes helper function to check WARC file exists
- Filters are very aggressive - real web data is low quality
- Requires WARC file download (not included in repo)