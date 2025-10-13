# Example 4: Statistics Collection

## Objective
Collect and analyze document, word, line, and language statistics from processed data.

## Components
- **JsonlReader**: Read tokenized documents from Example 3
- **DocStats**: Document-level statistics (length, whitespace ratio)
- **WordStats**: Word-level analysis (top-k words by domain)
- **LineStats**: Line structure metrics
- **LangStats**: Language detection statistics

## Implementation
**File:** `spec/phase1/examples/04_statistics.py`

## Data Requirements
- **Input:** `spec/phase1/output/03_tokenized/*.jsonl` (output from Example 3)
- **Output:** `spec/phase1/output/04_stats/summary/*.json`
- **Logs:** `spec/phase1/logs/04_statistics/`

## Expected Results
- Documents analyzed: 922
- Average length: ~1,875 chars/doc
- Whitespace ratio: ~16.71%
- Average lines: ~34 per doc
- English confidence: ~0.929 (0-1 scale)
- Stats saved as JSON files in summary folders

## Status
- [x] Implemented
- [x] Tested
- [x] Documentation updated

## Notes
- Uses output from Example 3 (tokenized documents)
- All stats collectors write to same output folder
- Single task execution for simplicity
- Helper function parses and displays collected statistics