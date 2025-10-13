# Example 3: Tokenization Pipeline

## Objective
Tokenize documents using LLM tokenizers and analyze token statistics for ML workflows.

## Components
- **JsonlReader**: Read from HuggingFace C4 dataset
- **TokensCounter**: Count tokens with GPT-2 tokenizer (before and after filtering)
- **LambdaFilter**: Filter by token count (50-2048 tokens)
- **JsonlWriter**: Save tokenized results

## Implementation
**File:** `spec/phase1/examples/03_tokenization.py`

## Data Requirements
- **Input:** `hf://datasets/allenai/c4/en/` (glob: `c4-train.00000-of-01024.json.gz`, limit: 1000)
- **Output:** `spec/phase1/output/03_tokenized/tokenized_${rank}.jsonl` (no compression)
- **Logs:** `spec/phase1/logs/03_tokenization/`

## Expected Results
- Input: 1000 documents from C4
- After token count filter (50-2048 tokens): ~922 docs
- Total tokens: ~380K
- Average: 413 tokens/doc
- Token/word ratio: ~1.22

## Status
- [x] Implemented
- [x] Tested
- [x] Documentation updated

## Notes
- Demonstrates token counting before/after filtering
- Includes helper functions to analyze tokenization and compare tokenizers
- GPT-2 tokenizer commonly used for LLM training