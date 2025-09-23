# Example 3: Tokenization Pipeline

## Objective
Understand token processing for ML workflows using real data and standard tokenizers.

## Learning Goals
- Tokenize documents using common LLM tokenizers
- Count tokens for dataset statistics
- Understand token-based filtering
- Work with tokenizer outputs

## Implementation Details

### Pipeline Components
1. **JsonlReader**: Read from HuggingFace C4
2. **DocumentTokenizer**: Tokenize with GPT-2 tokenizer
3. **TokensCounter**: Count and report token statistics
4. **JsonlWriter**: Save results

### Data Source
- Continue using C4 dataset
- Same shard as Example 1 for consistency
- `hf://datasets/allenai/c4/en/`

### Pipeline Flow
```
JsonlReader("hf://datasets/allenai/c4/en/", limit=1000)
    ↓
TokensCounter(tokenizer_name="gpt2")  # Count before
    ↓
LambdaFilter(lambda doc: doc.metadata["tokens"] > 50)  # Filter by token count
    ↓
TokensCounter(tokenizer_name="gpt2")  # Count after
    ↓
JsonlWriter("output/tokenized/")
```

## Files to Create
1. `examples_local/03_tokenization.py` - Main pipeline

## Execution Plan
1. Read C4 data subset
2. Count tokens before filtering
3. Filter by token count
4. Count tokens after filtering
5. Compare statistics

## Success Metrics
- [x] Token counts calculated correctly
- [x] Filter works based on token count
- [x] Statistics show before/after differences
- [x] Understanding of tokenizer integration

## Notes
- GPT-2 tokenizer is commonly used
- Token counts affect model training costs
- Different tokenizers give different counts

## Implementation Notes (Completed)
- Processed 1000 C4 documents with GPT-2 tokenizer
- Filtered to keep docs with 50-2048 tokens
- Results: 1000 → 922 documents, ~380K total tokens
- Average 413 tokens/doc, token/word ratio ~1.22
- Fixed TokensCounter parameter issues (tokenizer_name_or_path not tokenizer_name)
- Compared GPT-2 vs BERT tokenizers on same text