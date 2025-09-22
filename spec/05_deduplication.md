# Example 5: Deduplication Pipeline (Simplified)

## Objective
Learn deduplication techniques starting with exact matching, preparing for distributed MinHash.

## Learning Goals
- Understand exact deduplication
- Learn about sentence-level dedup
- Introduction to MinHash concepts
- Handle duplicate removal efficiently

## Implementation Details

### Pipeline Components
1. **JsonlReader**: Read input data
2. **SentenceDedupFilter**: Remove duplicate sentences
3. **ExactDedupFilter**: Remove exact document matches
4. **JsonlWriter**: Save deduplicated results

### Data Source
- Use C4 subset with intentional duplicates
- Or create a dataset with some repeated documents
- Can duplicate some C4 docs manually for testing

### Pipeline Flow (Simple Exact Dedup)
```
JsonlReader("hf://datasets/allenai/c4/en/", limit=2000)
    ↓
SentenceDedupFilter()  # Remove duplicate sentences within docs
    ↓
JsonlWriter("output/dedup/")
```

### Alternative: URL-based Deduplication
```
JsonlReader(input_data)
    ↓
UrlDedupFilter()  # Remove documents with duplicate URLs
    ↓
JsonlWriter("output/url_dedup/")
```

## Deduplication Strategies

### Exact Deduplication
- Hash entire document text
- Remove exact matches
- Fast but misses near-duplicates

### Sentence Deduplication
- Remove duplicate sentences within/across documents
- Helps with boilerplate removal
- Good for cleaning web text

### URL Deduplication
- Remove documents from same URL
- Useful for web crawl data
- Simple but effective

## Files to Create
1. `examples_local/05_deduplication.py` - Main pipeline
2. `examples_local/05_create_duplicates.py` - Helper to add duplicates for testing

## Execution Plan
1. Create dataset with known duplicates
2. Run sentence dedup
3. Check statistics before/after
4. Verify duplicates removed

## Success Metrics
- [ ] Duplicates correctly identified
- [ ] Output has unique documents
- [ ] Statistics show reduction
- [ ] Understanding of dedup trade-offs

## Notes
- Exact dedup is memory-limited locally
- MinHash (Example 7) handles fuzzy matching
- Sentence dedup good for web text
- Order matters: sentence then document dedup

## Preview of Distributed MinHash
- MinHash creates signatures for fuzzy matching
- Requires multiple stages (signature → buckets → cluster → filter)
- Best suited for Slurm/distributed execution
- We'll implement this in Phase 2