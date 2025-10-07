# Example 2: Text Extraction and Quality Filtering

## Objective
Work with real Common Crawl data to extract text and apply quality filters.

## Learning Goals
- Read WARC files from Common Crawl samples
- Extract clean text using Trafilatura
- Apply language and quality filters
- Understand exclusion writing for analysis

## Implementation Details

### Pipeline Components
1. **WarcReader**: Read Common Crawl WARC files
2. **Trafilatura**: Extract text from HTML
3. **LanguageFilter**: Keep only English
4. **GopherQualityFilter**: Apply quality heuristics
5. **JsonlWriter**: Save results

### Data Source
- Use Common Crawl sample files (small WARC files ~1MB each)
- Download from: https://commoncrawl.org/get-started
- Or use local sample: `wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/1707947474671.63/warc/CC-MAIN-20240224032749-20240224062749-00000.warc.gz`
- This is real web data - no synthetic generation needed

### Pipeline Flow
```
WarcReader("data/CC-MAIN-sample.warc.gz")
    ↓
Trafilatura()
    ↓
LanguageFilter(languages=["en"])
    ↓
GopherQualityFilter()
    ↓
JsonlWriter("output/clean/")
```

## Files to Create
1. `spec/phase1/examples/02_extraction_filtering.py` - Main pipeline
2. Download a sample WARC file to `spec/phase1/examples/data/`

## Execution Plan
1. Download one Common Crawl sample WARC file
2. Run extraction pipeline locally
3. Check output size and quality
4. Add exclusion writers to see what gets filtered

## Success Metrics
- [x] Successfully reads WARC file
- [x] Extracts text from HTML records
- [x] Filters apply correctly
- [x] Output contains clean text documents

## Notes
- WARC files contain real web data - expect varied quality
- Single WARC sample is small enough for local testing
- This mirrors production Common Crawl processing

## Implementation Notes (Completed)
- Downloaded real Common Crawl WARC file (CC-MAIN-20180420081400-20180420101400-00000.warc.gz)
- Processed 200 records with Trafilatura extraction
- Applied GopherRepetition and GopherQuality filters
- Results: 200 → 8 clean documents (96% filtered - very aggressive)
- Learned about dependencies: libmagic (OS level), spacy (for repetition filter)
- Discovered filtered content was mostly adult/spam with repetitive SEO patterns