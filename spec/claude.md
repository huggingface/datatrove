# DataTrove Learning Plan

## Overview
This document outlines our structured approach to learning the DataTrove framework, progressing from simple local examples to complex distributed processing using Slurm.

## Goals
1. Understand the core concepts of DataTrove (pipelines, executors, documents)
2. Get hands-on experience with different pipeline components
3. Learn how to scale from local to distributed processing
4. Build familiarity with common data processing patterns

## Learning Path

### Phase 1: Local Development (MacBook Pro)
Running examples locally to understand the framework basics without infrastructure complexity.

#### Example 1: Basic Data Processing Pipeline
**File:** `spec/01_basic_filtering.md`
- **Purpose:** Learn the fundamentals of readers, filters, and writers
- **Components:** JsonlReader → LambdaFilter → JsonlWriter
- **Data:** Small sample dataset (create synthetic data or use a small public dataset)
- **Key Learnings:**
  - Document structure (text, id, metadata)
  - Pipeline composition
  - Basic I/O operations

#### Example 2: Text Extraction and Quality Filtering
**File:** `spec/02_extraction_filtering.md`
- **Purpose:** Work with real-world text processing
- **Components:** HtmlReader → Trafilatura → LanguageFilter → QualityFilter → JsonlWriter
- **Data:** Sample HTML files or small web scrape
- **Key Learnings:**
  - Text extraction from HTML
  - Chaining multiple filters
  - Quality assessment
  - Language detection

#### Example 3: Tokenization Pipeline
**File:** `spec/03_tokenization.md`
- **Purpose:** Understand token processing for ML workflows
- **Components:** JsonlReader → DocumentTokenizer → TokensCounter
- **Data:** Small text corpus
- **Key Learnings:**
  - Tokenization for LLMs
  - Token counting and statistics
  - Working with different tokenizers

#### Example 4: Statistics Collection
**File:** `spec/04_statistics.md`
- **Purpose:** Analyze dataset characteristics
- **Components:** JsonlReader → DocStats → WordStats → LineStats → StatsMerger
- **Data:** Medium-sized text dataset
- **Key Learnings:**
  - Collecting various statistics
  - Understanding data profiles
  - Merging statistics from parallel tasks

#### Example 5: Deduplication Pipeline (Simplified)
**File:** `spec/05_deduplication.md`
- **Purpose:** Remove duplicate content
- **Components:** JsonlReader → ExactDedupFilter → JsonlWriter
- **Data:** Dataset with intentional duplicates
- **Key Learnings:**
  - Deduplication strategies
  - Memory-efficient processing
  - Handling large-scale dedup (preparation for distributed)

### Phase 2: Scaling to Slurm

#### Example 6: Converting Local to Slurm
**File:** `spec/06_local_to_slurm.md`
- **Purpose:** Transform Example 1 to run on Slurm
- **Changes Required:**
  - Replace LocalPipelineExecutor with SlurmPipelineExecutor
  - Configure Slurm parameters
  - Set up logging directories
  - Handle S3/remote storage

#### Example 7: Multi-Stage Pipeline with Dependencies
**File:** `spec/07_multistage_slurm.md`
- **Purpose:** Complex workflow with dependent jobs
- **Implementation:** Minhash deduplication (4 stages)
- **Key Learnings:**
  - Job dependencies
  - Stage coordination
  - Resource allocation per stage

#### Example 8: Large-Scale Processing
**File:** `spec/08_production_pipeline.md`
- **Purpose:** Production-ready pipeline
- **Components:** Full FineWeb-style pipeline
- **Key Learnings:**
  - Best practices for production
  - Error handling and recovery
  - Performance optimization

## Execution Order

1. **Setup Phase**
   - Install DataTrove with required dependencies
   - Create sample datasets
   - Set up local working directories

2. **Local Examples (Phase 1)**
   - Run examples 1-5 sequentially
   - Each builds on previous knowledge
   - Gradually increase complexity

3. **Transition Phase**
   - Review Slurm concepts
   - Set up Slurm access (if available)
   - Prepare S3/remote storage

4. **Distributed Examples (Phase 2)**
   - Convert local examples to Slurm
   - Run multi-stage pipelines
   - Implement production patterns

## Directory Structure
```
datatrove/
├── spec/                      # Specifications
│   ├── claude.md             # This file
│   ├── 01_basic_filtering.md
│   ├── 02_extraction_filtering.md
│   └── ...
├── examples_local/           # Local example implementations
│   ├── 01_basic_filtering.py
│   ├── data/                # Sample data
│   └── output/              # Results
└── examples_slurm/          # Slurm example implementations
    ├── 06_local_to_slurm.py
    └── ...
```

## Success Criteria
- [ ] Successfully run all local examples
- [ ] Understand pipeline composition and data flow
- [ ] Convert at least one example to Slurm
- [ ] Run a multi-stage distributed pipeline
- [ ] Collect and analyze statistics from processed data
- [ ] Implement custom filtering or processing logic

## Notes
- Start with minimal data to ensure quick iteration
- Focus on understanding concepts over processing large datasets
- Document any issues or learnings in each spec file
- Keep examples self-contained and reproducible

## Current Progress

### Environment Setup
- **Conda Environment:** `datatrove-learn` (Python 3.10)
- **Installation:** `pip install -e ".[processing,io]"`
- **Current Branch:** `learning/phase1-local-examples`

### Phase 1: Local Examples Status

| Example | Status | Implementation | Notes |
|---------|--------|---------------|-------|
| 1. Basic Filtering | ✅ Complete | `examples_local/01_basic_filtering.py` | Successfully processes C4 data: 1000→77 docs |
| 2. Text Extraction | ✅ Complete | `examples_local/02_text_extraction.py` | WARC processing with quality filters: 200→8 docs |
| 3. Tokenization | ✅ Complete | `examples_local/03_tokenization.py` | Token counting, filtering by token count |
| 4. Statistics | ⏳ Not Started | - | - |
| 5. Deduplication | ⏳ Not Started | - | - |

### Quick Start for Next Session
```bash
# Activate environment
conda activate datatrove-learn

# Run Example 1
python examples_local/01_basic_filtering.py

# Continue with Example 2
# (See spec/02_extraction_filtering.md)
```