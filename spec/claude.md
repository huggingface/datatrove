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
- [x] Successfully run all local examples
- [x] Understand pipeline composition and data flow
- [x] Convert at least one example to Slurm
- [x] Run distributed processing with load balancing
- [x] Collect and analyze statistics from processed data
- [x] Implement custom filtering or processing logic

## Notes
- Start with minimal data to ensure quick iteration
- Focus on understanding concepts over processing large datasets
- Document any issues or learnings in each spec file
- Keep examples self-contained and reproducible

## Current Progress

### Environment Setup
- **Conda Environment:** `datatrove-learn` (Python 3.10)
- **Installation:** `pip install -e ".[processing,io]"`
- **Current Branch:** `learning/phase2-slurm-distributed`
- **Phase 2 Start Date:** 2025-09-23

### Phase 1: Local Examples Status

| Example | Status | Implementation | Notes |
|---------|--------|---------------|-------|
| 1. Basic Filtering | ✅ Complete | `examples_local/01_basic_filtering.py` | Successfully processes C4 data: 1000→77 docs |
| 2. Text Extraction | ✅ Complete | `examples_local/02_text_extraction.py` | WARC processing with quality filters: 200→8 docs |
| 3. Tokenization | ✅ Complete | `examples_local/03_tokenization.py` | Token counting with GPT-2: 1000→922 docs, ~380K tokens |
| 4. Statistics | ✅ Complete | `examples_local/04_statistics.py` | Collects doc/word/line/language stats: 922 docs analyzed |
| 5. Deduplication | ✅ Complete | `examples_local/05_deduplication.py` | Hash-based dedup: 14→10 docs (28.6% removed), C4: no duplicates in 5000 |

### Phase 2: Slurm/Distributed Status

| Example | Status | Implementation | Notes |
|---------|--------|---------------|-------|
| 6. RunPod Slurm Setup | ✅ Complete | `spec/06_runpod_slurm.md` | Managed Slurm clusters, 2x A100 nodes |
| 6a. Basic Filtering (Slurm) | ✅ Complete | `examples_slurm/01_basic_filtering_slurm.py` | 100→77→5 docs distributed processing |
| 6b. Statistics Collection (Slurm) | ✅ Complete | `examples_slurm/04_statistics_slurm.py` | True load balancing: 200 docs per node |
| 7. Lambda Managed Slurm | ⏸️ Parked | `spec/07_lambda_slurm.md` | 1-week minimum commitment constraint |
| 7b. Lambda Manual Slurm | ✅ Complete | `spec/07b_lambda_manual_slurm.md` | DIY H100 cluster from scratch |
| 7c. DataTrove Execution | ✅ Complete | `spec/07c_datatrove_slurm_execution.md` | Distributed processing on manual clusters |
| 8. Multi-Stage MinHash | ⏳ Future | - | 4-stage deduplication pipeline |

### Quick Start for Next Session
```bash
# Phase 1 (Local) - All complete
python examples_local/01_basic_filtering.py
python examples_local/04_statistics.py

# Phase 2 (Distributed) - Complete
# RunPod: spec/06_runpod_slurm.md (managed clusters)
# Lambda: spec/07b_lambda_manual_slurm.md + 07c_datatrove_slurm_execution.md

# Phase 3: PDF Processing (FinePDFs Reproduction) - In Progress
# Branch: learning/phase3-pdf-pipeline
# See: spec/08_finepdfs_reproduction.md
```

### Phase 3: PDF Processing Status

| Component | Status | Implementation | Notes |
|-----------|--------|---------------|-------|
| PDF WARC Reader | ✅ Complete | `src/.../readers/pdf_warc.py` | PDF-only CommonCrawl filtering |
| PDF Truncation Detector | ✅ Complete | `src/.../filters/pdf_truncation.py` | Identify truncated PDFs |
| PDF Classifier Model Training | ✅ Complete | `spec/08b_pdf_classifier_model.md` | XGBoost model + threshold analysis |
| Basic PDF Pipeline | ⏳ Planned | `examples/finepdfs.py` | Docling-only extraction |