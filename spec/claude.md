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
├── spec/                       # Learning specifications and examples
│   ├── claude.md              # This file - overall learning plan
│   ├── phase1/                # Phase 1: Local examples
│   │   ├── 01_basic_filtering.md
│   │   ├── 02_extraction_filtering.md
│   │   ├── 03_tokenization.md
│   │   ├── 04_statistics.md
│   │   ├── 05_deduplication.md
│   │   ├── examples/          # Python implementations
│   │   ├── data/              # Sample data (gitignored)
│   │   ├── logs/              # Runtime logs (gitignored)
│   │   └── output/            # Results (gitignored)
│   ├── phase2/                # Phase 2: Slurm/distributed
│   │   ├── 06_runpod_slurm.md
│   │   ├── 07_lambda_slurm.md
│   │   ├── 07b_lambda_manual_slurm.md
│   │   ├── 07c_datatrove_slurm_execution.md
│   │   ├── examples/
│   │   ├── data/
│   │   ├── logs/
│   │   └── output/
│   └── phase3/                # Phase 3: PDF processing
│       ├── 08_finepdfs_reproduction.md
│       ├── 08b_pdf_classifier_model.md
│       ├── 08c_rolmocr_integration.md
│       ├── 08d_routing_pipeline.md
│       ├── 08e_media_objects_refactor.md
│       ├── examples/          # Includes utils/ subfolder
│       ├── data/
│       ├── logs/
│       └── output/
├── docs/                      # Documentation
│   ├── design/                # Design decision documents
│   │   ├── pdf_router.md
│   │   ├── media_objects.md
│   │   └── inference_runner.md
│   └── guides/
│       └── lambda_setup.md
└── examples/                  # Production-ready examples
    └── finepdfs.py
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
| 1. Basic Filtering | ✅ Complete | `spec/phase1/examples/01_basic_filtering.py` | Successfully processes C4 data: 1000→77 docs |
| 2. Text Extraction | ✅ Complete | `spec/phase1/examples/02_text_extraction.py` | WARC processing with quality filters: 200→8 docs |
| 3. Tokenization | ✅ Complete | `spec/phase1/examples/03_tokenization.py` | Token counting with GPT-2: 1000→922 docs, ~380K tokens |
| 4. Statistics | ✅ Complete | `spec/phase1/examples/04_statistics.py` | Collects doc/word/line/language stats: 922 docs analyzed |
| 5. Deduplication | ✅ Complete | `spec/phase1/examples/05_deduplication.py` | Hash-based dedup: 14→10 docs (28.6% removed), C4: no duplicates in 5000 |

### Phase 2: Slurm/Distributed Status

| Example | Status | Implementation | Notes |
|---------|--------|---------------|-------|
| 6. RunPod Slurm Setup | ✅ Complete | `spec/phase2/06_runpod_slurm.md` | Managed Slurm clusters, 2x A100 nodes |
| 6a. Basic Filtering (Slurm) | ✅ Complete | `spec/phase2/examples/01_basic_filtering_slurm.py` | 100→77→5 docs distributed processing |
| 6b. Statistics Collection (Slurm) | ✅ Complete | `spec/phase2/examples/04_statistics_slurm.py` | True load balancing: 200 docs per node |
| 7. Lambda Managed Slurm | ⏸️ Parked | `spec/phase2/07_lambda_slurm.md` | 1-week minimum commitment constraint |
| 7b. Lambda Manual Slurm | ✅ Complete | `spec/phase2/07b_lambda_manual_slurm.md` | DIY H100 cluster from scratch |
| 7c. DataTrove Execution | ✅ Complete | `spec/phase2/07c_datatrove_slurm_execution.md` | Distributed processing on manual clusters |
| 8. Multi-Stage MinHash | ⏳ Future | - | 4-stage deduplication pipeline |

### Quick Start for Next Session
```bash
# Phase 1 (Local) - All complete
python spec/phase1/examples/01_basic_filtering.py
python spec/phase1/examples/04_statistics.py

# Phase 2 (Distributed) - Complete
python spec/phase2/examples/01_basic_filtering_slurm.py
# See: spec/phase2/06_runpod_slurm.md, spec/phase2/07b_lambda_manual_slurm.md

# Phase 3: PDF Processing - ✅ COMPLETED
python spec/phase3/examples/08_finepdfs_local.py
python spec/phase3/examples/08d_docling_test.py
python spec/phase3/examples/08c_rolmocr_test.py
# See: spec/phase3/08_finepdfs_reproduction.md, spec/phase3/08d_routing_pipeline.md
```

### Phase 3: PDF Processing Status ✅ COMPLETED (2025-10-01)

| Component | Status | Implementation | Notes |
|-----------|--------|---------------|-------|
| PDF WARC Reader | ⏭️ Skipped | `src/.../readers/pdf_warc.py` | Using WarcReaderFast instead (see 08e) |
| PDF Truncation Detector | ✅ Complete | `src/.../filters/pdf_truncation.py` | Identify truncated PDFs |
| PDF Classifier Model Training | ✅ Complete | `spec/phase3/08b_pdf_classifier_model.md` | XGBoost model + threshold analysis |
| Docling Component Testing | ✅ Complete | `spec/phase3/examples/08d_docling_test.py` | Refactored to use Media objects |
| RolmOCR Component Testing | ✅ Complete | `spec/phase3/examples/08c_rolmocr_test.py` | Refactored to use Media objects |
| PDFRouter Component | ✅ Complete | `src/.../filters/pdf_router.py` | XGBoost classification with Media objects |
| Routing Pipeline Spec | ✅ Complete | `spec/phase3/08d_routing_pipeline.md` | Three-stage architecture defined |
| **Media Objects Refactor** | ✅ **Complete** | `spec/phase3/08e_media_objects_refactor.md` | Media.__post_init__ base64 decoding |
| Two-tiered Routing Pipeline (Test) | ✅ **Complete** | `spec/phase3/examples/08_finepdfs_local.py` | Tested on Lambda with 6 PDFs |
| Two-tiered Routing Pipeline (Prod) | ✅ **Complete** | `examples/finepdfs.py` | Production-ready pipeline |
| **Code Refactoring** | ✅ **Complete** | Multiple files | Moved duplicated code to proper locations |

**Refactoring (2025-10-02)**:
- ✅ `PersistentContextJsonlWriter` → `src/datatrove/pipeline/writers/jsonl.py`
- ✅ `ExtractInferenceText` → `src/datatrove/pipeline/inference/post_process.py`
- ✅ `rolmocr_query_builder` → `src/datatrove/pipeline/inference/query_builders/vision.py`
- ✅ All test scripts verified working with refactored imports

**Test Results** (Lambda - 6 PDFs):
- ✅ Classification: 3 low OCR, 3 high OCR routed correctly
- ✅ Docling: 24.6K, 9.6K, 1.5K chars extracted
- ✅ RolmOCR: 5.9K, 19K, 1.2K chars extracted
- ✅ PDFs and PNGs saved for cross-reference