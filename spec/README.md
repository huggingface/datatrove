# DataTrove Learning Plan

## Overview
Structured approach to learning DataTrove: local development → distributed processing → production pipelines.

**For creating new examples:** See [GUIDELINES.md](GUIDELINES.md) for spec and implementation standards.

## Learning Phases

### Phase 1: Local Development ✅ Complete
**Location:** `spec/phase1/*.md`
**Goal:** Understand DataTrove basics with local examples
**Topics:** Readers, filters, writers, tokenization, statistics, deduplication
**Examples:** 5 progressively complex pipelines (01-05)

### Phase 2: Distributed Processing ✅ Complete
**Location:** `spec/phase2/*.md`
**Goal:** Scale to Slurm clusters (RunPod, Lambda)
**Topics:** SlurmPipelineExecutor, cluster setup, distributed processing
**Examples:** Managed and manual Slurm setups (06-07c)

### Phase 3: PDF Processing ✅ Complete
**Location:** `spec/phase3/*.md`
**Goal:** Production PDF extraction pipeline (FinePDFs reproduction)
**Topics:** XGBoost routing, Docling, RolmOCR, Media objects
**Examples:** Classification, extraction, two-tiered routing (08a-08e)

## Directory Structure
```
spec/
├── claude.md                  # This file - high-level roadmap
├── phase1/*.md               # Local pipeline examples + implementations
├── phase2/*.md               # Slurm/distributed examples + implementations
└── phase3/*.md               # PDF processing examples + implementations

docs/
├── design/*.md               # Design decisions (pdf_router, media_objects, etc)
└── guides/*.md               # Setup guides (lambda_setup, etc)

examples/
└── *.py                      # Production-ready pipelines
```

## Status Summary

| Phase | Status | Key Achievements |
|-------|--------|-----------------|
| Phase 1: Local | ✅ Complete | 5 examples: filtering, extraction, tokenization, stats, dedup |
| Phase 2: Distributed | ✅ Complete | Slurm setup (RunPod/Lambda), distributed processing |
| Phase 3: PDF Processing | ✅ Complete | XGBoost routing, Docling/RolmOCR, Media objects, production pipeline |