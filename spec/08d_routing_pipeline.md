# Two-Tiered PDF Processing Pipeline

## Overview
Implement intelligent routing of PDFs based on XGBoost classifier predictions. Low OCR probability PDFs go to Docling for direct text extraction, high OCR probability PDFs go to RolmOCR for GPU-based OCR.

## Architecture

Following the FineWeb pattern of multi-stage dependent pipelines:

```
Stage 1: Classification
PDFWarcReader → PDFTruncationDetector → PDFRouter → JsonlWriter(classified/)

Stage 2: Low OCR Path (depends on Stage 1)
JsonlReader(classified/) → ConditionalFilter(low_ocr) → DoclingExtractor → JsonlWriter(text_extraction/)

Stage 3: High OCR Path (depends on Stage 1)
JsonlReader(classified/) → ConditionalFilter(high_ocr) → RolmOCR → JsonlWriter(ocr_extraction/)
```

**Benefits**:
- PDFs classified once, intermediate results saved
- Stages 2 & 3 can run in parallel after Stage 1
- Follows DataTrove's linear pipeline model with dependencies

## Components Already Built

### 1. Input Processing ✅
- **PDFWarcReader** - Streams PDFs from CommonCrawl WARC files
- **PDFTruncationDetector** - Filters out truncated PDFs

### 2. Classification ✅
- **XGBoost Model** - Trained on real data (`pdf_classifier_real_data.xgb`)
- **Features** - 124-dimensional feature vector (7 doc + 117 page features)
- **Threshold** - 0.5 optimal split point

### 3. Low OCR Path ✅
- **DoclingExtractor** - Direct text extraction
- Working on Linux A100 with OpenVINO
- Tested across OCR thresholds (very_low: 9,187 chars, high: 2,173 chars)

### 4. High OCR Path ✅
- **RolmOCR** - GPU-based OCR on LMDeploy
- PersistentContextJsonlWriter for multi-document support
- FinePDFs preprocessing (1280px, 2048 visual tokens)

## Implementation Plan

### Step 1: Create Routing Filter
**File**: `src/datatrove/pipeline/filters/pdf_router.py`

```python
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.media.predictor.scanned_pdf_predictor import PDFScannedPredictor

class PDFRouter(PipelineStep):
    """Route PDFs based on OCR probability to different processing paths."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        low_ocr_tag: str = "text_extraction",
        high_ocr_tag: str = "ocr_extraction"
    ):
        super().__init__()
        self.predictor = PDFScannedPredictor(path_to_model=model_path)
        self.threshold = threshold
        self.low_ocr_tag = low_ocr_tag
        self.high_ocr_tag = high_ocr_tag

    def run(self, data, rank=0, world_size=1):
        for doc in data:
            # Get OCR probability
            ocr_prob = self.predictor.predict_single_document(doc)

            # Add routing metadata
            doc.metadata["ocr_probability"] = ocr_prob
            doc.metadata["processing_route"] = (
                self.high_ocr_tag if ocr_prob >= self.threshold
                else self.low_ocr_tag
            )

            self.stat_update(f"routed_to_{doc.metadata['processing_route']}")
            yield doc
```

### Step 2: Use LambdaFilter for Conditional Routing
**File**: Use existing `src/datatrove/pipeline/filters/lambda_filter.py`

No new filter needed - LambdaFilter already provides this functionality:

```python
from datatrove.pipeline.filters.lambda_filter import LambdaFilter

# Filter for low OCR probability PDFs
low_ocr_filter = LambdaFilter(
    filter_function=lambda doc: doc.metadata.get("processing_route") == "text_extraction"
)

# Filter for high OCR probability PDFs
high_ocr_filter = LambdaFilter(
    filter_function=lambda doc: doc.metadata.get("processing_route") == "ocr_extraction"
)
```

### Step 3: Create Three-Stage Pipeline
**File**: `examples/finepdfs.py`

Following the FineWeb pattern of dependent stages:

```python
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers.pdf_warc import PDFWarcReader
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters.pdf_truncation import PDFTruncationDetector
from datatrove.pipeline.filters.pdf_router import PDFRouter
from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from datatrove.pipeline.extractors.docling import DoclingExtractor
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceConfig
from datatrove.pipeline.writers.jsonl import JsonlWriter

# Shared paths
WARC_INPUT = "data/warcs"
CLASSIFIED_OUTPUT = "output/classified"
TEXT_EXTRACTION_OUTPUT = "output/text_extraction"
OCR_EXTRACTION_OUTPUT = "output/ocr_extraction"
MODEL_PATH = "examples_local/pdf_classifier_real_data.xgb"

# Stage 1: Classify all PDFs and save with routing metadata
stage1_classification = LocalPipelineExecutor(
    job_name="pdf_classification",
    pipeline=[
        PDFWarcReader(
            data_folder=WARC_INPUT,
            glob_pattern="*.warc.gz",
        ),
        PDFTruncationDetector(),
        PDFRouter(
            model_path=MODEL_PATH,
            threshold=0.5
        ),
        JsonlWriter(CLASSIFIED_OUTPUT),  # Save ALL with metadata
    ],
    tasks=4,
    logging_dir="logs/classification"
)

# Stage 2: Process low OCR PDFs (text extraction path)
stage2_text_extraction = LocalPipelineExecutor(
    job_name="pdf_text_extraction",
    pipeline=[
        JsonlReader(CLASSIFIED_OUTPUT),  # Read pre-classified PDFs
        LambdaFilter(
            filter_function=lambda doc: doc.metadata.get("processing_route") == "text_extraction"
        ),
        DoclingExtractor(),
        JsonlWriter(TEXT_EXTRACTION_OUTPUT),
    ],
    tasks=4,  # Parallel CPU processing
    logging_dir="logs/text_extraction",
    depends=stage1_classification  # Wait for classification
)

# Stage 3: Process high OCR PDFs (OCR path)
stage3_ocr_extraction = LocalPipelineExecutor(
    job_name="pdf_ocr_extraction",
    pipeline=[
        JsonlReader(CLASSIFIED_OUTPUT),  # Read same pre-classified PDFs
        LambdaFilter(
            filter_function=lambda doc: doc.metadata.get("processing_route") == "ocr_extraction"
        ),
        InferenceRunner(
            query_builder=rolmocr_query_builder,
            config=InferenceConfig(
                server_type="lmdeploy",
                model_name_or_path="Reducto/RolmOCR",
                model_max_context=8096,
                max_concurrent_requests=1,
                max_concurrent_tasks=1,
                model_kwargs={"chat_template": "internlm"},
            ),
            post_process_steps=[
                PostProcessOCRResults(),
                PersistentContextJsonlWriter(OCR_EXTRACTION_OUTPUT)
            ]
        ),
    ],
    tasks=1,  # GPU-bound, single task
    logging_dir="logs/ocr_extraction",
    depends=stage1_classification  # Wait for classification
)

# Run the pipeline (stages 2 & 3 run in parallel after stage 1)
stage3_ocr_extraction.run()
```

## Testing Strategy

### Phase 1: Component Testing
1. Test PDFRouter with known PDFs
2. Verify ConditionalFilter logic
3. Validate metadata propagation

### Phase 2: Small-Scale Integration
1. Use 10-20 sample PDFs with known OCR probabilities
2. Verify correct routing (low/high OCR)
3. Check both output paths produce results
4. Compare output quality

### Phase 3: Full WARC Processing
1. Process complete WARC files
2. Collect statistics on routing distribution
3. Sample outputs from each route for quality check
4. Document processing times

## Expected Results

### Routing Distribution (from threshold analysis)
- **Low OCR (< 0.5)**: ~75% of PDFs → Docling
- **High OCR (≥ 0.5)**: ~25% of PDFs → RolmOCR

### Performance Targets
- **Docling path**: Fast, parallel processing (4+ workers)
- **RolmOCR path**: GPU-bound, fewer workers (1-2)
- **Overall**: Better quality than single-path approach

### Output Structure
```
output/
├── text_extraction/
│   └── 00000.jsonl.gz  (Low OCR PDFs - Docling)
└── ocr_extraction/
    └── 00000.jsonl.gz  (High OCR PDFs - RolmOCR)
```

## Files to Create

| File | Type | Purpose |
|------|------|---------|
| `src/.../filters/pdf_router.py` | New | Route PDFs based on XGBoost prediction |
| `src/.../filters/lambda_filter.py` | Existing | Filter by metadata field (already available) |
| `examples/finepdfs.py` | New | Main two-tiered pipeline |
| `examples_local/test_routing.py` | New | Test routing logic with samples |
| `tests/pipeline/test_pdf_router.py` | New | Unit tests for router |

## Implementation Steps

1. ✅ Review existing components (all built)
2. ✅ **Create PDFRouter filter** - XGBoost classification integration
3. ✅ **Use LambdaFilter** - Leverage existing metadata-based filtering
4. **Test routing logic** - Small sample validation
5. **Create main pipeline** - Full integration
6. **Test with sample WARCs** - End-to-end validation
7. **Process full dataset** - Production run with statistics

## Success Criteria

- ✅ PDFs correctly routed based on OCR probability
- ✅ Both paths produce valid output
- ✅ Output quality matches/exceeds single-path approach
- ✅ Processing statistics collected and analyzed
- ✅ Sample outputs saved for evaluation

## Notes

- Start with small test set (10-20 PDFs) before full processing
- Monitor GPU memory for RolmOCR path
- Consider chunking for large WARC files
- Save routing statistics for analysis
- Keep sample PDFs from each route for quality comparison