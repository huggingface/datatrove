# FinePDFs Reproduction Plan

## Step-by-Step Implementation

### Step 1: PDF-only CommonCrawl Reader
**File**: `src/datatrove/pipeline/readers/pdf_warc.py`
**Test**: `tests/pipeline/test_pdf_warc_reader.py`
**Goal**: Filter CommonCrawl WARCs to extract only PDF files

```python
# Basic structure - extends existing WarcReader
class PDFWarcReader(WarcReader):
    def __init__(self, mime_types=["application/pdf"], **kwargs):
        # Filter for PDFs only
```

**Test locally**: Use small WARC file, verify PDF extraction

### Step 2: PDF Truncation Detection
**File**: `src/datatrove/pipeline/filters/pdf_truncation.py`
**Test**: `tests/pipeline/test_pdf_truncation.py`
**Goal**: Identify truncated PDFs in CommonCrawl

```python
class PDFTruncationDetector(BaseFilter):
    def filter(self, doc):
        # Check file size < 1MB (pre-2019)
        # Check content_truncated field (post-2019)
```

**Test locally**: Create truncated vs complete PDF samples

### Step 3: Test Existing PDF Classifier ✅
**File**: `tests/pipeline/test_pdf_classifier.py`
**Goal**: Verify `PDFScannedPredictor` works without model file

```python
def test_pdf_feature_extraction():
    # Test feature extraction without XGBoost model
    # Verify 124 features generated correctly (not 127)
```

**Status**: ✅ Complete (with Step 3b: XGBoost model training)
- Found existing classifier needs pre-trained model
- Created comprehensive test suite (9 test cases)
- Discovered real feature dimension is 124, not 127
- **Deep dive**: Built complete XGBoost training pipeline (see `spec/08b_pdf_classifier_model.md`)

### Step 4: Basic PDF Extraction (Docling only)
**File**: `examples/finepdfs_basic.py`
**Goal**: End-to-end pipeline without inference server

```python
# Simplified pipeline: PDFs -> Docling -> Output
# No OCR, no model classification
```

### Step 5: Add PDF Re-fetching (if needed)
**File**: `src/datatrove/pipeline/readers/pdf_refetch.py`
**Goal**: Re-fetch truncated PDFs from original URLs

## Testing Strategy
- Each component has unit tests
- Use small local data files
- Test components independently before integration
- No inference servers until basic pipeline works

## File Structure
```
examples/finepdfs.py              # Main pipeline (follows fineweb.py structure)
src/datatrove/pipeline/readers/pdf_warc.py
src/datatrove/pipeline/filters/pdf_truncation.py
tests/pipeline/test_pdf_*.py      # Unit tests for each component
```

## Implementation Status

| Step | Component | Status | Implementation | Notes |
|------|-----------|--------|---------------|-------|
| 1 | PDF WARC Reader | ✅ Complete | `src/.../readers/pdf_warc.py` | PDF-only CommonCrawl filtering |
| 1 | Reader Tests | ✅ Complete | `tests/.../test_pdf_warc_reader.py` | Unit tests with real WARC data |
| 2 | PDF Truncation Detector | ✅ Complete | `src/.../filters/pdf_truncation.py` | Identifies truncated PDFs |
| 2 | Truncation Tests | ✅ Complete | `tests/.../test_pdf_truncation.py` | Unit tests for filter logic |
| 3 | PDF Classifier Tests | ✅ Complete | `tests/.../test_pdf_classification.py` | 9 test cases, found 124 features not 127 |
| 3b | XGBoost Model Training | ✅ Complete | `spec/08b_pdf_classifier_model.md` | Deep dive: training pipeline + analysis |
| 4 | Basic PDF Pipeline | ⏳ Next | `examples/finepdfs_basic.py` | Docling-only, no OCR routing |
| 5 | PDF Re-fetching | ⏳ Future | `src/.../readers/pdf_refetch.py` | Re-fetch truncated PDFs |

## Course Correction

**Original Plan**: Simple pipeline testing without ML complexity
**What Happened**: Deep dive into XGBoost model training (Step 3b)
**Next**: Return to simple Docling-only pipeline (Step 4)