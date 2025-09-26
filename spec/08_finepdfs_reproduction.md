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

### Step 3: Test Existing PDF Classifier
**File**: `tests/pipeline/test_pdf_classifier.py`
**Goal**: Verify `PDFScannedPredictor` works without model file

```python
def test_pdf_feature_extraction():
    # Test feature extraction without XGBoost model
    # Verify 127 features generated correctly
```

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

## Success Criteria
1. Extract PDFs from small WARC file
2. Detect truncated vs complete PDFs
3. Extract text using Docling (no OCR)
4. End-to-end pipeline runs locally
5. All unit tests pass