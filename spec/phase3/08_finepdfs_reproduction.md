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
- **Deep dive**: Built complete XGBoost training pipeline (see `spec/phase3/08b_pdf_classifier_model.md`)

### Step 4: Full PDF Processing Pipeline with XGBoost Routing
**File**: `examples/finepdfs.py`
**Goal**: Complete pipeline with trained XGBoost model routing to Docling or OCR

#### Step 4a: Test Docling Component ✅
- Fixed Docling-sync integration issues (version conflicts, import errors)
- Fixed critical JsonlReader metadata bug (line 77) that was dropping WARC metadata
- DoclingExtractor works on Linux A100 with OpenVINO (no macOS ARM64 GridSample issues)
- Tested across OCR probability thresholds: very_low_ocr (0.001) extracts 9,187 chars, high_ocr (0.758) extracts 2,173 chars
- **Key finding**: DoclingExtractor doesn't auto-route - we need to implement XGBoost routing

#### Step 4b: Test OCR Component
- Setup Reducto/RolmOCR extraction for scanned PDFs
- Test with scanned PDFs (OCR prob > 0.5)

#### Step 4c: Lambda OCR Server
- Spin up Lambda server for OCR processing
- Test remote OCR functionality

#### Step 4d: Full Pipeline Integration
```python
# Complete pipeline: PDFs -> XGBoost Classifier -> Docling/OCR -> Output
reader = PDFWarcReader(...)
truncation_filter = PDFTruncationDetector()
classifier = PDFScannedPredictor(path_to_model="spec/phase3/examples/pdf_classifier_real_data.xgb")
# Route to Docling or RolmOCR based on classification
```

#### Step 4e: Process All 3 WARC Files
- Run complete pipeline on all CommonCrawl data
- Generate samples from each processing stage
- Document output quality and routing decisions

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
| 3b | XGBoost Model Training | ✅ Complete | `spec/phase3/08b_pdf_classifier_model.md` | Deep dive: training pipeline + analysis |
| 4a | Test Docling Component | ✅ Complete | `spec/phase3/examples/08d_docling_test.py` | DoclingExtractor working on Linux A100 with OpenVINO |
| 4b | Test OCR Component | ✅ Complete | `spec/phase3/examples/08c_rolmocr_test.py` | RolmOCR integrated with PersistentContextJsonlWriter fix |
| 4c | Lambda OCR Server | ✅ Complete | LMDeploy integration | RolmOCR on LMDeploy with DataTrove InferenceRunner |
| 4d | Full Pipeline Integration | ✅ Complete | `examples/finepdfs.py` + `spec/phase3/examples/08_finepdfs_local.py` | Three-stage routing pipeline tested on Lambda |
| 4d+ | Code Refactoring | ✅ Complete | Multiple files | Moved duplicated code to proper repo locations |
| 4e | Process All WARC Files | ⏳ Next | Pipeline execution | Complete dataset with samples |
| 5 | PDF Re-fetching | ⏳ Future | `src/.../readers/pdf_refetch.py` | Re-fetch truncated PDFs |

## Course Correction

**Original Plan**: Simple Docling-only pipeline without ML complexity
**What Happened**: Deep dive into XGBoost model training (Step 3b)
**New Plan**: Use trained XGBoost model for intelligent routing (Step 4)
- Test both Docling and OCR extraction paths
- Full pipeline with classification-based routing
- Complete evaluation on all CommonCrawl data