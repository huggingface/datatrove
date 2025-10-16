# Step 3b: PDF Classifier Model Training

## Goal
Test existing `PDFScannedPredictor` and create missing XGBoost model for OCR routing.

## Background
- Step 3 found existing PDF classifier at `src/datatrove/pipeline/media/predictor/scanned_pdf_predictor.py`
- Classifier expects pre-trained XGBoost model but none exists in repo
- Need model to route PDFs: text extraction vs OCR

## Implementation

### Test Existing Classifier
**File**: `tests/pipeline/test_pdf_classification.py`
- Verify `PDFScannedPredictor` initialization
- Test feature extraction (124 features, not 127)
- Test with real CommonCrawl PDFs
- Mock XGBoost model for testing

### Train XGBoost Model
**File**: `spec/phase3/examples/08b_pdf_classifier_training.py`
- Process CommonCrawl PDFs for training data
- Extract 124 features per PDF using existing extractor
- Generate heuristic labels (scanner metadata, forms, garbled text)
- Train XGBoost classifier
- Save model and metadata

### Analyze Feature Distribution
**File**: `spec/phase3/examples/08b_pdf_feature_analysis.py`
- Validate feature dimensions across real PDFs
- Analyze page count distribution
- Check feature extraction success rates
- Document real-world characteristics

### Threshold Analysis
**File**: `spec/phase3/examples/08b_pdf_threshold_analysis.py`
- Process all CommonCrawl PDFs
- Generate OCR probability distribution
- Extract sample PDFs for each threshold range
- Create visualizations for manual evaluation

## Expected Results

### Feature Analysis
- Feature dimension: 124 (7 document + 117 page features)
- Success rate: ~50% (CommonCrawl PDFs are often corrupted)
- Page distribution: Most PDFs ≤8 pages

### Classification Distribution
- Very low OCR (0.0-0.2): ~80% - text extraction
- Very high OCR (0.8-1.0): ~15% - needs OCR
- Medium ranges: ~5% - edge cases
- Optimal threshold: 0.5

### Files Generated
```
spec/phase3/
├── examples/
│   ├── 08b_pdf_classifier_training.py      # Training script
│   ├── 08b_pdf_feature_analysis.py         # Feature analysis
│   └── 08b_pdf_threshold_analysis.py       # Threshold evaluation
├── data/
│   ├── pdf_classifier_real_data.xgb        # Trained model (gitignored)
│   └── pdf_classifier_metadata.pkl         # Model metadata (gitignored)
└── threshold_analysis/                     # Analysis results (gitignored)
    ├── samples/                            # Sample PDFs by threshold
    ├── plots/                              # Visualizations
    └── statistics.json                     # Dataset stats
```

## Usage

### Train Model
```bash
python spec/phase3/examples/08b_pdf_classifier_training.py --max-pdfs 1000
```

### Analyze Features
```bash
python spec/phase3/examples/08b_pdf_feature_analysis.py --limit 100
```

### Evaluate Thresholds
```bash
python spec/phase3/examples/08b_pdf_threshold_analysis.py
```

## Integration
- Model routes PDFs to OCR or direct text extraction
- Compatible with existing `PDFScannedPredictor` interface
- Ready for Step 4: basic PDF extraction pipeline