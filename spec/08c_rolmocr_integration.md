# RolmOCR Integration Specification

## Overview
Integrate RolmOCR with DataTrove's existing inference infrastructure to enable GPU-based OCR processing for high OCR probability PDFs. This follows the exact approach from the FinePDFs paper.

## Background

From FinePDFs paper:
> "For the GPU-based pipeline, we used RolmOCR, running on top of a modified LMDeploy framework and orchestrated through the Datatrove inference block. All PDFs were rescaled such that the longest dimension is no smaller than 1280px, while ensuring the representation does not exceed 2048 image tokens, before being passed to the model. The total context length of the model, including the input, was set to 8096 tokens."

## Current Infrastructure Analysis

DataTrove already has all required components:
- ✅ `LMDeployServer` class in `src/datatrove/pipeline/inference/servers/lmdeploy_server.py`
- ✅ `InferenceRunner` orchestration in `src/datatrove/pipeline/inference/run_inference.py`
- ✅ PDF page rendering in `src/datatrove/pipeline/inference/utils/page_rendering.py`
- ❌ LMDeployServer not integrated into InferenceRunner

## Implementation Plan

### Step 1: Integrate LMDeployServer into InferenceRunner
**File**: `src/datatrove/pipeline/inference/run_inference.py`

```python
# Update imports
from datatrove.pipeline.inference.servers import (
    DummyServer,
    InferenceServer,
    SGLangServer,
    VLLMServer,
    LMDeployServer,  # Add this
)

# Update Literal type
server_type: Literal["sglang", "vllm", "dummy", "lmdeploy"]  # Add lmdeploy

# Update _init_server method
elif stype == "lmdeploy":
    return LMDeployServer(
        self.config.model_name_or_path,
        self.config.model_max_context,
        self.config.model_kwargs,
    )
```

### Step 2: Update LMDeployServer Import
**File**: `src/datatrove/pipeline/inference/servers/__init__.py`

```python
from datatrove.pipeline.inference.servers.lmdeploy_server import LMDeployServer

__all__ = [
    "InferenceServer",
    "SGLangServer",
    "VLLMServer",
    "DummyServer",
    "LMDeployServer"  # Add this
]
```

### Step 3: Fix LMDeployServer Constructor
**File**: `src/datatrove/pipeline/inference/servers/lmdeploy_server.py`

Current constructor signature doesn't match base class. Update:

```python
def __init__(self, model_name_or_path: str, max_context: int, model_kwargs: Optional[dict] = None):
    super().__init__(model_name_or_path, max_context, model_kwargs)
    self.chat_template = model_kwargs.get('chat_template', 'internlm2-chat') if model_kwargs else 'internlm2-chat'
```

### Step 4: Create RolmOCR Query Builder
**File**: `examples_local/test_rolmocr.py`

```python
import base64
import fitz
from datatrove.pipeline.inference.utils.page_rendering import render_page_to_base64png_pymupdf

def rolmocr_query_builder(runner: InferenceRunner, doc: Document) -> dict:
    """Convert PDF document to RolmOCR vision request."""

    # Open PDF from document text (PDF bytes)
    pdf_bytes = doc.text if isinstance(doc.text, bytes) else doc.text.encode()
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Process all pages
    page_images = []
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)

        # Use FinePDFs preprocessing: longest side ≥ 1280px, max 2048 tokens
        base64_image = render_page_to_base64png_pymupdf(
            page,
            resize_longest_side_pixels=1280,
            max_visual_tokens=2048
        )

        page_images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })

    pdf_doc.close()

    # Create OpenAI-compatible vision request
    return {
        "model": runner.config.model_name_or_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this PDF document using OCR. Return only the extracted text."},
                    *page_images
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0
    }
```

### Step 5: Create RolmOCR Test Script
**File**: `examples_local/test_rolmocr.py`

```python
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceConfig
from datatrove.pipeline.writers.jsonl import JsonlWriter

def test_rolmocr_integration():
    """Test RolmOCR using DataTrove's inference infrastructure."""

    # Load high OCR probability PDFs
    documents = load_high_ocr_pdfs()

    # Configure RolmOCR with LMDeploy
    config = InferenceConfig(
        server_type="lmdeploy",
        model_name_or_path="Reducto/RolmOCR",  # Or actual model path
        model_max_context=8096,  # From FinePDFs paper
        max_concurrent_requests=1,
        max_concurrent_tasks=1,
        model_kwargs={
            "chat_template": "internlm",  # Use "internlm" not "internlm2-chat"
            "vision_max_batch_size": 128  # From LMDeployServer
        }
    )

    # Create inference runner
    runner = InferenceRunner(
        query_builder=rolmocr_query_builder,
        config=config,
        post_process_steps=[JsonlWriter("examples_local/output/rolmocr_results")]
    )

    # Run RolmOCR inference
    runner.run(documents, rank=0, world_size=1)
```

## Testing Strategy

### Phase 1: Integration Testing
1. Update DataTrove infrastructure code
2. Test LMDeployServer integration without actual model
3. Verify query builder creates correct requests

### Phase 2: RolmOCR Model Testing
1. Download/setup Reducto/RolmOCR model
2. Test on high OCR probability PDFs
3. Compare results with DoclingExtractor

### Phase 3: Performance Validation
1. Measure processing time: RolmOCR vs DoclingExtractor
2. Validate text extraction quality improvement
3. Confirm FinePDFs approach benefits

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/.../run_inference.py` | Modify | Add lmdeploy server type and import |
| `src/.../servers/__init__.py` | Modify | Export LMDeployServer |
| `src/.../lmdeploy_server.py` | Fix | Update constructor to match base class |
| `examples_local/test_rolmocr.py` | New | RolmOCR integration test script |

## Implementation Order

1. **Fix LMDeployServer constructor** - Ensure compatibility with base class
2. **Update server exports** - Make LMDeployServer available
3. **Integrate into InferenceRunner** - Add lmdeploy server type
4. **Create query builder** - PDF to vision request conversion
5. **Test integration** - Verify infrastructure works
6. **Test with actual model** - RolmOCR processing validation

## Server Setup Requirements

### CUDA and GPU Setup
```bash
# Install NVIDIA drivers for A100
sudo apt update
sudo apt install -y nvidia-utils-535-server nvidia-driver-535-server

# Reboot to load drivers
sudo reboot

# Verify CUDA availability
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Required Python Packages
```bash
# Install LMDeploy with CUDA support
pip install lmdeploy

# Install Qwen VL utilities (required for RolmOCR)
pip install qwen_vl_utils

# Verify installations
lmdeploy --help
```

## Implementation Results

### Phase 1: Infrastructure Integration ✅ Complete
- ✅ Fixed LMDeployServer constructor compatibility
- ✅ Added LMDeployServer to InferenceRunner server types
- ✅ Updated server exports and imports

### Phase 2: Model Integration ✅ Complete
- ✅ RolmOCR model auto-downloads from HuggingFace Hub
- ✅ PDF preprocessing with FinePDFs specifications (1280px, 2048 visual tokens)
- ✅ OpenAI-compatible vision request format
- ✅ Proper post-processing following DataTrove patterns
- ✅ Chat template: "internlm" (not "internlm2-chat") for clean OCR output

### Phase 3: Testing Results ✅ Complete
- ✅ **True OCR extraction** - correctly extracted Belgian sports form text
- ✅ **High accuracy** - perfect Dutch text recognition, form structure preserved
- ✅ **High resolution processing** - Using FinePDFs spec: 1280px, 2048 visual tokens
- ✅ **Multi-document support** - All 3 PDFs successfully processed and saved
- ✅ **Framework bug discovered and fixed** - PersistentContextJsonlWriter workaround

### Test Results Summary
```
Input: 3 high OCR probability PDFs (capacitor spec, ISO certificate, sports form)
Processing: 3 successful requests, 4,950 prompt tokens, 8,612 completion tokens
Output: All 3 documents saved to JSONL with proper text extraction
Quality: Clean markdown tables, proper formatting, excellent OCR accuracy
```

### Framework Bug Discovery
**Issue**: InferenceRunner calls `post_process_steps.run()` separately for each document, causing JsonlWriter to enter/exit context per document. The `with self:` in `run()` closes files between documents, and reopening in `"wb"` mode truncates the file.

**Solution**: Created `PersistentContextJsonlWriter` that:
- Enters context once on first `run()` call
- Keeps file handles open across all documents
- Explicitly closes in `finally` block to ensure gzip finalization

**Verification**: Tested with `test_inference_simple.py` - confirmed vanilla example also has this bug (only saves last of 3 documents).

## Implementation Complete

✅ **Infrastructure** - LMDeployServer integrated into InferenceRunner
✅ **Query Builder** - PDF to vision request with FinePDFs preprocessing
✅ **Post-Processing** - Proper text extraction following DataTrove patterns
✅ **Bug Workaround** - PersistentContextJsonlWriter for multi-document support
✅ **Testing** - Validated with 3 high OCR PDFs, sample outputs saved

## Next Steps

1. **Implement two-tiered routing** - XGBoost classifier → Docling (low OCR) / RolmOCR (high OCR)
2. **Test chunking and checkpointing** - Scale to 1000s of PDFs with resumption support
3. **Production pipeline** - PDFWarcReader streaming from CommonCrawl with distributed processing