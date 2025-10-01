# Refactor to Use Media Objects (Spec 08e)

## Overview

Refactor PDF handling to use the framework's intended Media object design instead of storing PDF bytes in the `text` field. This aligns with DataTrove's design patterns and fixes JSONL serialization issues.

## Problem Statement

**Current approach** (our workaround):
- PDF bytes stored in `doc.text` field (violates type hint `text: str`)
- Can't serialize to JSONL (orjson can't handle bytes)
- Bypasses framework's Media handling
- Inconsistent with DataTrove's design

**Intended approach** (framework design):
- PDF bytes stored in `doc.media[].media_bytes`
- JsonlWriter automatically base64 encodes Media objects
- Type-safe and serializable
- Works with all BaseMediaExtractor subclasses

## Background: Why We Got Here

### The Two Patterns in DataTrove

**Pattern 1: WarcReaderFast + BaseMediaExtractor** (CORRECT - Framework Design)
```python
# WarcReaderFast populates Media objects
doc.media = [Media(media_bytes=pdf_bytes, ...)]

# BaseMediaExtractor.run() iterates over media
for media in doc.media:
    text, metadata = extractor.process_document(media.media_bytes, self.extract)
doc.text = "".join(texts)  # Replace bytes with extracted text
```

**Pattern 2: PDFWarcReader + direct extract() calls** (WORKAROUND - Our Approach)
```python
# PDFWarcReader puts bytes in text field
doc.text = pdf_bytes  # Violates type hint

# We call extract() directly, bypassing run()
text, metadata = extractor.extract(pdf_bytes)
```

### Why We Created the Wrong Pattern

1. **WarcReader base class** decodes bytes → string (can't handle binary PDFs)
2. **We didn't know WarcReaderFast existed** in `media/readers/warc_threaded.py`
3. **Calling `.extract()` directly** seemed simpler than understanding Media objects
4. **Tests worked** so we didn't realize it was wrong until JSONL serialization

### Why WarcReader vs PDFWarcReader

**WarcReader** (`src/datatrove/pipeline/readers/warc.py`):
```python
content_bytes = record.content_stream().read()
html = content_bytes.decode(charset)  # ← Assumes text/HTML
return {"text": html, ...}  # Returns string
```
- Designed for HTML/text content
- Decodes bytes to string
- Can't handle binary PDFs

**PDFWarcReader** (our code):
```python
content_bytes = record.content_stream().read()
return {"text": content_bytes, ...}  # ← Returns bytes
```
- Preserves binary PDF data
- But puts it in wrong field (`text` instead of `media`)
- Created because we didn't know about WarcReaderFast

**WarcReaderFast** (framework's solution):
```python
record.media.append(Media(
    media_bytes=content_bytes,  # ← Correct location
    ...
))
```
- Handles binary data properly
- Uses Media objects as intended
- Already exists in framework!

## Key Discovery

**WarcReaderFast already implements this correctly!** We reinvented the wheel with PDFWarcReader because we didn't know it existed.

## Refactoring Tasks

### Task 1: Update Test Files to Create Media Objects

**Files to update**:
- `examples_local/test_local_pdfs.py`
- `examples_local/test_rolmocr.py`
- `examples_local/test_routing.py`

**Change pattern**:

```python
# BEFORE (current)
doc = Document(
    text=pdf_bytes,  # Wrong: bytes in text field
    id=pdf_info['id'],
    metadata={...}
)

# AFTER (correct)
from datatrove.data import Media, MediaType

doc = Document(
    text="",  # Empty until extracted
    id=pdf_info['id'],
    media=[
        Media(
            id=pdf_info['id'],
            type=MediaType.DOCUMENT,
            media_bytes=pdf_bytes,  # Correct: bytes in Media object
            url=f"file://{pdf_path}",
        )
    ],
    metadata={...}
)
```

**Testing Note**: When running DoclingExtractor tests, set the environment variable:
```bash
export LAYOUT_VINO_PATH="../Docling-sync/models/v2-quant.xml"
python examples_local/test_local_pdfs.py
```

**Status**: ✅ COMPLETED
- test_local_pdfs.py: Both functions refactored and tested successfully
- DoclingExtractor signature fixed: `def extract(self, media_bytes: bytes | None)`
- Tested on Lambda: Low OCR (9,187 chars) and High OCR (2,173 chars) both working

**Estimated time**: 1 hour

---

### Task 2: Update PDFRouter to Read from Media Objects

**File**: `src/datatrove/pipeline/filters/pdf_router.py`

**Current code** (lines 67-69):
```python
pdf_bytes = doc.text if isinstance(doc.text, bytes) else doc.text.encode()
prediction = self.predictor.predict(pdf_bytes)
```

**Updated code**:
```python
# Get PDF bytes from Media object
if not doc.media:
    self.stat_update("no_media")
    doc.metadata["prediction_error"] = "No media objects found"
    continue

if not doc.media[0].media_bytes:
    self.stat_update("no_media_bytes")
    doc.metadata["prediction_error"] = "Media object has no bytes"
    continue

pdf_bytes = doc.media[0].media_bytes
prediction = self.predictor.predict(pdf_bytes)
```

**Estimated time**: 30 minutes

---

### Task 3: Delete PDFWarcReader (Use WarcReaderFast Instead)

**Files to delete**:
- `src/datatrove/pipeline/readers/pdf_warc.py`
- `tests/pipeline/test_pdf_warc_reader.py` (if exists)

**Files to update** (import changes):
```python
# BEFORE
from datatrove.pipeline.readers.pdf_warc import PDFWarcReader

# AFTER
from datatrove.pipeline.media.readers.warc_threaded import WarcReaderFast
```

**Update usage**:
```python
# BEFORE
PDFWarcReader(
    data_folder=warc_dir,
    pdf_mime_types=["application/pdf"]
)

# AFTER
# WarcReaderFast works with documents that already have warc_filename and warc_record_offset
# It fetches the content and populates doc.media
WarcReaderFast(
    data_folder="s3://commoncrawl",  # or local path
    workers=4
)
```

**Note**: WarcReaderFast expects documents with `warc_filename` and `warc_record_offset` metadata. For production WARC streaming, use the pattern from `pdf_docling_test.py`:

```python
pipeline = [
    JsonlReader(metadata_file),  # Contains WARC offsets
    WarcReaderFast(data_folder),  # Fetches PDFs into doc.media
    PDFRouter(...),
    # ... rest of pipeline
]
```

**Estimated time**: 15 minutes

---

### Task 4: Update Pipeline Usage Patterns

**Current pattern** (calling extract() directly):
```python
doc = Document(text=pdf_bytes, ...)
extracted_text, metadata = extractor.extract(pdf_bytes)
```

**Updated pattern** (using pipeline):
```python
docs = [doc1, doc2, doc3]  # Documents with Media objects

pipeline = [
    docs,
    DoclingExtractor(),  # Automatically processes doc.media
    JsonlWriter(output_dir)
]

executor = LocalPipelineExecutor(pipeline=pipeline, tasks=1)
executor.run()
```

**Reference implementation**: `examples_local/pdf_docling_test.py` lines 31-44

**Estimated time**: 30 minutes

---

### Task 5: Verify JSONL Serialization Round-Trip

**Test that**:
1. Media objects with PDF bytes can be written to JSONL
2. JsonlWriter base64 encodes `media_bytes`
3. JsonlReader can read them back
4. Base64 decoding works automatically

**Test script**:
```python
# Write documents with Media objects
docs = [Document(
    text="",
    id="test",
    media=[Media(id="test", type=MediaType.DOCUMENT, media_bytes=pdf_bytes)]
)]

JsonlWriter(output_dir, save_media_bytes=True).write(docs[0], rank=0)

# Read back
reader = JsonlReader(output_dir)
for doc in reader.read_file("00000.jsonl.gz"):
    assert len(doc.media) == 1
    assert doc.media[0].media_bytes == pdf_bytes  # Check bytes match
```

**Estimated time**: 30 minutes

---

## Implementation Order

1. ✅ **Task 1**: Update test files (safest, easiest to verify)
2. ✅ **Task 2**: Update PDFRouter (small, critical)
3. ✅ **Task 5**: Verify serialization works (validation)
4. ✅ **Task 4**: Update pipeline patterns (integration)
5. ✅ **Task 3**: Delete PDFWarcReader (cleanup)

## Testing Strategy

### Unit Tests
- Test PDFRouter with Media objects
- Test document creation with Media objects
- Test JSONL write/read round-trip

### Integration Tests
- Run `test_routing.py` end-to-end
- Verify routing statistics are correct
- Check output JSONL files are valid

### Regression Tests
- Run existing Docling tests
- Run existing RolmOCR tests
- Ensure no functionality broken

## Success Criteria

- ✅ All tests pass with Media objects
- ✅ PDFRouter correctly reads from `doc.media[0].media_bytes`
- ✅ JSONL serialization works (no orjson errors)
- ✅ Round-trip write→read preserves PDF bytes
- ✅ Routing pipeline runs end-to-end
- ✅ Code follows DataTrove patterns
- ✅ No PDF bytes in `doc.text` field

## Rollback Plan

If refactoring causes issues:
1. Revert to previous branch
2. Keep PDFWarcReader for compatibility
3. Maintain both patterns (Media for production, text for tests)

**Risk**: LOW - All changes in our code, no external dependencies

## Files Modified Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `examples_local/test_local_pdfs.py` | Modify | Create Media objects instead of text field |
| `examples_local/test_rolmocr.py` | Modify | Create Media objects instead of text field |
| `examples_local/test_routing.py` | Modify | Create Media objects instead of text field |
| `src/datatrove/pipeline/filters/pdf_router.py` | Modify | Read from `doc.media[0].media_bytes` |
| `src/datatrove/pipeline/readers/pdf_warc.py` | Delete | Use WarcReaderFast instead |
| `tests/pipeline/test_pdf_warc_reader.py` | Delete | No longer needed |
| `spec/08d_routing_pipeline.md` | Update | Reference WarcReaderFast instead of PDFWarcReader |

## Estimated Total Time

| Phase | Time |
|-------|------|
| Task 1: Update test files | 1 hour |
| Task 2: Update PDFRouter | 30 min |
| Task 3: Delete PDFWarcReader | 15 min |
| Task 4: Update pipeline patterns | 30 min |
| Task 5: Verify serialization | 30 min |
| Testing and validation | 30 min |
| **Total** | **3.5 hours** |

## Benefits

### Immediate
- ✅ Fixes JSONL serialization issue (blocker for routing pipeline)
- ✅ Type-safe code (`text: str` is correct)
- ✅ Framework-compliant (using intended design)

### Long-term
- ✅ Maintainable (follows DataTrove patterns)
- ✅ Reusable (works with all Media extractors)
- ✅ Scalable (multi-media support built-in)
- ✅ No technical debt

## Next Steps After Refactoring

Once refactoring is complete:
1. Test simple routing pipeline (in-memory)
2. Implement production WARC streaming pattern
3. Run end-to-end test with sample WARCs
4. Document the correct usage patterns

## References

- **WarcReaderFast**: `src/datatrove/pipeline/media/readers/warc_threaded.py`
- **BaseMediaExtractor**: `src/datatrove/pipeline/media/extractors/media_extractor.py`
- **Example usage**: `examples_local/pdf_docling_test.py`
- **Design analysis**: `spec/DESIGN_ANALYSIS.md`
- **Cost analysis**: `spec/REFACTORING_COST_ANALYSIS.md`
