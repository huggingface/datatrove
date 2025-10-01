# DataTrove PDF Processing Design Analysis

## Question 1: Why PDF bytes in `text` field?

### The Design Pattern Discovery

**BaseMediaExtractor's Contract** (line 56-89 in media_extractor.py):
```python
def run(self, data: DocumentsPipeline, ...):
    for doc in data:
        texts = []
        for media in doc.media:  # ← Expects PDF in media objects!
            text, metadata = extractor.process_document(media.media_bytes, self.extract)
            texts.append(text)
        doc.text = "".join(texts)  # ← Replaces with extracted text
        yield doc
```

**The INTENDED design**:
- PDF bytes should be in `doc.media[].media_bytes`
- Extractors iterate through `doc.media` array
- Extract text from each media object
- Replace `doc.text` with extracted string

**But our PDFWarcReader puts PDF in `doc.text`**! Why?

### Why We Put PDF Bytes in `text` Field

Looking at test_local_pdfs.py line 64-74:
```python
# Create Document with PDF bytes as text (how media extractors expect it)
doc = Document(
    text=pdf_bytes,  # DoclingExtractor expects PDF bytes in text field
    id=pdf_info['id'],
    metadata={...}
)

# DoclingExtractor.extract() expects (pdf_bytes, metadata) tuple
extracted_text, metadata = extractor.extract((pdf_bytes, doc.metadata))
```

**Two different APIs**:
1. **BaseMediaExtractor.run()** expects: `doc.media[].media_bytes`
2. **DoclingExtractor.extract()** expects: raw `bytes` directly

We're calling `extract()` directly, bypassing `run()`!

### Why We Created pdf_warc.py Separately

Looking at `warc.py` line 89-142:
```python
def process_record(record):
    content_bytes = record.content_stream().read()

    # Decode the response bytes
    html = content_bytes.decode(charset)  # ← Expects HTML, decodes to string

    return {"text": html, ...}  # ← Returns string, not bytes
```

**WarcReader assumptions**:
- Content is HTML/text (decodable)
- `text` field contains strings
- No binary data

**PDFWarcReader needs different behavior**:
- Content is PDF (binary)
- Can't decode to string
- Returns bytes in `text` field

**We created pdf_warc.py because**:
1. WarcReader decodes bytes → string (line 119-131)
2. PDFs are binary, can't be decoded
3. Need to preserve raw bytes for downstream processors

### The Inconsistency

**PDFRouter.run()** (our code):
```python
pdf_bytes = doc.text if isinstance(doc.text, bytes) else doc.text.encode()
prediction = self.predictor.predict(pdf_bytes)
```

**PDFScannedPredictor.predict()** expects:
```python
def predict(self, media_bytes: bytes | None):
    pymupdf_doc = pymupdf.open(stream=io.BytesIO(media_bytes), filetype="pdf")
```

Everyone expects PDF bytes somewhere, but there's no standard location:
- **Pattern A**: PDF bytes in `doc.text` (our tests, PDFWarcReader)
- **Pattern B**: PDF bytes in `doc.media[0].media_bytes` (BaseMediaExtractor design)

---

## Question 2: WARC Writer for Streaming Architecture

### Your Proposed Architecture

```
Stage 1: Classify and Write WARCs Locally
  CommonCrawl WARCs (remote)
    → PDFWarcReader (streaming)
    → PDFRouter (classify + annotate)
    → WarcWriter (write to local disk with routing metadata)

Stage 2: Low OCR Path
  Local WARCs
    → PDFWarcReader (local filesystem)
    → LambdaFilter (processing_route == text_extraction)
    → DoclingExtractor
    → JsonlWriter

Stage 3: High OCR Path
  Local WARCs
    → PDFWarcReader (local filesystem)
    → LambdaFilter (processing_route == ocr_extraction)
    → RolmOCR
    → JsonlWriter
```

### Analysis: Does This Make Sense?

**Pros**:
1. ✅ **Streaming** - No need to hold all PDFs in memory
2. ✅ **Resumable** - Can re-run stages without re-classification
3. ✅ **No serialization issues** - WARCs store binary data natively
4. ✅ **Follows CommonCrawl patterns** - WARCs are the standard format
5. ✅ **Metadata enrichment** - Can add routing info to WARC headers
6. ✅ **PDFWarcReader already exists** - Can read what we write

**Cons**:
1. ❌ **WarcWriter doesn't exist** - Need to implement it
2. ❌ **WARC overhead** - Larger files than JSONL (headers per record)
3. ❌ **Complexity** - WARC format more complex than JSONL

### Does WarcWriter Exist?

**No.** DataTrove has no WARC writer. Only readers:
- `WarcReader` - for HTML/text
- `WarcIndexReprocess` - for reprocessing with offsets
- `PDFWarcReader` - for PDFs

### Alternative: Media Objects with Base64 Encoding

**Actually, looking at the INTENDED design more carefully...**

If we use Media objects properly:
```python
# PDFWarcReader should create:
doc = Document(
    text="",  # Empty, PDF hasn't been processed yet
    id=id_,
    media=[
        Media(
            id=id_,
            type=MediaType.DOCUMENT,
            url=url,
            media_bytes=pdf_bytes  # ← PDF here
        )
    ],
    metadata={...}
)
```

Then:
- JsonlWriter already handles `media[].media_bytes` via base64 encoding (line 49-52)
- No custom serialization needed
- BaseMediaExtractor works out of the box

**But we'd need to refactor**:
1. PDFWarcReader to create Media objects
2. PDFRouter to read from `doc.media[0].media_bytes` instead of `doc.text`
3. All our tests

### The Real Question: What's the Production Pattern?

Looking at how FineWeb/FinePDFs actually works:

**CommonCrawl → Processing → Storage**:
1. Read WARCs from CommonCrawl (streaming)
2. Process in memory (filters, extractors)
3. Write **extracted text** to JSONL (not PDFs)

**They DON'T store intermediate PDFs!**

The full pipeline runs:
```
WARC (PDFs) → Extract Text → Filter → Deduplicate → JSONL (text only)
```

No intermediate PDF storage. Classification happens inline.

### Recommendation: Hybrid Approach

**For Testing** (small scale):
```python
# Simple, no intermediate files
documents = load_pdfs_from_disk()
pipeline = [
    documents,
    PDFRouter(...),
    LambdaFilter(low_ocr),
    DoclingExtractor(),
    JsonlWriter(output)
]
```

**For Production** (WARC streaming):
```python
# Option 1: Inline classification (no intermediate storage)
pipeline = [
    PDFWarcReader(s3_warcs),
    PDFTruncationDetector(),
    PDFRouter(threshold=0.5),
    # Branch A: Low OCR
    LambdaFilter(low_ocr),
    DoclingExtractor(),
    JsonlWriter(text_output),
]

# Run separate job for high OCR:
pipeline = [
    PDFWarcReader(s3_warcs),  # Re-read (streaming, no storage)
    PDFTruncationDetector(),
    PDFRouter(threshold=0.5),
    # Branch B: High OCR
    LambdaFilter(high_ocr),
    RolmOCR(),
    JsonlWriter(ocr_output),
]
```

**Option 2: Store classification results (metadata only)**
```python
# Stage 1: Just classification metadata
PDFWarcReader → PDFRouter → JsonlWriter(
    adapter=lambda doc: {"id": doc.id, "metadata": doc.metadata}
)

# Stage 2/3: Re-read PDFs + merge metadata
# (Complex, requires matching)
```

**Option 3: Implement WarcWriter**
```python
# Stage 1: Classify and write enriched WARCs
PDFWarcReader(remote_warcs) → PDFRouter → WarcWriter(local_warcs)

# Stage 2/3: Read local WARCs
PDFWarcReader(local_warcs) → LambdaFilter → Process
```

### My Recommendation

**Don't store intermediate PDFs at all.**

**Instead**: Run two separate streaming pipelines that re-read the WARCs:

1. **Low OCR Pipeline**: WARC → PDFRouter → Filter(low) → Docling → Output
2. **High OCR Pipeline**: WARC → PDFRouter → Filter(high) → RolmOCR → Output

**Why?**:
- WARCs are already compressed and efficient to stream
- Re-reading is faster than writing+reading intermediate files
- No need to implement WarcWriter
- Follows FineWeb/FinePDFs pattern
- Each pipeline can run independently with different resources

**Trade-off**:
- Classification runs twice (but it's fast compared to extraction)
- Saves disk space and complexity

**If classification is expensive**, then implement WarcWriter. But for XGBoost on PDF features, it's probably negligible compared to Docling/OCR time.

---

## Summary

### Question 1 Answer:
**Why PDF bytes in text field?**
- Because we bypass BaseMediaExtractor.run() and call extract() directly
- WarcReader decodes to string (can't handle binary)
- So we created PDFWarcReader that preserves bytes in text field
- This is a workaround, not the intended design
- **Proper design**: Use Media objects with media_bytes

### Question 2 Answer:
**Should we use WARC writer?**
- **Your instinct is correct** - WARC writer makes sense for intermediate storage
- **But it doesn't exist** and would take time to implement
- **Better approach**: Re-read WARCs in each pipeline (streaming)
- Classification is cheap, avoid premature optimization
- If classification becomes bottleneck, then implement WarcWriter

**Immediate path forward**:
1. Test routing with simple in-memory pipeline (like test_rolmocr.py)
2. For production, run parallel pipelines that stream from same WARCs
3. Only implement WarcWriter if classification becomes a proven bottleneck
