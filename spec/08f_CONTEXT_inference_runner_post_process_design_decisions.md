# InferenceRunner Post-Process Steps Design Analysis

## The Central Question

**Why does InferenceRunner have `post_process_steps` instead of just using regular pipeline steps after it?**

```python
# Current approach: post_process_steps
pipeline = [
    documents,
    InferenceRunner(
        query_builder=...,
        config=...,
        post_process_steps=[
            PostProcessOCRResults(),
            JsonlWriter(output)
        ]
    )
]

# Intuitive alternative: regular pipeline
pipeline = [
    documents,
    InferenceRunner(query_builder=..., config=...),
    PostProcessOCRResults(),
    JsonlWriter(output)
]
```

Why doesn't the second approach work?

---

## The Core Answer

**InferenceRunner is a terminal node - it doesn't yield documents forward.**

### Critical Insight: The Type Signature

From `src/datatrove/pipeline/inference/run_inference.py:667`:

```python
def run(
    self,
    data: Iterable[Document],
    rank: int = 0,
    world_size: int = 1,
) -> None:  # ← Returns None, not Generator[Document]!
    """
    Consume `data`, run inference and post-processing, do not yield further documents.
    """
    with self.track_time():
        asyncio.run(self.run_async(data, rank, world_size))
```

**Normal PipelineSteps:**
```python
def run(self, data: Iterable[Document], ...) -> Generator[Document]:
    for doc in data:
        # process doc
        yield doc  # ← Passes to next step
```

**InferenceRunner:**
```python
def run(self, data: Iterable[Document], ...) -> None:
    # processes all documents internally
    # NEVER yields anything
```

**This fundamentally breaks the pipeline chain.**

---

## Why InferenceRunner is Designed This Way

### 1. Async Processing Requirements

InferenceRunner is inherently asynchronous:

```python
async def run_async(self, data, rank, world_size):
    """Process documents with concurrent API requests."""

    # Start multiple documents in parallel
    tasks = []
    for doc in data:
        task = asyncio.create_task(self._process_document(doc))
        tasks.append(task)

    # Documents complete out-of-order
    for completed in asyncio.as_completed(tasks):
        result = await completed
        await self._save_document(result, rank, world_size)
```

**Key characteristics:**
- **Concurrent execution**: Process N documents simultaneously
- **Out-of-order completion**: Document 3 may finish before document 1
- **Async/await pattern**: Uses asyncio event loop
- **Queue management**: Tracks pending/running/completed requests

### 2. Regular Pipeline Assumptions

The standard DataTrove pipeline pattern assumes:

```python
# Sequential, synchronous processing
for step in pipeline:
    docs = step.run(docs, rank, world_size)  # Expects generator
    for doc in docs:  # Process one at a time
        # next step starts only after previous doc finishes
```

**Assumptions:**
- ✅ Synchronous processing
- ✅ In-order document flow
- ✅ Document N+1 starts after document N completes
- ✅ Each step yields to the next

**These are incompatible with async inference!**

---

## The Callback Pattern: `_save_document()`

From `run_inference.py:428-482`:

```python
async def _save_document(self, document: Document, rank: int, world_size: int, ...):
    """
    Save processed document through post-processing pipeline.
    Called as a callback when each document finishes inference.
    """
    # Track metrics
    try:
        inference_results = document.metadata.get("inference_results", [])
        # ... metrics tracking ...
        self.stat_update("successful_documents", value=1)
    except Exception as e:
        logger.warning(f"Failed to process inference results for metrics: {e}")
        self.stat_update("failed_documents", value=1)

    # Run through post-processing pipeline
    tmp_gen = (d for d in [document])  # ← Single-document generator!
    for step in self.post_process_steps:
        tmp_gen = step.run(tmp_gen, rank, world_size=world_size)

    # Exhaust the generator to trigger all post-processing steps
    deque(tmp_gen, maxlen=0)  # ← Force execution, discard results
```

**What's happening:**
1. Each document completes inference independently (async)
2. Calls `_save_document()` as a callback
3. Creates a **mini-pipeline for JUST THAT ONE DOCUMENT**
4. Runs post-process steps on that single document
5. Exhausts the generator (forces execution, discards results)

**Key insight:** `post_process_steps` is called **N times** (once per document), not once for all documents.

---

## Why Not Just Yield Documents?

### Hypothetical: If InferenceRunner Yielded

```python
pipeline = [
    documents,  # 100 documents
    InferenceRunner(...),  # What if this yielded?
    PostProcessOCRResults(),
    JsonlWriter(output)
]
```

**Timeline if InferenceRunner yielded:**
```
Time 0: InferenceRunner starts doc 1, 2, 3 inference (async)
Time 1: Doc 3 finishes first
        ↓ yield doc 3
        ↓ PostProcessOCRResults processes doc 3
        ↓ JsonlWriter writes doc 3
        ↓ Returns control to InferenceRunner
Time 2: Doc 1 finishes
        ↓ yield doc 1
        ↓ PostProcessOCRResults processes doc 1
        ... etc
```

**Problems:**
1. ❌ **Blocking**: Post-process steps block InferenceRunner
2. ❌ **Sequential post-processing**: Can't write doc 1 while processing doc 2
3. ❌ **Reduced parallelism**: Limited by slowest post-process step
4. ❌ **Ordering complexity**: Must buffer completed docs to yield in order

### With `post_process_steps` (Current Design)

```python
InferenceRunner(
    post_process_steps=[PostProcessOCRResults(), JsonlWriter()]
)
```

**Timeline:**
```
Time 0: Start inference for docs 1, 2, 3, 4, 5 (all concurrent)
Time 1: Doc 3 finishes
        → post-process doc 3 (in callback)
        → write doc 3 to JSONL
        (does NOT block other inferences)

Time 2: Doc 1 finishes
        → post-process doc 1 (in callback)
        → write doc 1 to JSONL

Time 3: Doc 5 finishes
        → post-process doc 5
        → write doc 5

... etc (out-of-order, fully async)
```

**Benefits:**
- ✅ **Full async**: All inferences run concurrently
- ✅ **No blocking**: Post-processing doesn't slow down inference
- ✅ **Maximum throughput**: Limited only by API/GPU speed
- ✅ **Out-of-order OK**: Don't need to wait for doc 1 before saving doc 3

---

## The Bug We Hit: PersistentContextJsonlWriter

### Why Regular JsonlWriter Failed

Each document calls `_save_document()` independently:

```python
# Document 1 finishes
tmp_gen = (d for d in [doc1])  # New generator
for step in post_process_steps:
    tmp_gen = step.run(tmp_gen, ...)  # JsonlWriter.run() called

# Document 2 finishes
tmp_gen = (d for d in [doc2])  # New generator again!
for step in post_process_steps:
    tmp_gen = step.run(tmp_gen, ...)  # JsonlWriter.run() called AGAIN
```

**Regular JsonlWriter behavior:**
```python
def run(self, data, rank, world_size):
    with self:  # __enter__() opens file
        for doc in data:
            self.write(doc, rank)
    # __exit__() closes file immediately
```

**When called 3 times (3 documents completing async):**
```
Call 1 (doc 3): open file → write doc 3 → close file
Call 2 (doc 1): open file → write doc 1 → close file (OVERWRITES!)
Call 3 (doc 2): open file → write doc 2 → close file (OVERWRITES!)

Result: Only doc 2 remains in file
```

### The Fix: PersistentContextJsonlWriter

From `test_finepdfs_local.py:85-115`:

```python
class PersistentContextJsonlWriter(JsonlWriter):
    """JsonlWriter that keeps file context open across multiple run() calls.

    Workaround for framework bug where InferenceRunner calls post_process_steps
    separately for each document, causing JsonlWriter to close/reopen files
    between documents, which truncates the output file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context_entered = False

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        # Enter context only ONCE, on first call
        if not self._context_entered:
            self.__enter__()  # Open file
            self._context_entered = True

        # Write documents WITHOUT entering/exiting context
        for document in data:
            with self.track_time():
                self.write(document, rank)
            yield document

    def __del__(self):
        # Clean up context when object is destroyed
        if self._context_entered:
            try:
                self.__exit__(None, None, None)
            except:
                pass
```

**Explicit cleanup required:**
```python
try:
    stage3_ocr_extraction.run()
finally:
    # Find the writer and explicitly close it
    for step in stage3.pipeline:
        if isinstance(step, InferenceRunner):
            for post_step in step.post_process_steps:
                if isinstance(post_step, PersistentContextJsonlWriter):
                    if post_step._context_entered:
                        post_step.__exit__(None, None, None)
```

**Why `__del__` isn't enough:**
- Python's garbage collection is non-deterministic
- `__del__` may not be called until program exit
- Gzip files need explicit finalization
- Without `__exit__`, file is corrupted/truncated

---

## Design Trade-offs

### Current Design (Callback Pattern)

**Pros:**
- ✅ Maximum async throughput
- ✅ Concurrent processing of all documents
- ✅ No blocking between inference and post-processing
- ✅ Out-of-order completion is fine

**Cons:**
- ❌ Breaks standard pipeline pattern
- ❌ Confusing API (why are some steps inside, some outside?)
- ❌ Requires special handling for stateful steps (like writers)
- ❌ Post-process steps called N times, not once
- ❌ No way to continue pipeline after InferenceRunner

### Alternative Design: Yielding with Queue

```python
# Hypothetical better design
async def run_async(self, data, rank, world_size):
    """Process with internal queue, yield documents as they complete."""

    async for doc in self._process_with_queue(data):
        yield doc  # Yield as soon as ANY document finishes
```

**Would allow:**
```python
pipeline = [
    documents,
    InferenceRunner(...),  # Yields documents async
    PostProcessOCRResults(),  # Processes as they arrive
    JsonlWriter(output)  # Writes as they arrive
]
```

**Benefits:**
- ✅ Standard pipeline pattern
- ✅ No special writer handling needed
- ✅ Can continue pipeline after inference
- ✅ Still async (yield immediately when any doc completes)

**Challenges:**
- Complex to implement (async generators in sync pipeline)
- May require async-aware pipeline executor
- Order preservation if needed becomes tricky

---

## When to Use Post-Process Steps

### Use Cases That Work Well

**1. Document-level transformations:**
```python
post_process_steps=[
    PostProcessOCRResults(),  # Extract text from inference_results
    LambdaFilter(lambda doc: len(doc.text) > 100),  # Filter by length
]
```

Each document processed independently, no cross-document state.

**2. Writing output (with care):**
```python
post_process_steps=[
    PersistentContextJsonlWriter(output)  # Must use persistent version!
]
```

Requires special handling for file writers.

### Use Cases That Don't Work

**1. Deduplication:**
```python
post_process_steps=[
    ExactDedupFilter()  # ❌ BROKEN: Needs to see ALL documents
]
```

Can't deduplicate when processing one document at a time.

**2. Statistics across all documents:**
```python
post_process_steps=[
    ComputeGlobalStats()  # ❌ BROKEN: Needs full dataset
]
```

Each invocation only sees one document.

**3. Sorting/ordering:**
```python
post_process_steps=[
    SortByScore()  # ❌ BROKEN: Documents arrive out-of-order
]
```

Can't sort when documents complete asynchronously.

---

## Recommendations

### For DataTrove Framework Developers

**Consider refactoring InferenceRunner to yield:**
```python
def run(self, data, rank, world_size) -> Generator[Document]:
    """Yield documents as they complete inference."""
    # Use async generator internally
    for doc in asyncio.run(self._async_generator(data, rank, world_size)):
        yield doc
```

This would:
- Restore standard pipeline pattern
- Remove need for PersistentContextJsonlWriter
- Allow post-inference pipeline steps
- Maintain async performance

### For Users of InferenceRunner

**Current workarounds:**

1. **Always use PersistentContextJsonlWriter in post_process_steps:**
   ```python
   post_process_steps=[
       PostProcessOCRResults(),
       PersistentContextJsonlWriter(output)  # Not JsonlWriter!
   ]
   ```

2. **Always add explicit cleanup:**
   ```python
   try:
       executor.run()
   finally:
       # Find and close the writer
       for step in executor.pipeline:
           if isinstance(step, InferenceRunner):
               for post_step in step.post_process_steps:
                   if isinstance(post_step, PersistentContextJsonlWriter):
                       post_step.__exit__(None, None, None)
   ```

3. **Only use stateless post-processing:**
   - Document-level transformations: ✅
   - Filters based on doc content: ✅
   - Writers: ⚠️ Use PersistentContextJsonlWriter
   - Cross-document operations: ❌ Don't use

---

## Memory and Performance Implications

### Does PersistentContextJsonlWriter Cause Memory Issues?

**Short answer: No. It changes nothing about memory usage.**

#### The Write Path

From `disk_base.py:185` and `jsonl.py:52`:

```python
def write(self, document: Document, rank: int = 0, **kwargs):
    # ... filename logic ...
    self._write(self.adapter(document), self.output_mg.get_file(output_filename), original_name)

def _write(self, document: dict, file_handler: IO, _filename: str):
    file_handler.write(orjson.dumps(document, option=orjson.OPT_APPEND_NEWLINE))
```

**Key insight:** `file_handler.write()` writes to disk **immediately**. Documents are NOT buffered in memory.

#### Memory Comparison

**Regular JsonlWriter:**
```python
def run(self, data, rank, world_size):
    with self:  # Open file
        for doc in data:
            self.write(doc, rank)  # Write immediately → disk
    # Close file
```

**Memory held:**
- File handle: ~32 KB (gzip buffer)
- Current document: Transient (released after write)
- **NOT** all documents

**PersistentContextJsonlWriter:**
```python
def run(self, data, rank, world_size):
    if not self._context_entered:
        self.__enter__()  # Open file once

    for doc in data:
        self.write(doc, rank)  # Write immediately → disk
    # Don't close yet
```

**Memory held:**
- File handle: ~32 KB (gzip buffer)
- Current document: Transient (released after write)
- **NOT** all documents

**Difference: ZERO**

Both write immediately to disk. Neither buffers documents.

#### File Handle Memory Overhead

```python
# Gzip file handle size
>>> import gzip, sys
>>> f = gzip.open('test.gz', 'wt')
>>> sys.getsizeof(f)
208 bytes  # Object itself

>>> sys.getsizeof(f._compressor)
~8-32 KB  # Internal compression buffer (constant size)
```

**File handle overhead is:**
1. **Tiny** (~32 KB for gzip)
2. **Constant size** (doesn't grow with more documents)
3. **Negligible** compared to document data

#### Where Memory Actually Accumulates

**The real memory usage is documents in flight:**

```python
# InferenceRunner processing N documents concurrently
async def run_async(self, data, rank, world_size):
    # max_concurrent_requests documents held in memory here
    for doc in data:
        task = self._process_document(doc)  # ← Doc in memory during inference

    # After inference completes
    await self._save_document(result, ...)  # ← Writes immediately, releases memory
```

**Memory profile:**
- **During inference**: N documents in memory (N = max_concurrent_requests)
- **After write**: Document released, memory freed
- **File handle**: Constant 32 KB overhead

**Per-document memory:**
- PDF bytes: 100 KB - 10 MB
- Rendered PNG images: 100 KB - 1 MB
- Inference results: 1 KB - 100 KB

**File handle (32 KB) is completely negligible compared to documents (100 KB - 10 MB each).**

#### At Scale: 1 Million Documents

**Regular JsonlWriter (if it worked):**
```
Open file → close file (repeated 1M times)
File handle peak memory: 32 KB
Document peak memory: N × (100KB - 10MB) where N = concurrent batch size
```

**PersistentContextJsonlWriter:**
```
Open file (once) → keep open → close file (once)
File handle peak memory: 32 KB
Document peak memory: N × (100KB - 10MB) where N = concurrent batch size
```

**Difference:** None. File handle overhead is constant.

**The REAL memory constraint:**
```python
InferenceRunner(
    config=InferenceConfig(
        max_concurrent_requests=10,  # ← This controls memory usage!
    )
)
```

With `max_concurrent_requests=10`:
- 10 documents in memory at peak
- File handle: 32 KB (irrelevant)
- Total: ~1 MB - 100 MB depending on document sizes

#### Traditional Sequential Pattern Comparison

**If InferenceRunner yielded (hypothetical):**
```python
with JsonlWriter() as writer:
    for doc in documents:  # 1000 documents processed sequentially
        writer.write(doc)
# File closes here

# Duration file open: ~minutes (entire batch)
# Memory: 1 document at a time + 32 KB file handle
```

**PersistentContext (actual):**
```python
writer = PersistentContextJsonlWriter()
writer.__enter__()

# Process 1000 documents (async, concurrent)
for doc in documents:
    writer.write(doc)  # Writes happen independently

writer.__exit__()

# Duration file open: ~minutes (same duration!)
# Memory: N concurrent documents + 32 KB file handle
```

**Key insight:** File is open for the same duration in both patterns. The only difference is:
- **Traditional**: 1 document in memory, sequential processing
- **Async**: N documents in memory, concurrent processing
- **File handle**: Same 32 KB in both cases

#### Summary: Memory Is NOT a Concern

**PersistentContextJsonlWriter does NOT cause memory issues because:**

1. ✅ Documents written to disk immediately (not buffered)
2. ✅ File handle is tiny (~32 KB) and constant size
3. ✅ Gzip buffer is tiny (~32 KB) and constant size
4. ✅ Memory dominated by documents in flight during inference
5. ✅ File being open longer doesn't accumulate memory

**Memory is controlled by:**
```python
max_concurrent_requests = N  # N documents in memory during inference
```

**NOT by the file handle.**

The persistent context is purely about **preventing file truncation in async callbacks**, not about memory management.

---

## Summary

**Why `post_process_steps` exists:**

InferenceRunner is async and doesn't yield documents. Instead, it uses a callback pattern where `post_process_steps` runs on each document independently as it completes (potentially out-of-order).

**Could they be regular pipeline steps?**

Technically yes, if InferenceRunner yielded documents. But that would require significant refactoring to maintain async parallelism.

**The design tension:**

- **Async performance** (current design) vs **API simplicity** (standard pipeline)
- DataTrove chose performance, at the cost of confusing API
- Requires special handling for stateful steps like writers

**Our workaround:**

PersistentContextJsonlWriter keeps file handle open across multiple `run()` calls, preventing file truncation when processing multiple documents through async inference.

**Memory impact:**

None. File handles are tiny (~32 KB) and constant size. Memory is dominated by documents in flight during inference, controlled by `max_concurrent_requests`.

---

## References

- **InferenceRunner source**: `src/datatrove/pipeline/inference/run_inference.py`
  - Line 667: `def run(...) -> None` (doesn't yield)
  - Line 428-482: `_save_document()` callback pattern
  - Line 477-482: How post_process_steps are invoked per-document

- **PersistentContextJsonlWriter**:
  - `examples_local/test_finepdfs_local.py:85-115`
  - `examples_local/test_rolmocr.py:42-73`
  - `examples/finepdfs.py:85-115`

- **Explicit cleanup pattern**:
  - `examples_local/test_finepdfs_local.py:340-353`
  - `examples/finepdfs.py:321-336`

- **Original bug discovery**: Commit 09d302c - "Add PersistentContextJsonlWriter to fix multi-document output bug"
- **Cleanup fix**: Commit 1ec7ebc - "Add explicit writer cleanup to ensure gzip file is closed"
