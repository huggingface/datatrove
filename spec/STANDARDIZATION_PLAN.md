# Spec & Implementation Standardization Plan

## Summary of Issues Found

### Inconsistencies Across Phases

**Phase 1:**
- ✅ Good: Detailed docstrings, uses `main()`, has inspection helpers
- ❌ Issues: Uses `print()` instead of logger, hardcoded relative paths

**Phase 2:**
- ❌ Issues: No docstrings, no `main()`, minimal structure, hardcoded `/tmp/` paths

**Phase 3:**
- ❌ Issues: Inconsistent use of `sys.path.insert(0, 'src')`, mixed patterns (some use `main()`, some don't), verbose helper classes inline

**Cross-cutting:**
- No logging library usage (all use `print()`)
- Inconsistent use of `if __name__ == "__main__"`
- Path handling varies (relative, absolute, hardcoded `/tmp/`)
- No consistent comment/docstring format

---

## Proposed Templates

### Spec File Template

```markdown
# Example XX: [Title]

## Objective
[1-2 sentence description of what this example teaches]

## Components
- Component1: Purpose
- Component2: Purpose

## Implementation
**File:** `spec/phaseN/examples/XX_example_name.py`

## Data Requirements
- Input: [description or "None - uses HuggingFace datasets"]
- Output: `spec/phaseN/output/XX_example_name/`

## Expected Results
[Brief description of what success looks like - metrics, file counts, etc]

## Status
- [x] Implemented
- [x] Tested
- [ ] Documentation updated

## Notes
[Any important learnings, gotchas, or context]
```

### Implementation File Template

```python
#!/usr/bin/env python3
"""
Example XX: [Title]

[1-2 sentence description]

Components:
- Component1: Purpose
- Component2: Purpose

Usage:
    python spec/phaseN/examples/XX_example_name.py
"""

from datatrove.utils.logging import logger
from datatrove.utils._import_utils import check_required_dependencies
from datatrove.executor.local import LocalPipelineExecutor
# ... other imports

# Check dependencies if example uses optional libraries directly
# check_required_dependencies("pymupdf", ["fitz"])

# Configuration - paths only (keep other values inline for readability)
OUTPUT_DIR = "spec/phaseN/output/XX_example_name"
LOGS_DIR = "spec/phaseN/logs/XX_example_name"


def main():
    """Main pipeline execution."""
    logger.info("Starting Example XX: [Title]")

    pipeline = [
        # Pipeline steps with inline comments
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir=LOGS_PATH
    )

    executor.run()

    logger.info(f"Pipeline completed. Check: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

---

## Standardization Tasks

### Phase 1: Spec Standardization

**1.1 Update all spec files to follow template**
- [ ] `spec/phase1/01_basic_filtering.md`
- [ ] `spec/phase1/02_extraction_filtering.md`
- [ ] `spec/phase1/03_tokenization.md`
- [ ] `spec/phase1/04_statistics.md`
- [ ] `spec/phase1/05_deduplication.md`

**Workflow per example:**
1. Read the FULL implementation file
2. Verify actual paths used (input, output, logs)
3. Check directory structure exists/matches
4. Update spec file to match template
5. Immediately update implementation file (while context is fresh)
6. Move to next example

**Time estimate:** 6-8 hours (combined specs + implementations)

---

**Tasks per spec file:**
- Simplify to match template (remove verbose content)
- Add "Status" section with checkboxes
- Add "Expected Results" section with actual metrics
- Ensure "Data Requirements" section matches implementation
- Ensure paths are relative to repo root

**Tasks per implementation file:**
- [ ] `spec/phase1/examples/01_basic_filtering.py`
- [ ] `spec/phase1/examples/02_text_extraction.py`
- [ ] `spec/phase1/examples/03_tokenization.py`
- [ ] `spec/phase1/examples/04_statistics.py`
- [ ] `spec/phase1/examples/05_deduplication.py`

**Tasks per file:**
- Add shebang `#!/usr/bin/env python3`
- **Move ALL imports to top of file** (no lazy imports in functions unless absolutely necessary)
- **Import style**: stdlib one per line, from imports compact per module
  ```python
  import json
  import os
  import sys

  from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
  ```
- Simplify docstring to match template
- Replace all `print()` with `logger.info()`, `logger.warning()`, `logger.error()`
- Change import: `from datatrove.utils.logging import logger` (NOT `from loguru import logger`)
- Move output/logs paths to constants at top
- Ensure all use `def main()` + `if __name__ == "__main__"`
- Update helper functions to use logger instead of print
- **Remove verbose inline comments in pipeline** - component names are self-documenting
- Use relative paths from repo root

**Time estimate:** 4-5 hours

---

### Phase 2: Spec Standardization

**2.1 Update all spec files to follow template**
- [ ] `spec/phase2/06_runpod_slurm.md`
- [ ] `spec/phase2/07_lambda_slurm.md`
- [ ] `spec/phase2/07b_lambda_manual_slurm.md`
- [ ] `spec/phase2/07c_datatrove_slurm_execution.md`

**Tasks per file:**
- These are more like guides than specs - keep detailed setup instructions
- Add "Status" section at top
- Add "Expected Results" section
- Clearly separate "Setup Guide" vs "Example Implementation"

**Time estimate:** 2-3 hours

---

**2.2 Update Phase 2 implementations**
- [ ] `spec/phase2/examples/01_basic_filtering_slurm.py`
- [ ] `spec/phase2/examples/04_statistics_slurm.py`

**Tasks per file:**
- Add shebang
- Add docstring following template
- Add loguru import and replace prints (if any)
- Add path constants
- Wrap in `def main()` + `if __name__ == "__main__"`
- Change hardcoded `/tmp/` to relative paths or configurable constants

**Time estimate:** 1-2 hours

---

### Phase 3: Spec Standardization

**3.1 Update all spec files to follow template**
- [ ] `spec/phase3/08_finepdfs_reproduction.md`
- [ ] `spec/phase3/08b_pdf_classifier_model.md`
- [ ] `spec/phase3/08c_rolmocr_integration.md`
- [ ] `spec/phase3/08d_routing_pipeline.md`
- [ ] `spec/phase3/08e_media_objects_refactor.md`

**Tasks per file:**
- Simplify to match template
- 08e is more of a design doc - consider moving to `docs/design/`
- Update all to reference correct implementation files
- Ensure status sections are accurate

**Time estimate:** 3-4 hours

---

**3.2 Update Phase 3 implementations**
- [ ] `spec/phase3/examples/08_finepdfs_local.py`
- [ ] `spec/phase3/examples/08_finepdfs_https.py`
- [ ] `spec/phase3/examples/08_finepdfs_warc.py`
- [ ] `spec/phase3/examples/08b_pdf_classifier_training.py`
- [ ] `spec/phase3/examples/08b_pdf_feature_analysis.py`
- [ ] `spec/phase3/examples/08b_pdf_threshold_analysis.py`
- [ ] `spec/phase3/examples/08c_rolmocr_test.py`
- [ ] `spec/phase3/examples/08d_docling_test.py`
- [ ] `spec/phase3/examples/08d_docling_detailed_test.py`
- [ ] `spec/phase3/examples/08d_ocr_detailed_test.py`
- [ ] `spec/phase3/examples/08d_routing_test.py`

**Tasks per file:**
- Remove `sys.path.insert(0, 'src')` (should run from repo root)
- Add/update docstring to match template
- Replace all `print()` with logger
- Add loguru import
- Ensure all use `def main()` pattern
- Move helper classes to separate utility files if reused
- Consolidate related tests (too many variations)
- Use path constants

**Time estimate:** 6-8 hours

---

### Phase 4: Logging Migration

**4.1 Replace print() with DataTrove logger**

**Standard patterns:**
```python
# Before
print("Starting pipeline...")
print(f"Processing {count} documents")

# After
from datatrove.utils.logging import logger

logger.info("Starting pipeline...")
logger.info(f"Processing {count} documents")
```

**IMPORTANT:** Use `from datatrove.utils.logging import logger`, NOT `from loguru import logger`
- DataTrove's logger is pre-configured with colorization based on `DATATROVE_COLORIZE_LOGS` env var
- Consistent with all source code in `src/datatrove/`

**Log levels to use:**
- `logger.debug()` - Detailed diagnostic info
- `logger.info()` - General progress/status messages (replaces most prints)
- `logger.warning()` - Warning conditions
- `logger.error()` - Error messages
- `logger.success()` - Success messages (loguru-specific)

**Implications:**
- ✅ Consistent with src/ code
- ✅ Pre-configured colorization
- ✅ Better control over verbosity
- ✅ Automatic timestamps and formatting
- ✅ File logging support

**Time estimate:** 3-4 hours (across all files)

---

**4.2 Add dependency checking where needed**

**When to use `check_required_dependencies()`:**
- ✅ If example directly imports optional libraries (`fitz`, `PIL`, `docling`, etc.)
- ❌ If only using DataTrove components (they check internally)

**Pattern:**
```python
from datatrove.utils._import_utils import check_required_dependencies

# At top of main() or before using the library
check_required_dependencies("pymupdf", ["fitz"])
check_required_dependencies("pillow", ["PIL"])

import fitz  # Now safe to import
```

**Files that need dependency checks:**
- Phase 3: Examples using `fitz`, `PIL`, `docling` directly
- Phase 1/2: Likely not needed (use DataTrove components)

**Time estimate:** 1 hour

---

### Phase 5: Path Standardization

**5.1 Define path conventions**

**Principle:** All paths relative to repo root, defined as constants

```python
# Good
BASE_DIR = Path(__file__).parent.parent.parent  # Repo root
OUTPUT_DIR = BASE_DIR / "spec/phase1/output/01_example"
LOGS_DIR = BASE_DIR / "spec/phase1/logs/01_example"

# Or simpler for examples
OUTPUT_DIR = "spec/phase1/output/01_example"
LOGS_DIR = "spec/phase1/logs/01_example"

# Bad
OUTPUT_DIR = "/tmp/output/"  # Hardcoded absolute path
OUTPUT_DIR = "output/"  # Ambiguous relative path
```

**5.2 Update all implementations**
- [ ] Phase 1: Replace hardcoded paths
- [ ] Phase 2: Replace `/tmp/` with configurable paths
- [ ] Phase 3: Standardize all path handling

**Time estimate:** 2-3 hours

---

### Phase 6: Consolidation

**6.1 Reduce redundant examples**

Phase 3 has many similar test files:
- `08d_docling_test.py`
- `08d_docling_detailed_test.py`
- `08d_ocr_detailed_test.py`

**Recommendation:** Consolidate into single files with CLI arguments:
```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    if args.detailed:
        detailed_test()
    else:
        main()
```

**Files to consolidate:**
- [ ] Merge `08d_docling_test.py` + `08d_docling_detailed_test.py`
- [ ] Merge OCR-related tests
- [ ] Merge FinePDFs variants (local/https/warc) into one with args

**Time estimate:** 3-4 hours

---

**6.2 Extract shared utilities**

Create `spec/phase3/examples/utils/` for reusable helpers:
- [ ] `pdf_helpers.py` - PDF loading, Media object creation
- [ ] `test_helpers.py` - Common test utilities
- [ ] `visualization.py` - Plotting/analysis helpers

**Time estimate:** 2-3 hours

---

## Total Time Estimates

| Phase | Time Estimate |
|-------|---------------|
| 1.1 Spec updates | 2-3 hours |
| 1.2 Implementation updates | 4-5 hours |
| 2.1 Spec updates | 2-3 hours |
| 2.2 Implementation updates | 1-2 hours |
| 3.1 Spec updates | 3-4 hours |
| 3.2 Implementation updates | 6-8 hours |
| 4.1 Logging migration | 3-4 hours |
| 4.2 Dependency checking | 1 hour |
| 5.1-5.2 Path standardization | 2-3 hours |
| 6.1-6.2 Consolidation | 5-7 hours |
| **Total** | **29-40 hours** |

---

## Phased Rollout Recommendation

### Week 1: Templates & Phase 1
- Finalize templates
- Complete Phase 1 standardization (specs + implementations)
- This creates the reference for other phases

### Week 2: Logging & Paths
- Migrate all examples to loguru
- Standardize path handling across all phases

### Week 3: Phase 2 & 3
- Update Phase 2 (smaller scope)
- Update Phase 3 specs
- Begin Phase 3 implementations

### Week 4: Consolidation
- Consolidate redundant examples
- Extract shared utilities
- Final review and testing

---

## Success Criteria

- [ ] All specs follow template (concise, consistent structure)
- [ ] All implementations follow template (shebang, docstring, main(), logger)
- [ ] No `print()` statements (all use logger)
- [ ] No `sys.path.insert(0, 'src')` hacks
- [ ] All paths are relative to repo root and configurable
- [ ] Similar examples are consolidated
- [ ] Shared utilities extracted to utils/ folders
- [ ] All examples run from repo root: `python spec/phaseN/examples/XX_name.py`
- [ ] Documentation matches implementation

---

## Open Questions

1. **Logging configuration**: Should examples set up their own logger config or rely on DataTrove defaults?
   - **Recommendation:** Use DataTrove defaults (simpler)

2. **Path handling**: Use `pathlib.Path` or strings?
   - **Recommendation:** Strings for simplicity (DataTrove uses strings)

3. **Test data**: Where should generated/downloaded test data live?
   - **Recommendation:** `spec/phaseN/data/` (already gitignored)

4. **Inspection helpers**: Keep, remove, or consolidate?
   - **Recommendation:** Remove from individual files, create single `spec/utils/inspect.py`

5. **CLI arguments**: Should examples support command-line configuration?
   - **Recommendation:** Only where necessary (Phase 2 Slurm configs, Phase 3 detailed tests)
