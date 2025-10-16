# Spec & Implementation Guidelines

This document defines standards for all spec files, implementation files, and documentation in the DataTrove repository.

## Spec File Template

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

## Implementation File Template

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

# Standard library imports (one per line)
import json
import os

# Third-party imports
import numpy as np

# DataTrove imports
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
from datatrove.utils.logging import logger

# Configuration - paths only
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
        logging_dir=LOGS_DIR
    )

    executor.run()

    logger.info(f"Pipeline completed. Check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

## Import Organization

**Order:**
1. Standard library imports (one per line)
2. Third-party imports
3. DataTrove imports

**Separate each group with a blank line.**

**Style:**
```python
# Good
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
from datatrove.utils.logging import logger

# Bad
import json, os  # Don't combine on one line
import numpy as np
from datatrove.utils.logging import logger  # Missing blank lines between groups
```

## Logging

**Always use DataTrove logger:**
```python
from datatrove.utils.logging import logger

# NOT from loguru import logger
```

**Log levels:**
- `logger.debug()` - Detailed diagnostic info
- `logger.info()` - General progress/status (replaces `print()`)
- `logger.warning()` - Warning conditions
- `logger.error()` - Error messages

**Never use:**
- `print()` statements
- Decorative separators: `logger.info("=" * 80)`

**Good:**
```python
logger.info("Starting pipeline...")
logger.info(f"Processed {count} documents")
```

**Bad:**
```python
print("Starting pipeline...")
logger.info("=" * 80)
logger.info("STARTING PIPELINE")
logger.info("=" * 80)
```

## Path Handling

**Use constants at top of file:**
```python
OUTPUT_DIR = "spec/phase1/output/01_example"
LOGS_DIR = "spec/phase1/logs/01_example"
```

**Use string concatenation for sub-paths:**
```python
# Good
JsonlWriter(OUTPUT_DIR + "/classified")
logging_dir=LOGS_DIR + "/classification"

# Bad
JsonlWriter(f"{OUTPUT_DIR}/classified")  # Don't use f-strings
JsonlWriter(OUTPUT_DIR / "classified")   # Don't use Path division
```

**All paths relative to repo root:**
```python
# Good
OUTPUT_DIR = "spec/phase1/output/01_example"

# Bad
OUTPUT_DIR = "/tmp/output/"  # Hardcoded absolute
OUTPUT_DIR = "output/"       # Ambiguous relative
```

## File Structure

**Required elements:**
1. Shebang: `#!/usr/bin/env python3`
2. Module docstring with Example XX format
3. Imports (organized by category)
4. Configuration constants (OUTPUT_DIR, LOGS_DIR)
5. Helper functions (if needed)
6. `main()` function
7. `if __name__ == "__main__": main()`

**Forbidden:**
- `sys.path.insert(0, 'src')` - examples must run from repo root
- Lazy imports inside functions (unless absolutely necessary)
- Inline helper classes (extract to separate files if reused)

## Dependency Checking

**When to check:**
- Example directly imports optional libraries (`fitz`, `PIL`, etc.)

**When NOT to check:**
- Only using DataTrove components (they check internally)

**Pattern:**
```python
from datatrove.utils._import_utils import check_required_dependencies

# At top of main() or module level
check_required_dependencies("pymupdf", ["fitz"])

import fitz  # Now safe to import
```

## Documentation Guidelines

**Setup Guides** (in `docs/guides/`):
- Reference example files, don't duplicate code
- Use code blocks to show commands to run
- Keep focused on setup, not implementation details

**Good:**
```markdown
Run the example:
```bash
python spec/phase2/examples/01_basic_filtering_slurm.py
```
```

**Bad:**
```markdown
Create the pipeline:
```python
# 50 lines of embedded Python code...
```
```

## Common Patterns

**Main function:**
```python
def main():
    """Main pipeline execution."""
    logger.info("Starting Example XX: [Title]")

    # Implementation here

    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
```

**Configuration constants:**
```python
# Good - only paths
OUTPUT_DIR = "spec/phase1/output/01_example"
LOGS_DIR = "spec/phase1/logs/01_example"

# Bad - other values belong inline
NUM_WORKERS = 4        # Keep inline for readability
BATCH_SIZE = 32        # Keep inline for readability
```

**Pipeline executors:**
```python
executor = LocalPipelineExecutor(
    pipeline=pipeline,
    tasks=1,
    logging_dir=LOGS_DIR + "/stage1"
)

executor.run()
```

## Checklist for New Examples

- [ ] Has `#!/usr/bin/env python3` shebang
- [ ] Docstring follows "Example XX:" template
- [ ] Imports organized (stdlib → third-party → local)
- [ ] Uses `from datatrove.utils.logging import logger`
- [ ] No `print()` statements
- [ ] No `sys.path.insert(0, 'src')`
- [ ] Has OUTPUT_DIR and LOGS_DIR constants
- [ ] Uses string concatenation for paths (not f-strings)
- [ ] Has `main()` function
- [ ] Has `if __name__ == "__main__": main()`
- [ ] Runs from repo root: `python spec/phaseN/examples/XX_name.py`
- [ ] Spec file exists and follows template
