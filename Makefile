.PHONY: quality style quality-full style-full test

check_dirs := src tests examples
changed_python_files := $(shell (git diff --name-only --diff-filter=ACMR HEAD -- $(check_dirs); git ls-files --others --exclude-standard -- $(check_dirs)) 2>/dev/null | awk '/\.py$$/' | sort -u)

quality-full:
	ruff check $(check_dirs)  # linter
	ruff format --check $(check_dirs) # formatter

style-full:
	ruff check --fix $(check_dirs) # linter
	ruff format $(check_dirs) # formatter

quality:
ifeq ($(strip $(changed_python_files)),)
	@echo "No changed Python files in $(check_dirs)."
else
	ruff check $(changed_python_files)  # linter
	ruff format --check $(changed_python_files) # formatter
endif

style:
ifeq ($(strip $(changed_python_files)),)
	@echo "No changed Python files in $(check_dirs)."
else
	ruff check --fix $(changed_python_files) # linter
	ruff format $(changed_python_files) # formatter
endif

test:
	python -m pytest -sv ./tests/
