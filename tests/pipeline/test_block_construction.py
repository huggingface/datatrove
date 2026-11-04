"""Every pipeline block must construct with only its required arguments.

Optional arguments keep their defaults, so a default that crashes the constructor
(as in #549 and #552) fails here. Each block is also deep-copied and round-tripped
through dill, because the executors do both before they run a pipeline.
"""

import copy
import importlib
import inspect
import sys
from itertools import chain
from pathlib import Path

import dill
import pytest

import datatrove.pipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils._import_utils import check_required_dependencies


# Values for required arguments, by argument name. A new block with a required argument that is
# not listed here fails the test: add a plausible value for it.
REQUIRED_ARGUMENT_VALUES = {
    "bucket": "user/bucket",
    "dataset": "user/dataset",
    "filter_function": lambda doc: True,
    "index_name": "index",
    "language": "en",
    "mime_types": ["text/html"],
    "model_dataset": "wikipedia",
    "model_url": "model.bin",
    "regex_exp": "x",
    "save_filename": "merged",
    "tasks_stage_1": 1,
    "tokenizer_name_or_path": "gpt2",
    "words": ["word"],
}


def _bloom_filter_config():
    from datatrove.pipeline.dedup.bloom_filter import BloomFilterConfig

    return BloomFilterConfig(m_bytes=1024, k=3)


def _exact_dedup_config():
    from datatrove.pipeline.dedup.exact_dedup import ExactDedupConfig

    return ExactDedupConfig(content_getter=lambda doc: doc.text)


# Blocks with a required `config` argument.
CONFIG_FACTORIES = {
    "SingleBloomFilter": _bloom_filter_config,
    "ExactDedupSignature": _exact_dedup_config,
    "ExactFindDedups": _exact_dedup_config,
    "ExactDedupFilter": _exact_dedup_config,
    "ExactDedupBuildIndex": _exact_dedup_config,
}

SKIPPED_BLOCKS = {
    # required arguments are rich objects; their own test modules cover them
    "InferenceDatasetCardGenerator": "needs InferenceDatasetCardParams",
    "InferenceProgressMonitor": "needs InferenceDatasetCardParams",
    "InferenceRunner": "needs a rollout function, an InferenceConfig and an output writer",
    # keeps this test offline
    "UnigramLogProbFilter": "downloads word frequencies in __init__",
}


def _pipeline_blocks() -> list[type[PipelineStep]]:
    # walk the files, because some subpackages (e.g. media) have no __init__.py
    pipeline_dir = Path(datatrove.pipeline.__file__).parent
    for path in sorted(pipeline_dir.rglob("*.py")):
        parts = path.relative_to(pipeline_dir).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        try:
            importlib.import_module(".".join(("datatrove.pipeline", *parts)))
        except ImportError as error:
            if (error.name or "").split(".")[0] == "datatrove":
                raise
            # an optional third-party package is missing or broken, so this module's blocks are not tested
    found = set()
    to_visit = [PipelineStep]
    while to_visit:
        for subclass in to_visit.pop().__subclasses__():
            if subclass not in found:
                found.add(subclass)
                to_visit.append(subclass)
    blocks = [
        block
        for block in found
        if block.__module__.startswith("datatrove.pipeline.")
        and not inspect.isabstract(block)
        and not block.__name__.startswith("_")
    ]
    return sorted(blocks, key=lambda block: (block.__module__, block.__name__))


BLOCKS = _pipeline_blocks()


def _argument_value(block: type[PipelineStep], name: str, tmp_path) -> object:
    if name.endswith("_folder") or name.endswith("_dir"):
        return str(tmp_path / name)
    if name == "config" and block.__name__ in CONFIG_FACTORIES:
        return CONFIG_FACTORIES[block.__name__]()
    if name not in REQUIRED_ARGUMENT_VALUES:
        raise KeyError(
            f"{block.__name__} requires {name!r}: add a value to REQUIRED_ARGUMENT_VALUES or CONFIG_FACTORIES"
        )
    return REQUIRED_ARGUMENT_VALUES[name]


def _construct(block: type[PipelineStep], tmp_path) -> PipelineStep:
    if block.__name__ in SKIPPED_BLOCKS:
        pytest.skip(SKIPPED_BLOCKS[block.__name__])
    dependencies = list(chain.from_iterable(getattr(cls, "_requires_dependencies", []) for cls in block.mro()))
    if dependencies:
        try:
            check_required_dependencies(block.__name__, dependencies)
        except ImportError as error:
            pytest.skip(f"optional dependency not installed: {error}")
    required = [
        parameter.name
        for parameter in inspect.signature(block.__init__).parameters.values()
        if parameter.name != "self"
        and parameter.default is parameter.empty
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]
    return block(**{name: _argument_value(block, name, tmp_path) for name in required})


def test_discovers_blocks():
    names = {block.__name__ for block in BLOCKS}
    assert {"JsonlReader", "JsonlWriter", "HuggingFaceDatasetWriter"} <= names
    # media has no __init__.py; this module of it needs no optional dependency
    assert "datatrove.pipeline.media.media_readers.base" in sys.modules


@pytest.mark.parametrize("block", BLOCKS, ids=lambda block: block.__name__)
def test_block_constructs_with_only_required_arguments(block, tmp_path):
    _construct(block, tmp_path)


@pytest.mark.parametrize("block", BLOCKS, ids=lambda block: block.__name__)
def test_block_survives_deepcopy_and_dill(block, tmp_path):
    step = _construct(block, tmp_path)
    assert type(copy.deepcopy(step)) is block
    assert type(dill.loads(dill.dumps(step))) is block
