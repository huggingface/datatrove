import math
import tempfile
from functools import cache

import pytest


pytest.importorskip("hypothesis")
from hypothesis import given

from datatrove.data import Document
from datatrove.pipeline.stats import DocStats, LineStats, ParagraphStats, SentenceStats, WordStats

from ...strategies import doc_text, documents


STATS = {
    "doc": DocStats,
    "line": LineStats,
    "paragraph": ParagraphStats,
    "word": WordStats,
    "sentence": SentenceStats,
}


@cache
def get_stats(name: str):
    # extract_stats() keeps no state and never writes; the output folder is only required by the constructor.
    # Built once because the constructor is slow (it sets up a domain-name parser).
    return STATS[name](tempfile.gettempdir())


def assert_finite_stats(name, doc):
    stats = get_stats(name).extract_stats(doc)
    assert stats, f"{name}: no stats returned"
    for key, value in stats.items():
        assert isinstance(value, (int, float)) and math.isfinite(value), f"{name}: {key}={value!r}"


@pytest.mark.parametrize("name", STATS)
@given(doc=documents(doc_text()))
def test_stats_are_finite_numbers(name, doc):
    assert_finite_stats(name, doc)


# Empty and whitespace-only text used to raise ZeroDivisionError (#512); keep explicit cases for each block.
BLANK_TEXTS = ["", " ", "\n", "\n\n", "\t", "\r\n", "\u00a0"]


@pytest.mark.parametrize("text", BLANK_TEXTS)
@pytest.mark.parametrize("name", STATS)
def test_stats_on_blank_text(name, text):
    assert_finite_stats(name, Document(text=text, id="0"))
