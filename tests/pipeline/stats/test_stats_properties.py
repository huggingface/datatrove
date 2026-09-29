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


# Empty and whitespace-only text is excluded until #512 lands; see the known-bug test below.
@pytest.mark.parametrize("name", STATS)
@given(doc=documents(doc_text(allow_empty=False, allow_blank=False)))
def test_stats_are_finite_numbers(name, doc):
    stats = get_stats(name).extract_stats(doc)
    assert stats, f"{name}: no stats returned"
    for key, value in stats.items():
        assert isinstance(value, (int, float)) and math.isfinite(value), f"{name}: {key}={value!r}"


# Known open bug (#512): these stats blocks raise ZeroDivisionError on these blank texts on main.
# strict=True: once #512 is fixed they pass and fail CI until removed; then allow blank text in the property above.
ALL_BLANK = ["", " ", "\n", "\n\n", "\t", "\r\n", "\u00a0"]
BLANK_TEXT_CRASHES = {
    "doc": [""],
    "line": ["", "\n", "\n\n"],
    "paragraph": ALL_BLANK,
    "word": ALL_BLANK,
    "sentence": ALL_BLANK,
}


@pytest.mark.xfail(strict=True, raises=ZeroDivisionError, reason="stats blocks on empty/whitespace text (#512)")
@pytest.mark.parametrize(
    ("name", "text"), [(name, text) for name, texts in BLANK_TEXT_CRASHES.items() for text in texts]
)
def test_stats_on_blank_text(name, text):
    get_stats(name).extract_stats(Document(text=text, id="0"))
