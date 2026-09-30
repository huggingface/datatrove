import unicodedata

import pytest


pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

from datatrove.utils.text import simplify_text


# simplify_text() normalizes text for dedup and decontamination keys. Oracle: normalizing twice changes nothing,
# otherwise the same text can get different keys depending on earlier processing.


def has_no_combining_mark(char):
    # simplify_text() removes combining marks (category Mn) after NFD decomposition, as its last step
    return all(unicodedata.category(part) != "Mn" for part in unicodedata.normalize("NFD", char))


# Characters that are or contain a combining mark are excluded until the known bug below is fixed.
@given(text=st.text(st.characters(exclude_categories=["Cs"]).filter(has_no_combining_mark), max_size=200))
def test_simplify_text_is_idempotent(text):
    once = simplify_text(text)
    assert simplify_text(once) == once


# Known bug. strict=True: once fixed these pass and fail CI until the markers are removed.
# Open question: is idempotence intended? The comment in simplify_text() suggests the steps should be consistent.
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="combining marks are removed after number/space/punctuation steps"
)
@pytest.mark.parametrize("text", ["0\u03000", "a \u0301 b", "\u2260"])
def test_simplify_text_is_idempotent_with_combining_marks(text):
    once = simplify_text(text)
    assert simplify_text(once) == once
