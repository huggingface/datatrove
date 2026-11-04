import pytest


pytest.importorskip("hypothesis")
from hypothesis import event, given

from datatrove.data import Document
from datatrove.pipeline.filters import GopherQualityFilter, GopherRepetitionFilter
from datatrove.pipeline.filters.c4_filters import C4QualityFilter
from datatrove.pipeline.filters.fineweb_quality_filter import FineWebQualityFilter

from ...strategies import documents


FILTERS = {
    "gopher_repetition": GopherRepetitionFilter,
    "gopher_quality": GopherQualityFilter,
    "c4_quality": C4QualityFilter,
    "fineweb_quality": FineWebQualityFilter,
}


@pytest.mark.parametrize("name", FILTERS)
@given(doc=documents())
def test_filter_never_raises(name, doc):
    # a fresh filter per example: tokenizers are cached separately, and C4 accumulates stats on its instance
    result = FILTERS[name]().filter(doc)
    assert isinstance(result, bool) or (
        isinstance(result, tuple) and len(result) == 2 and result[0] is False and isinstance(result[1], str)
    ), f"unexpected filter result: {result!r}"
    # shows which checks the generated documents reach: pytest --hypothesis-show-statistics
    event(f"{name}: {result[1] if isinstance(result, tuple) else result}")


# Known open bug, kept out of the property above so it cannot hide unrelated failures.
# strict=True: once #531 is fixed this test passes and fails CI until the marker is removed
# (and the config is added to FILTERS).
@pytest.mark.xfail(
    strict=True, raises=ZeroDivisionError, reason="GopherQualityFilter(min_doc_words=None) on empty text (#531)"
)
@pytest.mark.parametrize("text", ["", " ", "\n\n\n"])
def test_gopher_quality_without_min_doc_words_on_empty_text(text):
    GopherQualityFilter(min_doc_words=None).filter(Document(text=text, id="0"))
