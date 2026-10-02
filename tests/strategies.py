"""Shared Hypothesis strategies for property-based tests."""

from hypothesis import strategies as st

from datatrove.data import Document
from datatrove.pipeline.filters.gopher_quality_filter import STOP_WORDS


# Words for generated prose: the Gopher stop words (so prose passes the stop-word check) mixed with random
# lowercase words (lengths 3-10, inside Gopher's mean word length bounds). Long enough prose gets past the
# early "too short" / "too few sentences" checks of the quality filters.
_word = st.one_of(
    st.sampled_from(STOP_WORDS),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=10),
)
_sentence = st.builds(
    lambda words, end: " ".join(words).capitalize() + end,
    st.lists(_word, min_size=4, max_size=15),
    st.sampled_from([".", ".", ".", "!", "?", "...", ""]),
)
_line = st.builds(
    lambda sentences, prefix: prefix + " ".join(sentences),
    st.lists(_sentence, min_size=1, max_size=2),
    st.sampled_from(["", "", "", "- ", "* "]),  # occasional bullet lines (Gopher bullet ratio)
)
# Multi-line, multi-paragraph prose: long enough to reach the deeper checks of the quality filters.
_prose = st.builds(
    lambda paragraphs, newline: (newline * 2).join(newline.join(lines) for lines in paragraphs),
    st.lists(st.lists(_line, min_size=1, max_size=5), min_size=1, max_size=3),
    st.sampled_from(["\n", "\n", "\r\n"]),
)

# Degenerate shapes that have crashed datatrove components before.
_whitespace = st.text(alphabet=" \t\n\r", min_size=1, max_size=20)
_repeated = st.builds(
    lambda line, n, sep: sep.join([line] * n),
    st.text(min_size=1, max_size=40),
    st.integers(min_value=2, max_value=6),
    st.sampled_from(["\n", "\n\n"]),
)
_punctuation_only = st.text(alphabet=".,;:!?-*#…'\"()", min_size=1, max_size=50)
# built by repetition: generating 1000+ random characters is slow
_long_word = st.builds(
    lambda char, n: char * n, st.sampled_from("abcxyz"), st.integers(min_value=1000, max_value=1200)
)


def doc_text(max_size: int = 500, allow_empty: bool = True, allow_blank: bool = True) -> st.SearchStrategy[str]:
    """Document text. allow_empty: include "". allow_blank: include whitespace-only text.
    st.text() excludes lone surrogates by default; that is deliberate until the surrogate policy is decided."""
    random_text = st.text(min_size=0 if allow_empty else 1, max_size=max_size)
    repeated = _repeated
    if not allow_blank:
        random_text = random_text.filter(str.strip)
        repeated = repeated.filter(str.strip)
    shapes = [_prose, repeated, _punctuation_only, _long_word, random_text]
    if allow_blank:
        shapes.insert(0, _whitespace)
    if allow_empty:
        shapes.insert(0, st.just(""))
    return st.one_of(shapes)


def documents(text: st.SearchStrategy[str] | None = None) -> st.SearchStrategy[Document]:
    # metadata is always empty here: filters and stats only look at text. Use a dedicated strategy for metadata.
    return st.builds(
        Document, text=text if text is not None else doc_text(), id=st.just("0"), metadata=st.builds(dict)
    )
