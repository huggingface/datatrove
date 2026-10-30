import tempfile

import pytest


pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

from datatrove.data import Document
from datatrove.pipeline.dedup.sentence_dedup import SentDedupConfig, SentenceDedupFilter
from datatrove.utils.text import TERMINAL_PUNCTUATION
from datatrove.utils.word_tokenizers import WhitespaceTokenizer, load_word_tokenizer

from ...strategies import doc_text


# Sentence dedup cuts documents at span_tokenize() boundaries, so text outside every span is lost.
# Oracle: spans are ordered, non-overlapping, inside the text, and cover every non-whitespace character.


def assert_spans_cover_text(spans, text):
    covered = set()
    previous_end = 0
    for start, end in spans:
        assert previous_end <= start < end <= len(text), f"bad span {(start, end)} after {previous_end}"
        covered.update(range(start, end))
        previous_end = end
    uncovered = [i for i, char in enumerate(text) if not char.isspace() and i not in covered]
    assert not uncovered, f"characters outside every span: {[text[i] for i in uncovered][:10]}"


@given(text=doc_text(allow_empty=False, allow_blank=False))
def test_spacy_spans_cover_text(text):
    pytest.importorskip("spacy")
    assert_spans_cover_text(load_word_tokenizer("en").span_tokenize(text), text)


# WhitespaceTokenizer (the fallback for ~70 languages) ends a sentence at terminal punctuation or a newline, and
# needs a character before it. Until the known bug below is fixed, only text made of such sentences is used.
sentence = st.builds(
    lambda words, end: " ".join(words) + end,
    st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10), min_size=1, max_size=8),
    st.sampled_from(sorted(TERMINAL_PUNCTUATION) + ["\n"]),
)
text_of_sentences = st.builds(
    lambda sentences, separator: separator.join(sentences),
    st.lists(sentence, min_size=1, max_size=6),
    st.sampled_from([" ", "\n", "\n\n"]),
)


@given(text=text_of_sentences)
def test_whitespace_tokenizer_spans_cover_text(text):
    assert_spans_cover_text(WhitespaceTokenizer().span_tokenize(text), text)


# Known bug. strict=True: once fixed these pass and fail CI until the markers are removed.
# (With exactly one recognized sentence, the span covers the whole text, so the tail is not always dropped.)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="WhitespaceTokenizer can drop trailing text and punctuation-only sentences",
)
@pytest.mark.parametrize("text", ["Hello world", "One. Two. three four", "?"])
def test_whitespace_tokenizer_keeps_trailing_text(text):
    assert_spans_cover_text(WhitespaceTokenizer().span_tokenize(text), text)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="sentence dedup can delete trailing text for WhitespaceTokenizer languages",
)
def test_sentence_dedup_keeps_trailing_text():
    # "crm" uses WhitespaceTokenizer; only the duplicated first sentence should be removed
    with tempfile.TemporaryDirectory() as folder:
        dedup = SentenceDedupFilter(data_folder=folder, language="crm", config=SentDedupConfig(n_sentences=1))
        document = Document(text="Dup line here. Kept one. trailing tail", id="0")
        kept_text, _ = dedup.remove_dup_sentences(document, [0])
    assert kept_text == "Kept one. trailing tail"
