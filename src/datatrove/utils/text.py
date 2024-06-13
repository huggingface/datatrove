import re
import unicodedata
from dataclasses import dataclass
from itertools import tee
from typing import Iterable

from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


PUNCTUATION = "!/—”:％１〈&(、━\\【#%「」，】；+^]~“《„';’{|∶´[=-`*．（–？！：$～«〉,><》)?）。…@_.\"}►»" + "".join(
    map(
        chr,
        (x for a, b in ((0, 9), (11, 13), (13, 32), (127, 160)) for x in range(a, b)),
    )
)
PUNCTUATION_SET = set(PUNCTUATION)


@dataclass
class TextNormConfig:
    lowercase: bool = True
    norm_whitespace: bool = True
    remove_punctuation: bool = True
    norm_unicode_diacritics: bool = True
    norm_numbers: bool = True
    norm_weekdays: bool = False
    norm_monthnames: bool = False


DEF_TEXT_NORM_CONFIG = TextNormConfig()
NUMBERS_PATTERN = re.compile(r"\d+")
WHITESPACE_PATTERN = re.compile(r"\s+")
# WARNING: english specific
WEEKDAYS_PATTERN = re.compile(r"monday|tuesday|wednesday|thursday|friday|saturday|sunday")
MONTHS_PATTERN = re.compile(r"january|february|march|april|may|june|july|august|september|october|november|december")


def simplify_text(text: str, config=DEF_TEXT_NORM_CONFIG) -> str:
    """Performs the following operations to increase recall when looking for matches between documents:
    - lowercase text
    - replace all whitespace with a single " "
    - remove all punctuation
    - convert diacritics
    - unicode normalize

    Args:
        text

    Returns:
        modified text
    """
    # lower case
    if config.lowercase:
        text = text.lower()
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    if config.norm_whitespace:
        text = WHITESPACE_PATTERN.sub(" ", text.strip())
    # remove punctuation
    if config.remove_punctuation:
        text = text.translate(str.maketrans("", "", PUNCTUATION))
    # diacritics/unicode normalization
    if config.norm_unicode_diacritics:
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    if config.norm_numbers:
        text = NUMBERS_PATTERN.sub("0", text)
    if config.norm_weekdays:
        text = WEEKDAYS_PATTERN.sub("WEEKDAY", text)
    if config.norm_monthnames:
        text = MONTHS_PATTERN.sub("MONTH", text)
    return text.strip()


# from https://tedboy.github.io/nlps/_modules/nltk/util.html#ngrams
def ngrams(sequence: Iterable, n: int):
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.


SPLIT_TEXT_DOCUMENTS = "DOCUMENT"
SPLIT_TEXT_SENTENCES = "SENTENCE"
SPLIT_TEXT_PARAGRAPHS = "PARAGRAPH"


def split_into_parts(text, mode="DOCUMENT", language=Languages.english):
    if mode == SPLIT_TEXT_DOCUMENTS:
        return [text]
    elif mode == SPLIT_TEXT_SENTENCES:
        tokenizer = load_word_tokenizer(language)
        spans = [b for _, b in tokenizer.span_tokenize(text)]
        return [text[a:b] for a, b in zip([0] + spans[:-1], spans[:-1] + [len(text)])]
    elif mode == SPLIT_TEXT_PARAGRAPHS:
        # merge whitespace with prev line
        og_lines = text.splitlines()
        lines = []
        next_line = []
        for li, line in enumerate(og_lines):
            if line.strip() and next_line:
                lines.append("".join(next_line))
                next_line = []
            next_line.append(line)
            if li != len(og_lines) - 1:
                next_line.append("\n")
        if next_line:
            lines.append("".join(next_line))
        return lines
    else:
        raise ValueError(f"Unknown {mode=}")
