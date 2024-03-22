import hashlib
import re
import struct
import unicodedata
from dataclasses import dataclass


PUNCTUATION = "!/—”:％１〈&(、━\\【#%「」，】；+^]~“《„';’{|∶´[=-`*．（–？！：$～«〉,><》)?）。…@_.\"}►»" + "".join(
    map(chr, (x for a, b in ((0, 9), (11, 13), (13, 32), (127, 160)) for x in range(a, b)))
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


# https://github.com/ekzhu/datasketch/blob/master/datasketch/hashfunc.py
def sha1_hash32(data):
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def sha1_hash64(data):
    """A 64-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 64-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 64 bits.
    """
    return struct.unpack("<Q", hashlib.sha1(data).digest()[:8])[0]
