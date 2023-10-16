import hashlib
import re
import struct
import unicodedata
from collections import defaultdict

import numpy as np
from nltk.tokenize import sent_tokenize


ESCAPE_PLACEHOLDER = " @_ESC_APE_@ "


class ExtensionHelperSD:
    stage_1_signature = ".c4_sig"
    stage_2_duplicates = ".c4_dup"


class ExtensionHelperES:
    stage_1_sequence = ".es_sequence"
    stage_1_sequence_size = ".es_sequence.size"
    stage_2_big_sequence = ".big_sequence"
    stage_2_bytes_offset = ".info"
    stage_3_bytes_ranges = ".bytearange"


PUNCTUATION = "!/—”:％１〈&(、━\\【#%「」，】；+^]~“《„';’{|∶´[=-`*．（–？！：$～«〉,><》)?）。…@_.\"}►»" + "".join(
    map(chr, list(range(0, 32)) + list(range(127, 160)))
)


def simplify_content(text: str):
    # lower case
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", PUNCTUATION))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    text = re.sub(r"\s+", " ", text.strip())
    # diacritics/unicode normalization
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text


def _b2i(b: bytes) -> int:
    return np.frombuffer(b, dtype=np.uint64, count=1, offset=0).item(0)


def str_hash(s: str) -> int:
    h = hashlib.sha1(bytes(s, encoding="utf-8"))
    return _b2i(h.digest())


def merge_docs(sen_list, n_sentences: int = 3) -> dict:
    # TODO IMPROVE!
    def to_sentences(idx: int):
        return {idx + i for i in range(n_sentences)}

    merged = defaultdict(set)
    for doc_id, sent_id in sen_list:
        merged[doc_id].update(to_sentences(sent_id))
    return merged  # {doc_id: set of sent ids}


# https://github.com/ekzhu/datasketch/blob/master/datasketch/hashfunc.py
def sha1_hash32(data):
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def tokenize_with_escapes(text):
    # TODO improve
    escaped_text = text.replace("\n", ESCAPE_PLACEHOLDER)
    sentences = sent_tokenize(escaped_text)
    sentences_with_escapes = [s.replace(ESCAPE_PLACEHOLDER, "\n") for s in sentences]
    return sentences_with_escapes
