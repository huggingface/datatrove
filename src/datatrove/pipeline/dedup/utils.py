import hashlib
import re
import string

import numpy as np


class ExtensionHelperES:
    stage_1_sequence = ".es_sequence"
    stage_1_sequence_size = ".es_sequence.size"
    stage_2_big_sequence = ".big_sequence"
    stage_2_bytes_offset = ".info"
    stage_3_bytes_ranges = ".byterange"


# taken from
# https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/data_processing/slimpajama/dedup/to_hash.py
def simplify_content(text: str):
    # TODO replace special chars: e' -> e
    # lower cased
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    text = re.sub(r"\s+", " ", text.strip())
    return text


def _b2i(b: bytes) -> int:
    return np.frombuffer(b, dtype=np.uint64, count=1, offset=0).item(0)


def str_hash(s: str) -> int:
    h = hashlib.sha1(bytes(s, encoding="utf-8"))
    return _b2i(h.digest())


def merge_docs(sen_list, n_sentences: int = 3) -> dict:
    # TODO IMPROVE!
    def to_sentences(idx: int):
        return (idx + i for i in range(n_sentences))

    new_l = [[sen_list[0][0], {sen_list[0][1]}]]
    for x in sen_list[1:]:
        if x[0] == new_l[-1][0]:
            new_l[-1][1].update(to_sentences(x[1]))
        else:
            new_l.append([x[0], {x[1]}])

    return {x[0]: x[1] for x in new_l}
