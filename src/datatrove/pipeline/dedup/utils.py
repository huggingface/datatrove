import hashlib
import re
import string


# taken from
# https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/data_processing/slimpajama/dedup/to_hash.py
def simplify_content(text: str):
    # lower cased
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    text = re.sub(r"\s+", " ", text.strip())
    return text


def get_hash(s: str):
    s = bytes(s, encoding="utf-8")
    return hashlib.sha1(s)
