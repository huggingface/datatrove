import itertools
import unittest
from functools import wraps
from typing import get_args, get_type_hints

from datatrove.utils.hashing import HashConfig


def use_hash_configs(
    precision: list[int] = list(get_args(get_type_hints(HashConfig)["precision"])), hash_fc: list[str] = ["xxhash"]
):
    """
    Decorator which runs the wrapped test function, with hash config of all combinations of given precision and hash_fc
    Args:
        precision (list[int]): List of precision values to use. Defaults to all possible values.
        hash_fc (list[str]): List of hash functions to use. Defaults to ["xxhash"].
    """

    def wrapper(f):
        @wraps(f)
        def inner_wraper(self: unittest.TestCase, *args, **kwargs):
            for p, h in itertools.product(precision, hash_fc):
                config = HashConfig(precision=p, hash_fc=h)
                self.setUp()
                f(self, *args, config, **kwargs)
                self.tearDown()
                self.doCleanups()

        return inner_wraper

    return wrapper


def require_nltk(test_case):
    try:
        import nltk  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires nltk")(test_case)
    return test_case


def require_tokenizers(test_case):
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires tokenizers")(test_case)
    return test_case


def require_trafilatura(test_case):
    try:
        import trafilatura  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires trafilatura")(test_case)
    return test_case


def require_readability(test_case):
    try:
        import readability  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires readability")(test_case)
    return test_case


def require_inscriptis(test_case):
    try:
        import inscriptis  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires inscriptis")(test_case)
    return test_case


def require_pyarrow(test_case):
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires pyarrow")(test_case)
    return test_case


def require_fasttext(test_case):
    try:
        import fasttext  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires fasttext")(test_case)
    return test_case


def require_tldextract(test_case):
    try:
        import tldextract  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires tldextract")(test_case)
    return test_case


def require_boto3(test_case):
    try:
        import boto3  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires boto3")(test_case)
    return test_case


def require_moto(test_case):
    try:
        import moto  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires moto")(test_case)
    return test_case


def require_s3fs(test_case):
    try:
        import s3fs  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires s3fs")(test_case)
    return test_case


def require_datasets(test_case):
    try:
        import datasets  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires datasets")(test_case)
    return test_case


def require_xxhash(test_case):
    try:
        import xxhash  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires xxhash")(test_case)
    return test_case


def require_lighteval(test_case):
    try:
        import lighteval  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires lighteval")(test_case)
    return test_case
