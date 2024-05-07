import importlib.resources
import os
from functools import lru_cache


ASSETS_PATH = os.path.join(importlib.resources.files(__package__.split(".")[0]), "assets")


@lru_cache
def _is_package_available(package_name):
    """

    Args:
      package_name:

    Returns:

    """
    return importlib.util.find_spec(package_name) is not None


def is_rich_available():
    return _is_package_available("rich")


def is_pyarrow_available():
    return _is_package_available("pyarrow")


def is_tokenizers_available():
    return _is_package_available("tokenizers")


# Used in tests


def is_boto3_available():
    return _is_package_available("boto3")


def is_s3fs_available():
    return _is_package_available("s3fs")


def is_moto_available():
    return _is_package_available("moto")


def is_torch_available():
    return _is_package_available("torch")
