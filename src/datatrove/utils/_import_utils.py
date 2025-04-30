import importlib.resources
import os
from functools import lru_cache
from typing import NoReturn


ASSETS_PATH = os.path.join(importlib.resources.files(__package__.split(".")[0]), "assets")


def check_required_dependencies(step_name: str, required_dependencies: list[str] | list[tuple[str, str]]):
    """Check whether the required dependencies are installed or not.

    Args:
        step_name: str
            The name of the step
        required_dependencies: List[str] | List[tuple[str, str]]
        required dependencies. If the format is a tuple, it is checked as (module name, pip name).
        When provided as a tuple, an error will be raised if the top-level module name is correct but the pip distribution name differs
        (e.g., (fasttext, fasttext-numpy2-wheel)).

    """
    missing_dependencies: dict[str, str] = {}
    for dependency in required_dependencies:
        dependency = dependency if isinstance(dependency, tuple) else (dependency, dependency)
        package_name, pip_name = dependency
        # case1: in case we didn't install package
        if not _is_package_available(package_name):
            missing_dependencies[package_name] = pip_name
        # case2: top-level package is installed but distribution is incorrect. (i. e. fasttext-numpy2-wheel; compatibility for numpy2)
        if not _is_distribution_available(pip_name):
            missing_dependencies[package_name] = pip_name
    if missing_dependencies:
        _raise_error_for_missing_dependencies(step_name, missing_dependencies)


def _raise_error_for_missing_dependencies(step_name: str, dependencies: dict[str, str]) -> NoReturn:
    """Helper to raise an ImportError for missing dependencies and prompt the user to install said dependencies

    Args:
        step_name: str
            The name of the step
        dependencies: dict[str, str]
            The missing dependencies

    """
    dependencies = dict(sorted(dependencies.items()))
    package_names = list(dependencies)
    if len(dependencies) > 1:
        package_names = (
            f"{','.join('`' + package_name + '`' for package_name in package_names[:-1])} and `{package_names[-1]}`"
        )
    else:
        package_names = f"`{package_names[0]}`"
    raise ImportError(
        f"Please install {package_names} to use {step_name} (`pip install {' '.join(list(dependencies.values()))}`)."
    )


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


def is_fasteners_available():
    return _is_package_available("fasteners")


# Distribution Check
@lru_cache
def _is_distribution_available(distribution_name: str):
    found = None
    for dist in importlib.metadata.distributions():
        if dist.metadata["Name"] and dist.metadata["Name"].lower() == distribution_name:
            found = True
    return found


# Used in tests


def is_boto3_available():
    return _is_package_available("boto3")


def is_s3fs_available():
    return _is_package_available("s3fs")


def is_moto_available():
    return _is_package_available("moto")


def is_torch_available():
    return _is_package_available("torch")
