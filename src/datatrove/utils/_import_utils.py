import importlib.metadata
import importlib.resources
import importlib.util
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
        The pip name may be a PEP 508 requirement, including a direct URL reference; only the
        distribution name is compared against installed packages.

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


def _normalize_distribution_name(distribution_name: str) -> str:
    """Extract a comparable distribution name from a pip requirement string.

    `_requires_dependencies` stores the argument passed to `pip install`, which may be a
    PEP 508 specifier such as ``readability-lxml @ git+https://...``. importlib.metadata
    exposes only the distribution Name (``readability-lxml``).

    Args:
        distribution_name: Raw pip requirement or distribution name.

    Returns:
        Lowercased distribution name with extras, version specifiers, markers, and URL
        references stripped.
    """
    name = distribution_name.strip()
    name = name.split("@", 1)[0]
    name = name.split(";", 1)[0]
    name = name.split("[", 1)[0]
    for sep in ("===", "==", "!=", "~=", ">=", "<=", ">", "<"):
        name = name.split(sep, 1)[0]
    return name.strip().lower()


# Distribution Check
@lru_cache
def _is_distribution_available(distribution_name: str) -> bool:
    normalized = _normalize_distribution_name(distribution_name)
    if not normalized:
        return False
    for dist in importlib.metadata.distributions():
        metadata = getattr(dist, "metadata", None)
        dist_name = metadata.get("Name") if metadata is not None else None
        if dist_name and dist_name.lower() == normalized:
            return True
    return False


# Used in tests


def is_boto3_available():
    return _is_package_available("boto3")


def is_s3fs_available():
    return _is_package_available("s3fs")


def is_moto_available():
    return _is_package_available("moto")


def is_torch_available():
    return _is_package_available("torch")


def is_dnspython_available():
    return _is_package_available("dnspython")
