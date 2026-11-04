import subprocess
import sys


def test_import_utils_work_in_a_fresh_interpreter():
    # importlib.metadata and importlib.util are submodules: they must be imported explicitly.
    # A fresh interpreter is needed because pytest itself has already imported them.
    code = (
        "from datatrove.utils._import_utils import _is_distribution_available, _is_package_available;"
        "_is_package_available('os');"
        "_is_distribution_available('numpy')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
