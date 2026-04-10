from types import SimpleNamespace
from unittest.mock import patch

from datatrove.utils._import_utils import _is_distribution_available


def test_is_distribution_available_ignores_distributions_without_name() -> None:
    missing_metadata = SimpleNamespace(metadata=None)
    nameless = SimpleNamespace(metadata={})
    target = SimpleNamespace(metadata={"Name": "vllm"})

    _is_distribution_available.cache_clear()
    with patch("importlib.metadata.distributions", return_value=[missing_metadata, nameless, target]):
        assert _is_distribution_available("vllm") is True
    _is_distribution_available.cache_clear()
