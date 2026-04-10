from unittest.mock import MagicMock, patch

from examples.inference.generate_data import _get_executor_job_id, _load_generation_defaults


def test_load_generation_defaults_reads_only_explicit_values() -> None:
    generation_config = MagicMock()
    generation_config.to_diff_dict.return_value = {
        "temperature": 0.6,
        "top_p": 0.95,
        "bos_token_id": 42,
    }

    with patch(
        "examples.inference.generate_data.GenerationConfig.from_pretrained", return_value=generation_config
    ) as from_pretrained:
        result = _load_generation_defaults("Qwen/Qwen3.5-2B", "main", False)

    assert result == {"temperature": 0.6, "top_p": 0.95}
    from_pretrained.assert_called_once_with("Qwen/Qwen3.5-2B", revision="main", trust_remote_code=False)


def test_load_generation_defaults_falls_back_to_server_defaults() -> None:
    with patch("examples.inference.generate_data.GenerationConfig.from_pretrained", side_effect=OSError("missing")):
        result = _load_generation_defaults("Qwen/Qwen3.5-2B", "main", False)

    assert result == {}


def test_get_executor_job_id_handles_local_and_slurm_executors() -> None:
    class LocalExecutor:
        pass

    class SlurmExecutor:
        job_id = "12345"

    assert _get_executor_job_id(LocalExecutor()) is None
    assert _get_executor_job_id(SlurmExecutor()) == "12345"
