import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from examples.inference.generate_data import _get_executor_job_id, _load_generation_config


def test_generate_data_help() -> None:
    script = Path(__file__).resolve().parents[3] / "examples" / "inference" / "generate_data.py"

    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "--model-name-or-path" in result.stdout


def test_load_generation_config_falls_back_to_model_config() -> None:
    sentinel = object()

    with (
        patch("examples.inference.generate_data.GenerationConfig.from_pretrained", side_effect=OSError("missing")),
        patch(
            "examples.inference.generate_data.GenerationConfig.from_model_config", return_value=sentinel
        ) as fallback,
    ):
        result = _load_generation_config("Qwen/Qwen3.5-2B", "main", False, config={"model_type": "qwen3_5"})

    assert result is sentinel
    fallback.assert_called_once_with({"model_type": "qwen3_5"})


def test_get_executor_job_id_handles_local_and_slurm_executors() -> None:
    class LocalExecutor:
        pass

    class SlurmExecutor:
        job_id = "12345"

    assert _get_executor_job_id(LocalExecutor()) is None
    assert _get_executor_job_id(SlurmExecutor()) == "12345"
