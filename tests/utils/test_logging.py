import os
import subprocess
import sys


def _run(level_env):
    code = (
        "from datatrove.utils.logging import setup_default_logger\n"
        "from loguru import logger\n"
        "setup_default_logger()\n"
        "logger.info('INFO_LINE')\n"
        "logger.warning('WARNING_LINE')\n"
    )
    env = os.environ.copy()
    if level_env is not None:
        env["DATATROVE_LOG_LEVEL"] = level_env
    else:
        env.pop("DATATROVE_LOG_LEVEL", None)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    return result.stderr


def test_default_level_shows_info():
    out = _run(None)
    assert "INFO_LINE" in out
    assert "WARNING_LINE" in out


def test_warning_level_hides_info():
    out = _run("WARNING")
    assert "INFO_LINE" not in out
    assert "WARNING_LINE" in out
