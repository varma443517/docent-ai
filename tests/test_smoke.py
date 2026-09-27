"""A first test so CI has something to run. Real tests arrive with llm.py."""

from typer.testing import CliRunner

from docent import __version__
from docent.cli import app


def test_version_command_prints_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
