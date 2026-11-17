import subprocess
import sys
from pathlib import Path

COMMAND = Path(sys.executable).parent / "vak"  # the script uv installs from [project.scripts]


def test_the_installed_command_runs_and_explains_itself():
    result = subprocess.run([COMMAND, "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.startswith("usage: vak")


def test_with_no_command_it_prints_the_help():
    result = subprocess.run([COMMAND], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.startswith("usage: vak")


def test_an_unknown_option_is_an_error():
    result = subprocess.run([COMMAND, "--bogus"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "unrecognized arguments: --bogus" in result.stderr
