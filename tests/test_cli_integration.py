import subprocess
import sys
import pytest

def test_training_cli_runs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "agents/qlearning.py",
            "train",
            "--episodes", "900000",
            "--seed", "1",
            "--save", str(tmp_path / "qtable.pkl"),
        ],
        cwd=".", capture_output=True, text=True
    )

    assert result.returncode == 0
    assert (tmp_path / "qtable.pkl").exists()

    assert "Saved Q-table" in result.stdout