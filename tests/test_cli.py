"""Smoke tests de CLI: os 5 scripts precisam expor --help sem erro."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "run_evaluation.py",
    "run_cross_label_evaluation.py",
    "run_impact_study.py",
    "train_cnn3d.py",
    "train_emotion_model.py",
    "train_multimodal.py",
]


def _help(script: str) -> str:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_scripts_show_help():
    for script in SCRIPTS:
        out = _help(script)
        assert "usage:" in out, f"{script} sem usage no --help"


def test_train_scripts_expose_config_flag():
    for script in ("train_cnn3d.py", "train_emotion_model.py", "train_multimodal.py"):
        out = _help(script)
        assert "--config" in out, f"{script} sem --config"
        assert "--device" in out
        assert "--epochs" in out
