"""
Tests for the training script.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import train_model


def test_train_model():
    """Test that training produces expected artifacts."""
    output_dir = "models_test"

    # Clean up any existing test directory
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    # Train model
    metrics = train_model(output_dir=output_dir, test_size=0.2)

    # Check that artifacts were created
    assert Path(output_dir).exists()
    assert (Path(output_dir) / "model.pkl").exists()
    assert (Path(output_dir) / "scaler.pkl").exists()
    assert (Path(output_dir) / "metrics.json").exists()

    # Check metrics
    assert "test_rmse" in metrics
    assert "train_rmse" in metrics
    assert "test_r2" in metrics
    assert "train_r2" in metrics
    assert metrics["test_rmse"] > 0
    assert metrics["train_rmse"] > 0

    # Check that metrics file matches returned metrics
    with open(Path(output_dir) / "metrics.json") as f:
        saved_metrics = json.load(f)

    assert saved_metrics == metrics

    # Clean up
    shutil.rmtree(output_dir)


def test_train_model_reproducibility():
    """Test that training is reproducible."""
    output_dir1 = "models_test1"
    output_dir2 = "models_test2"

    # Clean up any existing test directories
    for d in [output_dir1, output_dir2]:
        if Path(d).exists():
            shutil.rmtree(d)

    # Train model twice
    metrics1 = train_model(output_dir=output_dir1, test_size=0.2)
    metrics2 = train_model(output_dir=output_dir2, test_size=0.2)

    # Check that metrics are identical
    assert metrics1["test_rmse"] == metrics2["test_rmse"]
    assert metrics1["train_rmse"] == metrics2["train_rmse"]
    assert metrics1["test_r2"] == metrics2["test_r2"]
    assert metrics1["train_r2"] == metrics2["train_r2"]

    # Clean up
    for d in [output_dir1, output_dir2]:
        shutil.rmtree(d)

