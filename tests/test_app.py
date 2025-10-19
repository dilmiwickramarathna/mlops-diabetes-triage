"""
Tests for the diabetes progression prediction API.
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import app, load_model_artifacts

# Load model artifacts before running tests
load_model_artifacts()

client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data


def test_predict_valid_input():
    """Test prediction with valid input."""
    payload = {
        "age": 0.02,
        "sex": -0.044,
        "bmi": 0.06,
        "bp": -0.03,
        "s1": -0.02,
        "s2": 0.03,
        "s3": -0.02,
        "s4": 0.02,
        "s5": 0.02,
        "s6": -0.001,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_predict_missing_field():
    """Test prediction with missing required field."""
    payload = {
        "age": 0.02,
        "sex": -0.044,
        # Missing other required fields
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_invalid_range():
    """Test prediction with value outside reasonable range."""
    payload = {
        "age": 100.0,  # Way outside expected range
        "sex": -0.044,
        "bmi": 0.06,
        "bp": -0.03,
        "s1": -0.02,
        "s2": 0.03,
        "s3": -0.02,
        "s4": 0.02,
        "s5": 0.02,
        "s6": -0.001,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_multiple_samples():
    """Test predictions for multiple sample inputs."""
    samples = [
        {
            "age": 0.05,
            "sex": 0.05,
            "bmi": 0.06,
            "bp": 0.02,
            "s1": -0.03,
            "s2": -0.02,
            "s3": -0.01,
            "s4": 0.01,
            "s5": 0.04,
            "s6": 0.02,
        },
        {
            "age": -0.05,
            "sex": -0.04,
            "bmi": -0.03,
            "bp": -0.02,
            "s1": 0.02,
            "s2": 0.03,
            "s3": 0.01,
            "s4": -0.01,
            "s5": -0.02,
            "s6": -0.01,
        },
    ]

    for sample in samples:
        response = client.post("/predict", json=sample)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data


def test_model_artifacts_exist():
    """Test that model artifacts exist."""
    models_dir = Path("models")
    assert models_dir.exists()
    assert (models_dir / "model.pkl").exists()
    assert (models_dir / "scaler.pkl").exists()
    assert (models_dir / "metrics.json").exists()


def test_metrics_file_valid():
    """Test that metrics file is valid JSON and contains expected keys."""
    metrics_path = Path("models") / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    required_keys = ["model_version", "test_rmse", "train_rmse", "test_r2", "train_r2"]
    for key in required_keys:
        assert key in metrics
        if key != "model_version":
            assert isinstance(metrics[key], (int, float))

