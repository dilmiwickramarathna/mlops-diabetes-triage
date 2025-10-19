"""
FastAPI service for diabetes progression prediction.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Progression Prediction API",
    description="ML service for predicting short-term diabetes progression risk",
    version="0.2.0",
)

# Global variables for model and scaler
model = None
scaler = None
model_version = None


class PredictionRequest(BaseModel):
    """Input features for diabetes progression prediction."""

    age: float = Field(..., description="Age (standardized)")
    sex: float = Field(..., description="Sex (standardized)")
    bmi: float = Field(..., description="Body mass index (standardized)")
    bp: float = Field(..., description="Average blood pressure (standardized)")
    s1: float = Field(..., description="Total serum cholesterol (standardized)")
    s2: float = Field(..., description="Low-density lipoproteins (standardized)")
    s3: float = Field(..., description="High-density lipoproteins (standardized)")
    s4: float = Field(..., description="Total cholesterol / HDL (standardized)")
    s5: float = Field(..., description="Log of serum triglycerides (standardized)")
    s6: float = Field(..., description="Blood sugar level (standardized)")

    @validator("*")
    def check_reasonable_range(cls, v):
        """Validate that input values are in a reasonable range."""
        if not -10 <= v <= 10:
            raise ValueError(f"Value {v} is outside reasonable range [-10, 10]")
        return v


class PredictionResponse(BaseModel):
    """Response containing the progression prediction."""

    prediction: float = Field(..., description="Predicted progression index (higher = worse)")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_version: str = Field(..., description="Current model version")


def load_model_artifacts(model_dir: str = "models"):
    """Load model, scaler, and metadata."""
    global model, scaler, model_version

    model_path = Path(model_dir) / "model.pkl"
    scaler_path = Path(model_dir) / "scaler.pkl"
    metrics_path = Path(model_dir) / "metrics.json"

    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)

        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                model_version = metrics.get("model_version", "unknown")
        else:
            model_version = "unknown"

        logger.info(f"Successfully loaded model version {model_version}")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model_artifacts()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Health status and model version
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {"status": "ok", "model_version": model_version}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> Dict[str, float]:
    """
    Predict diabetes progression for a patient.

    Args:
        request: Patient features

    Returns:
        Predicted progression index (higher values indicate greater risk)
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to numpy array in correct order
        features = np.array(
            [
                [
                    request.age,
                    request.sex,
                    request.bmi,
                    request.bp,
                    request.s1,
                    request.s2,
                    request.s3,
                    request.s4,
                    request.s5,
                    request.s6,
                ]
            ]
        )

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = float(model.predict(features_scaled)[0])

        logger.info(f"Prediction: {prediction:.2f}")

        return {"prediction": prediction}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Diabetes Progression Prediction API",
        "version": model_version,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

