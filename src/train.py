"""
Training script for diabetes progression prediction model

Version 0.1: Baseline with LinearRegression and StandardScaler
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_data():
    """Load the diabetes dataset"""
    print("Loading diabetes dataset...")
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame.drop(columns=["target"])
    y = diabetes.frame["target"]
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def train_model(X, y, output_dir="models"):
    """Train the baseline model and save artifacts"""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Preprocessing: StandardScaler
    print("\nScaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model: LinearRegression
    print("\nTraining LinearRegression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predictions
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    metrics = {
        "model_type": "LinearRegression",
        "scaler_type": "StandardScaler",
        "random_seed": RANDOM_SEED,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "rmse": float(test_rmse),
        "r2_score": float(test_r2),
        "mae": float(test_mae),
    }

    print("\n" + "=" * 70)
    print("TRAINING RESULTS - Baseline Model (v0.1)")
    print("=" * 70)
    print(f"Model: {metrics['model_type']}")
    print(f"Scaler: {metrics['scaler_type']}")
    print(f"Random Seed: {metrics['random_seed']}")
    print(f"\nDataset:")
    print(f"  Train samples: {metrics['train_samples']}")
    print(f"  Test samples: {metrics['test_samples']}")
    print(f"  Features: {metrics['n_features']}")
    print(f"\nPerformance Metrics:")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Train MAE: {train_mae:.2f}")
    print(f"  Test MAE: {test_mae:.2f}")
    print("=" * 70)

    # Save artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "model.pkl"
    scaler_path = output_path / "scaler.pkl"
    metrics_path = output_path / "metrics.json"

    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)

    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)

    print(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Training complete! All artifacts saved.")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train diabetes progression prediction model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model artifacts",
    )
    args = parser.parse_args()

    X, y = load_data()
    metrics = train_model(X, y, output_dir=args.output_dir)

    return metrics


if __name__ == "__main__":
    main()
