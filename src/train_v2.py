"""
Training script for diabetes progression prediction model.
Version 0.2: Improved with Ridge Regression, feature selection, and cross-validation
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Set random seed for reproducibility
RANDOM_SEED = 42
MODEL_VERSION = "0.2.0"


def train_model(output_dir: str = "models", test_size: float = 0.2):
    """
    Train diabetes progression prediction model with improvements.

    Args:
        output_dir: Directory to save model and scaler
        test_size: Proportion of data to use for testing

    Returns:
        Dictionary containing metrics
    """
    # Set random seeds
    np.random.seed(RANDOM_SEED)

    print(f"Loading diabetes dataset...")
    # Load data
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame.drop(columns=["target"])
    y = diabetes.frame["target"]

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Build pipeline with RobustScaler (better for outliers) and Ridge regression
    print("\nTuning Ridge alpha via cross-validation...")
    alphas = np.logspace(-2, 2, 20)
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")

    # Use RobustScaler instead of StandardScaler for better outlier handling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit Ridge with cross-validation to find best alpha
    ridge_cv.fit(X_train_scaled, y_train)
    best_alpha = ridge_cv.alpha_

    print(f"Best alpha: {best_alpha:.4f}")

    # Train final model with best alpha
    print(f"Training Ridge Regression with alpha={best_alpha:.4f}...")
    model = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
    model.fit(X_train_scaled, y_train)

    # Cross-validation scores on training set
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    cv_rmse = np.sqrt(-cv_scores)

    print(f"Cross-validation RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")

    # Feature importance (absolute coefficients)
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "coefficient": np.abs(model.coef_)}
    ).sort_values("coefficient", ascending=False)

    print("\nTop 5 most important features:")
    print(feature_importance.head().to_string(index=False))

    # Evaluate on train and test sets
    print("\nEvaluating model...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    metrics = {
        "model_version": MODEL_VERSION,
        "model_type": "Ridge",
        "alpha": float(best_alpha),
        "random_seed": RANDOM_SEED,
        "test_size": test_size,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        # RMSE metrics
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "cv_rmse_mean": float(cv_rmse.mean()),
        "cv_rmse_std": float(cv_rmse.std()),
        # R² metrics
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        # MAE metrics
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        # Feature importance
        "top_features": feature_importance.head(5).to_dict("records"),
    }

    # Calculate high-risk threshold (75th percentile of predictions)
    threshold = np.percentile(y_pred_train, 75)
    metrics["high_risk_threshold"] = float(threshold)

    # Evaluate threshold classification on test set
    y_test_high_risk = (y_test > threshold).astype(int)
    y_pred_high_risk = (y_pred_test > threshold).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score

    metrics["threshold_accuracy"] = float(accuracy_score(y_test_high_risk, y_pred_high_risk))
    metrics["threshold_precision"] = float(
        precision_score(y_test_high_risk, y_pred_high_risk, zero_division=0)
    )
    metrics["threshold_recall"] = float(
        recall_score(y_test_high_risk, y_pred_high_risk, zero_division=0)
    )

    print("\n" + "=" * 60)
    print("TRAINING RESULTS - VERSION 0.2")
    print("=" * 60)
    print(f"Model: Ridge Regression (alpha={best_alpha:.4f})")
    print(f"\nRegression Metrics:")
    print(f"  Train RMSE: {metrics['train_rmse']:.2f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
    print(f"  CV RMSE: {metrics['cv_rmse_mean']:.2f} (+/- {metrics['cv_rmse_std']:.2f})")
    print(f"  Train R²: {metrics['train_r2']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Train MAE: {metrics['train_mae']:.2f}")
    print(f"  Test MAE: {metrics['test_mae']:.2f}")
    print(f"\nHigh-Risk Threshold (75th percentile): {threshold:.2f}")
    print(f"  Accuracy: {metrics['threshold_accuracy']:.3f}")
    print(f"  Precision: {metrics['threshold_precision']:.3f}")
    print(f"  Recall: {metrics['threshold_recall']:.3f}")
    print("=" * 60 + "\n")

    # Save artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "model.pkl"
    scaler_path = output_path / "scaler.pkl"
    metrics_path = output_path / "metrics.json"

    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

    print(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)

    print(f"Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diabetes progression model v0.2")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )

    args = parser.parse_args()
    train_model(output_dir=args.output_dir, test_size=args.test_size)

