"""
Compare metrics between model versions.
"""

import json
import sys
from pathlib import Path


def load_metrics(version_dir: str):
    """Load metrics from a model directory."""
    metrics_path = Path(version_dir) / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def compare_metrics(v1_metrics, v2_metrics):
    """Compare metrics between two versions."""
    if not v1_metrics or not v2_metrics:
        print("Error: Could not load metrics for one or both versions")
        return

    print("=" * 70)
    print("MODEL COMPARISON: v0.1.0 → v0.2.0")
    print("=" * 70)

    # RMSE comparison
    v1_rmse = v1_metrics["test_rmse"]
    v2_rmse = v2_metrics["test_rmse"]
    rmse_improvement = ((v1_rmse - v2_rmse) / v1_rmse) * 100

    print("\nTest RMSE:")
    print(f"  v0.1.0: {v1_rmse:.2f}")
    print(f"  v0.2.0: {v2_rmse:.2f}")
    print(f"  Change: {rmse_improvement:+.2f}% {'✓' if rmse_improvement > 0 else '✗'}")

    # R² comparison
    v1_r2 = v1_metrics["test_r2"]
    v2_r2 = v2_metrics["test_r2"]
    r2_improvement = ((v2_r2 - v1_r2) / v1_r2) * 100

    print("\nTest R²:")
    print(f"  v0.1.0: {v1_r2:.4f}")
    print(f"  v0.2.0: {v2_r2:.4f}")
    print(f"  Change: {r2_improvement:+.2f}% {'✓' if r2_improvement > 0 else '✗'}")

    # Model details
    print("\nModel Details:")
    print(f"  v0.1.0: LinearRegression with StandardScaler")
    alpha_val = v2_metrics.get('alpha', 'N/A')
    if isinstance(alpha_val, (int, float)):
        print(f"  v0.2.0: Ridge (alpha={alpha_val:.4f}) with RobustScaler")
    else:
        print(f"  v0.2.0: Ridge (alpha={alpha_val}) with RobustScaler")

    # Additional v0.2 metrics
    if "test_mae" in v2_metrics:
        print("\nAdditional v0.2 Metrics:")
        print(f"  MAE: {v2_metrics['test_mae']:.2f}")
        print(f"  CV RMSE: {v2_metrics.get('cv_rmse_mean', 0):.2f} "
              f"(+/- {v2_metrics.get('cv_rmse_std', 0):.2f})")

    if "threshold_accuracy" in v2_metrics:
        print("\nHigh-Risk Classification (v0.2):")
        print(f"  Threshold: {v2_metrics['high_risk_threshold']:.2f}")
        print(f"  Accuracy: {v2_metrics['threshold_accuracy']:.3f}")
        print(f"  Precision: {v2_metrics['threshold_precision']:.3f}")
        print(f"  Recall: {v2_metrics['threshold_recall']:.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if rmse_improvement > 0 or r2_improvement > 0:
        print("✓ v0.2.0 shows improvement over v0.1.0")
        if rmse_improvement > 0:
            print(f"  - Lower RMSE (better): {rmse_improvement:.2f}% improvement")
        if r2_improvement > 0:
            print(f"  - Higher R² (better): {r2_improvement:.2f}% improvement")
    else:
        print("✗ v0.2.0 shows no clear improvement")

    print("\nKey Improvements in v0.2.0:")
    print("  - Ridge regression with L2 regularization")
    print("  - Cross-validation for hyperparameter tuning")
    print("  - RobustScaler for better outlier handling")
    print("  - Feature importance analysis")
    print("  - High-risk threshold classification")
    print("=" * 70)


if __name__ == "__main__":
    v1_dir = sys.argv[1] if len(sys.argv) > 1 else "models_v1"
    v2_dir = sys.argv[2] if len(sys.argv) > 2 else "models"

    print(f"Loading v0.1.0 metrics from: {v1_dir}")
    print(f"Loading v0.2.0 metrics from: {v2_dir}\n")

    v1_metrics = load_metrics(v1_dir)
    v2_metrics = load_metrics(v2_dir)

    compare_metrics(v1_metrics, v2_metrics)

