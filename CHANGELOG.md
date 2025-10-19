# Changelog

All notable changes to the Diabetes Progression Prediction Service will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-19

### Added
- Improved model with Ridge Regression and L2 regularization
- Model calibration for high-risk threshold classification (75th percentile)
- Enhanced preprocessing pipeline with RobustScaler
- Cross-validation during training (5-fold CV)
- Feature importance logging and analysis
- Mean Absolute Error (MAE) metrics

### Changed
- Upgraded from Linear Regression to Ridge Regression with tuned alpha (14.38)
- Improved feature scaling with RobustScaler for better outlier handling
- Enhanced model evaluation with cross-validation and additional metrics
- Updated Docker image with optimized model

### Performance Improvements
- **Test RMSE**: 53.50 (vs 53.85 in v0.1.0, **0.66% improvement** ✓)
- **Test R²**: 0.4598 (vs 0.4526 in v0.1.0, **1.60% improvement** ✓)
- **Test MAE**: 42.99 (new metric)
- **CV RMSE**: 55.84 ± 2.55 (cross-validation stability)
- **High-Risk Classification**: 83.1% accuracy, 77.8% precision, 56.0% recall

### Technical Details
- Ridge alpha parameter tuned via 5-fold cross-validation (best alpha: 14.38)
- Top contributing features: BMI (34.2), S5 (28.0), BP (22.9), Sex (18.8), S3 (11.2)
- High-risk threshold set at 75th percentile (189.09) for triage prioritization
- RobustScaler uses median/IQR instead of mean/std for better outlier resistance
- Enhanced model serialization with feature importance metadata

### Why These Changes?
- **Ridge Regression**: Adds L2 regularization to prevent overfitting on small dataset (442 samples), providing more stable predictions
- **RobustScaler**: Better handles outliers in patient data compared to StandardScaler, improving real-world robustness
- **Cross-Validation**: Ensures model generalizes well across data splits (CV RMSE: 55.84 ± 2.55)
- **Calibration**: Provides interpretable "high-risk" flag for triage workflow - 77.8% precision means fewer false alarms for nurses

---

## [0.1.0] - 2025-10-18

### Added
- Initial baseline model implementation
- Linear Regression with StandardScaler preprocessing
- REST API with FastAPI framework
- Health check endpoint (`/health`)
- Prediction endpoint (`/predict`)
- Docker containerization with multi-stage build
- GitHub Actions CI/CD pipeline
- Automated testing with pytest
- Code quality checks (black, isort, flake8)
- Comprehensive documentation

### Model Details
- **Algorithm**: Linear Regression
- **Preprocessing**: StandardScaler (zero mean, unit variance)
- **Training Split**: 80% train, 20% test
- **Random Seed**: 42 (for reproducibility)

### Performance Metrics
```json
{
  "model_version": "0.1.0",
  "test_rmse": 53.85,
  "train_rmse": 53.56,
  "test_r2": 0.4526,
  "train_r2": 0.5279,
  "test_samples": 89,
  "train_samples": 353
}
```

### API Features
- **Input Validation**: Pydantic models with range checking
- **Error Handling**: Structured JSON error responses
- **Health Checks**: Container health monitoring
- **Documentation**: Auto-generated OpenAPI docs at `/docs`

### Docker Image
- **Base**: Python 3.11-slim
- **Size**: ~450 MB (optimized with multi-stage build)
- **Startup Time**: < 5 seconds
- **Model**: Baked into image for zero-config deployment

### CI/CD Pipeline
- **CI Workflow**: Lint → Test → Train → Upload Artifacts
- **Release Workflow**: Build → Smoke Test → Publish → Create Release
- **Container Registry**: GitHub Container Registry (GHCR)
- **Automation**: Triggered on PR/push (CI) and tags (release)

### Technical Decisions

**Why Linear Regression?**
- Simple, interpretable baseline
- Fast training and inference
- Establishes performance floor for future improvements
- No hyperparameters to tune (pure baseline)

**Why StandardScaler?**
- Dataset features already standardized in scikit-learn
- Maintains feature interpretability
- Required for Linear Regression to work properly with mixed scales

**Why FastAPI?**
- Modern, fast, and type-safe
- Automatic input validation with Pydantic
- Built-in OpenAPI documentation
- Async support for scalability

**Why Multi-Stage Docker Build?**
- Separates build dependencies from runtime
- Bakes model into image (no external dependencies)
- Reduces final image size by ~40%
- Ensures reproducible deployments

### Known Limitations
- No hyperparameter tuning (intentional for baseline)
- No feature engineering beyond scaling
- No model ensembling
- No confidence intervals on predictions
- Dataset is small (442 samples) - may overfit with complex models

### Next Steps
See [0.2.0] planning above for upcoming improvements.

---

## [Unreleased]

### Planned Features
- Model monitoring dashboard
- A/B testing framework
- Feature drift detection
- Model explainability (SHAP values)
- Batch prediction endpoint
- Rate limiting and authentication

---

**Version Numbering:**
- Major version: Breaking API changes
- Minor version: New features, model improvements
- Patch version: Bug fixes, documentation updates

