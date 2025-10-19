# Diabetes Progression Prediction Service

[![CI Status](https://github.com/SuviDeSilva94/MAIO-Assignment3/actions/workflows/ci.yml/badge.svg)](https://github.com/SuviDeSilva94/MAIO-Assignment3/actions/workflows/ci.yml)
[![Release Status](https://github.com/SuviDeSilva94/MAIO-Assignment3/actions/workflows/release.yml/badge.svg)](https://github.com/SuviDeSilva94/MAIO-Assignment3/actions/workflows/release.yml)

An ML service that predicts short-term diabetes progression to help hospital virtual clinics prioritize patient follow-ups.

## Overview

This service provides a REST API for predicting diabetes progression risk based on patient vitals, labs, and lifestyle features. Nurses can use the continuous risk score to prioritize which patients need immediate follow-up calls.

### Features

- 🔬 **ML Model**: Trained on scikit-learn diabetes dataset
- 🚀 **REST API**: FastAPI-based service with validation
- 🐳 **Docker**: Self-contained, production-ready container
- 🔄 **CI/CD**: Automated testing, building, and deployment via GitHub Actions
- 📊 **Observability**: Structured JSON responses and health checks
- 🔁 **Reproducibility**: Pinned dependencies and random seeds

## Quick Start

### Using Docker (Recommended)

Pull and run the latest release:

```bash
# Pull the image (v0.1.0 - Baseline)
docker pull ghcr.io/suvidesilva94/maio-assignment3:v0.1.0

# Run the container
docker run -d -p 8000:8000 --name diabetes-api ghcr.io/suvidesilva94/maio-assignment3:v0.1.0

# Or pull v0.2.0 (Improved with Ridge Regression)
docker pull ghcr.io/suvidesilva94/maio-assignment3:v0.2.0
docker run -d -p 8000:8000 --name diabetes-api ghcr.io/suvidesilva94/maio-assignment3:v0.2.0
```

The API will be available at `http://localhost:8000`.

### Local Development

1. **Clone the repository:**

```bash
git clone https://github.com/SuviDeSilva94/MAIO-Assignment3.git
cd MAIO-Assignment3
```

2. **Create virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements-dev.txt
```

4. **Train the model:**

```bash
python src/train.py
```

5. **Run the API server:**

```bash
python -m uvicorn src.app:app --reload
```

## API Usage

### Health Check

Check if the service is running and get the model version:

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_version": "0.1.0"
}
```

### Make a Prediction

Send patient features to get a progression risk score:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001
  }'
```

**Response:**
```json
{
  "prediction": 198.52
}
```

**Note:** Higher prediction values indicate greater risk of disease progression. The exact value depends on the model version:
- v0.1.0 (LinearRegression): Returns ~235.95 for the sample payload
- v0.2.0 (Ridge Regression): Returns ~198.52 for the sample payload

### Input Features

All features are standardized (mean=0, std=1):

| Field | Description |
|-------|-------------|
| `age` | Age (standardized) |
| `sex` | Sex (standardized) |
| `bmi` | Body mass index |
| `bp` | Average blood pressure |
| `s1` | Total serum cholesterol |
| `s2` | Low-density lipoproteins |
| `s3` | High-density lipoproteins |
| `s4` | Total cholesterol / HDL ratio |
| `s5` | Log of serum triglycerides |
| `s6` | Blood sugar level |

### Error Handling

The API returns structured JSON errors:

**Missing fields (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Invalid values (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "Value 100.0 is outside reasonable range [-10, 10]",
      "type": "value_error"
    }
  ]
}
```

## Model Information

### Version 0.1.0 (Baseline)

- **Algorithm**: Linear Regression with StandardScaler
- **Training**: 80/20 train-test split (353 train, 89 test samples)
- **Random Seed**: 42 (for reproducibility)

**Performance Metrics:**
- Test RMSE: 53.85
- Test R²: 0.4526
- Train RMSE: 53.56
- Train R²: 0.5279

### Version 0.2.0 (Improved)

- **Algorithm**: Ridge Regression with RobustScaler
- **Improvements**: Cross-validation tuning, feature selection, high-risk calibration
- **Alpha**: 14.38 (tuned via 5-fold CV)
- **Training**: Same 80/20 split for fair comparison

**Performance Metrics:**
- Test RMSE: 53.50 (⬇️ 0.66% vs v0.1)
- Test R²: 0.4598 (⬆️ 1.60% vs v0.1)
- CV RMSE: 55.84 ± 2.55
- Test MAE: 42.99
- High-Risk Precision: 77.8%
- High-Risk Recall: 56.0%

**What Improved:**
- Lower RMSE = more accurate predictions
- Higher R² = better model fit
- Added cross-validation for robustness
- High-risk calibration for clinical decision support

See [CHANGELOG.md](CHANGELOG.md) for complete version history and technical details.

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_app.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
```

### Building Docker Image Locally

```bash
# Build the image
docker build -t diabetes-prediction:local .

# Run the container
docker run -d -p 8000:8000 --name diabetes-local diabetes-prediction:local

# Test the container
curl http://localhost:8000/health

# Stop and remove
docker stop diabetes-local && docker rm diabetes-local
```

### Testing Both Versions Locally

To test both v0.1 (baseline) and v0.2 (improved) side-by-side:

**Test v0.1:**
```bash
# Build v0.1 (make sure train.py has LinearRegression code)
docker build -t diabetes-triage:v0.1 .

# Run on port 8001
docker run -d -p 8001:8000 --name diabetes-v01 diabetes-triage:v0.1

# Test v0.1
curl http://localhost:8001/health
# Expected: {"status":"ok","model_version":"0.1.0"}

curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,
       "s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
# Expected: {"prediction":235.95}
```

**Test v0.2:**
```bash
# Update train.py to v0.2 code (Ridge Regression)
cp src/train_v2.py src/train.py

# Build v0.2
docker build -t diabetes-triage:v0.2 .

# Run on port 8002
docker run -d -p 8002:8000 --name diabetes-v02 diabetes-triage:v0.2

# Test v0.2
curl http://localhost:8002/health
# Expected: {"status":"ok","model_version":"0.2.0"}

curl -s -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,
       "s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
# Expected: {"prediction":198.52}
```

**Compare both versions:**
```bash
# Both containers running simultaneously!
docker ps | grep diabetes

# Extract and compare metrics
docker exec diabetes-v01 cat /app/models/metrics.json
docker exec diabetes-v02 cat /app/models/metrics.json

# Cleanup when done
docker stop diabetes-v01 diabetes-v02
docker rm diabetes-v01 diabetes-v02
```

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment:

### CI Workflow (PR/Push)

Triggered on pull requests and pushes to `main`/`develop`:

1. **Lint**: Black, isort, flake8
2. **Test**: pytest with coverage
3. **Train**: Run training script
4. **Artifacts**: Upload model artifacts and metrics

### Release Workflow (Tags)

Triggered when pushing version tags (`v*`):

1. **Build**: Create Docker image
2. **Smoke Tests**: Test container endpoints
3. **Publish**: Push to GitHub Container Registry
4. **Release**: Create GitHub Release with metrics

### Creating a Release

```bash
# Commit your changes
git add .
git commit -m "Prepare v0.1.0 release"

# Create and push tag
git tag v0.1.0
git push origin main --tags
```

The release workflow will automatically:
- Build the Docker image
- Run smoke tests
- Publish to `ghcr.io/suvidesilva94/maio-assignment3:v0.1.0`
- Create a GitHub Release with metrics

## Project Structure

```
MAIO-Assignment3/
├── .github/
│   └── workflows/
│       ├── ci.yml           # CI pipeline
│       └── release.yml      # Release pipeline
├── src/                     # Source code
│   ├── app.py              # FastAPI application
│   ├── train.py            # Training script (v0.1)
│   ├── train_v2.py         # Training script (v0.2)
│   └── compare_versions.py # Model comparison tool
├── tests/                   # Test suite
│   ├── test_app.py         # API tests
│   └── test_train.py       # Training tests
├── models/                  # Model artifacts (generated)
│   ├── model.pkl
│   ├── scaler.pkl
│   └── metrics.json
├── Dockerfile              # Multi-stage Docker build
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── README.md               # This file
└── CHANGELOG.md            # Version history
```

## Reproducibility

All aspects of the pipeline are reproducible:

- **Dependencies**: Pinned in `requirements.txt`
- **Random Seeds**: Set to 42 in training script
- **Docker**: Multi-stage build with locked versions
- **CI/CD**: Deterministic GitHub Actions workflows

To reproduce locally:

```bash
# Install exact dependencies
pip install -r requirements.txt

# Train model (deterministic)
python src/train.py

# Results should match published metrics
cat models/metrics.json
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is part of an academic assignment for MAIO (ML/AI in Operations).

## Contact

**Author**: Suvi De Silva (@SuviDeSilva94)  
**Repository**: https://github.com/SuviDeSilva94/MAIO-Assignment3

For questions or issues, please open a GitHub issue.

