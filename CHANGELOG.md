# Changelog

All notable changes to radiolens are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

*No unreleased changes at this time.*

---

## [0.1.0] - 2025-xx-xx

### Added

#### Core Classifier
- `ThoraxClassifier`: MobileNetV2-based binary classifier for chest X-rays.
  Wraps a Keras model with a clean `build()` / `load_weights()` /
  `run_inference()` interface. Returns typed `InferenceResult` objects rather
  than raw arrays.
- `ImageNormalizer` (`core/preprocessor.py`): handles JPEG, PNG, and DICOM
  input formats. Resizes to the configured `IMAGE_HEIGHT × IMAGE_WIDTH`,
  normalises pixel values to [0, 1], and adds the batch dimension expected
  by TensorFlow.
- `InferenceResult` dataclass: typed container for `label`, `probability`,
  `confidence`, and `certainty_tier` fields. Certainty tiers are `HIGH`
  (≥ 0.85), `MODERATE` (≥ 0.65), and `LOW` (< 0.65).

#### Data Pipeline
- `RadiographBalancer`: class-equalising dataset preparation. Undersamples
  the majority class to achieve a 1:1 NORMAL:PNEUMONIA ratio, then creates
  stratified train / validation / test splits using configurable fractions.
  Writes balanced splits to a structured output directory.
- `build_training_flow()` and `build_validation_flow()`: factory functions
  that return augmented and non-augmented `ImageDataGenerator` flows
  respectively. Augmentation parameters (rotation range, shift, zoom, flip,
  brightness) are driven by `Settings`.

#### Training
- `DetectorTrainer`: encapsulates the Keras `model.fit()` call with
  `EarlyStopping` (on validation loss), `ReduceLROnPlateau`, and
  `ModelCheckpoint` (saves best validation loss checkpoint). Returns a
  `TrainingOutcome` dataclass summarising epochs run, best validation loss,
  and the path to the saved checkpoint.

#### Evaluation
- `ClinicalMetricsReport`: dataclass holding 15+ clinical performance
  metrics: accuracy, sensitivity (recall), specificity, precision (PPV),
  NPV, F1 score, ROC-AUC, PR-AUC, Matthews correlation coefficient (MCC),
  Cohen's Kappa, and decision-threshold-specific counts (TP, TN, FP, FN).
- `compute_binary_metrics()`: computes all metrics in `ClinicalMetricsReport`
  from `y_true` and `y_pred_proba` arrays. Threshold defaults to 0.5 but is
  configurable. Returns a `ClinicalMetricsReport` instance.
- `DiagnosticVisualizer`: generates 8 publication-quality diagnostic plots:
  ROC curve, precision-recall curve, confusion matrix, probability
  distribution histogram, calibration (reliability) diagram, threshold sweep,
  bootstrap accuracy distribution, and a per-class sample grid.
- `BootstrapValidator`: statistical generalisation testing. Performs n
  resamples (default 1 000) on two accuracy arrays and computes the
  two-tailed p-value and 95% confidence interval for the accuracy gap.
  Produces the bootstrap accuracy distribution plot via `DiagnosticVisualizer`.

#### REST API
- FastAPI application (`api/server.py`) with lifespan-managed singleton model
  loading. All endpoints are versioned under `/api/v1/`.
- `POST /api/v1/classify`: accepts multipart/form-data with a `file` field
  (JPEG, PNG, or DICOM). Returns `ClassificationResponse` JSON.
- `GET /api/v1/health`: liveness and readiness check returning `HealthStatus`.
- `GET /api/v1/info`: model metadata (`ModelInfo`) including version,
  backbone, input shape, and output classes.
- `GET /api/v1/performance`: pre-computed performance statistics
  (`PerformanceStats`) for both internal and cross-operator test sets.
- Interactive Swagger UI at `/docs`; ReDoc at `/redoc`.
- Pydantic v2 request and response contracts in `api/contracts.py`.
- Dependency injection wiring in `api/dependencies.py` and `api/providers.py`.

#### Streamlit Dashboard
- Browser-based diagnostic interface (`app/dashboard.py`). Supports JPEG,
  PNG, and DICOM uploads. Displays the uploaded image alongside the
  prediction label, probability bar, certainty tier, and model information.
- PDF report generation: single-page clinical summary containing the image,
  prediction, confidence score, certainty tier, and key performance
  statistics, downloadable from the sidebar.
- Streamlit `st.cache_resource` for zero-latency repeat predictions after the
  first model load.

#### Configuration
- `Settings` (Pydantic BaseSettings, `config.py`): complete configuration
  class with `RADIOLENS_` env prefix. Covers model architecture, training
  hyperparameters, data augmentation, dataset splitting fractions, API server,
  and bootstrap testing parameters.
- `get_settings()`: cached singleton accessor. Returns the same `Settings`
  instance throughout the process lifetime.
- `.env.example`: template environment file documenting every supported
  variable with its default and a brief description.

#### Infrastructure
- Structured JSON logging via `structlog`. All library code uses
  `structlog.get_logger(__name__)`. Log level is configurable via
  `RADIOLENS_LOG_LEVEL`.
- `docker/Dockerfile.api`: multi-stage build for the FastAPI service.
  Final image based on `python:3.11-slim`, non-root user, health check
  on `GET /api/v1/health`.
- `docker/Dockerfile.dashboard`: Streamlit dashboard image with matching
  conventions.
- `docker-compose.yml`: orchestrates both services with shared model volume,
  environment variable passthrough, and `depends_on` health check gate.
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`) with three jobs:
  `lint` (ruff check + format + mypy), `test-unit` (pytest unit suite,
  coverage gate 80%), and `test-integration` (slow tests with model stub).
- `Makefile` with targets: `lint`, `format`, `typecheck`, `test-fast`,
  `test-slow`, `test`, `serve`, `dashboard`, `docker-up`, `docker-down`,
  `clean`.
- `pyproject.toml` with `[project.optional-dependencies]` groups: `api`,
  `dashboard`, `training`, `dev`.
- 80%+ unit test coverage across `core`, `data`, `evaluation`, `training`,
  and `api` subpackages.

### Changed

*Initial release — no prior version to compare against.*

### Deprecated

*Nothing deprecated in the initial release.*

### Removed

*Nothing removed in the initial release.*

### Fixed

*No bug fixes in the initial release.*

### Security

*No security advisories for the initial release.*

---

[Unreleased]: https://github.com/your-org/radiolens/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/radiolens/releases/tag/v0.1.0
