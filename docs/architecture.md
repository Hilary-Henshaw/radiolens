# Architecture

This document describes the design decisions and module boundaries in radiolens.

---

## Module Map

```
src/radiolens/
├── config.py               # Settings singleton (Pydantic + env vars)
├── api/
│   ├── contracts.py        # Pydantic request / response models
│   ├── endpoints.py        # Route handlers (classify, health, info, performance)
│   ├── guards.py           # Request validation helpers
│   ├── providers.py        # FastAPI dependency injection & lifespan
│   └── server.py           # FastAPI app factory
├── core/
│   ├── detector.py         # ThoraxClassifier — build, load, infer
│   └── preprocessor.py     # ImageNormalizer — PIL/bytes/path → float32 array
├── data/
│   ├── augmentor.py        # Keras ImageDataGenerator augmentation wrappers
│   ├── balancer.py         # RadiographBalancer — equalise & split raw data
│   └── dicom_reader.py     # pydicom → PIL conversion
├── evaluation/
│   ├── metrics.py          # compute_binary_metrics(), ClinicalMetricsReport
│   ├── significance.py     # BootstrapValidator, BootstrapResult
│   └── visualizer.py       # DiagnosticVisualizer — ROC, confusion matrix
└── training/
    ├── callbacks.py        # Custom Keras callbacks
    └── runner.py           # DetectorTrainer
```

---

## Inference Pipeline

```
Raw file (JPEG / PNG / DICOM)
        │
        ▼
  ImageNormalizer
  • convert to RGB
  • resize to 224 × 224 (BILINEAR)
  • scale to [0, 1] float32
        │
        ▼
  ThoraxClassifier.run_inference(pixel_array)
  ┌──────────────────────────────────────────┐
  │  MobileNetV2 backbone                    │
  │  (ImageNet weights, top removed)         │
  │           ↓                              │
  │  GlobalAveragePooling2D                  │
  │           ↓                              │
  │  Dropout(0.3)                            │
  │           ↓                              │
  │  Dense(128, relu)                        │
  │           ↓                              │
  │  Dropout(0.2)                            │
  │           ↓                              │
  │  Dense(1, sigmoid)  → p(pneumonia)       │
  └──────────────────────────────────────────┘
        │
        ▼
  InferenceResult
  • label:         "PNEUMONIA" | "NORMAL"
  • probability:   raw sigmoid value
  • confidence:    max(p, 1-p)
  • certainty_tier: "HIGH" | "MODERATE" | "LOW"
```

### Certainty Tiers

| Tier | Confidence threshold |
|------|---------------------|
| HIGH | ≥ 0.85 |
| MODERATE | ≥ 0.70 |
| LOW | < 0.70 |

---

## Training Pipeline

```
Raw dataset (class folders: normal/, pneumonia/)
        │
        ▼
  RadiographBalancer.equalize_and_split()
  • undersample majority class to match minority count
  • stratified 70 / 20 / 10 split
  • copies files to output_dir/{train,val,test}/{normal,pneumonia}/
        │
        ▼
  build_training_flow()   build_validation_flow()
  (augmented generators)  (no augmentation)
        │
        ▼
  DetectorTrainer
  • EarlyStopping (patience=7, monitor=val_loss)
  • ReduceLROnPlateau (factor=0.5, patience=4)
  • ModelCheckpoint (saves best val_loss checkpoint)
        │
        ▼
  best_weights.keras
        │
        ▼
  compute_binary_metrics() → ClinicalMetricsReport
  DiagnosticVisualizer    → ROC curve, confusion matrix, calibration plot
```

---

## Deployment Surfaces

```
┌──────────────────────────┐   ┌───────────────────────────┐
│   FastAPI REST API       │   │   Streamlit Dashboard      │
│   :8000                  │   │   :8501                    │
│                          │   │                            │
│  POST /api/v1/classify   │   │  Upload JPEG / DICOM       │
│  GET  /api/v1/health     │   │  View prediction + plots   │
│  GET  /api/v1/info       │   │  Download PDF report       │
│  GET  /api/v1/performance│   │                            │
└──────────┬───────────────┘   └──────────────┬────────────┘
           │                                   │
           └──────────┬────────────────────────┘
                      │
              ThoraxClassifier
              (singleton loaded at startup via lifespan)
```

Both surfaces share the same `ThoraxClassifier` instance. The API loads
the classifier in the FastAPI `lifespan` context manager
(`api/providers.py`) so the model is ready before any request arrives.

---

## Design Principles

### Stateless inference
Each `/classify` request is fully independent. No session state is stored
server-side. The classifier singleton is read-only after startup.

### Pydantic settings
All configuration is declared in a single `Settings` class. There are no
scattered `os.getenv()` calls. Validation (e.g. split fractions summing to
1.0) runs at import time.

### Dependency injection
FastAPI dependencies (`provide_classifier`, `provide_settings`,
`_provide_normalizer`) keep the route handlers thin and testable without
mocking global state.

### Pure NumPy metrics implementation
`compute_binary_metrics()` and `_compute_roc_auc()` are implemented in
pure NumPy. scikit-learn is listed as a base dependency (used elsewhere in
the evaluation workflow), but the core metric functions have no sklearn
calls, keeping them auditable and free of external AUC implementations.

### DICOM handled at the boundary
DICOM decoding is isolated in `data/dicom_reader.py` and called only in
the API endpoint. The core `ImageNormalizer` always operates on PIL
Images, keeping the preprocessing path clean.
