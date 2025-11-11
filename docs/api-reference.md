# API Reference

The radiolens REST API is built with FastAPI and exposed at
`http://localhost:8000` by default. Interactive Swagger UI is available at
`/docs` and ReDoc at `/redoc`.

---

## Base URL

```
http://<host>:<port>/api/v1
```

Default: `http://localhost:8000/api/v1`

---

## Endpoints

### `POST /classify`

Accept a chest X-ray image file and return a pneumonia classification.

**Accepted file types:** JPEG (`.jpg`, `.jpeg`), PNG (`.png`), DICOM (`.dcm`)

**Request**

| Field | Type | Description |
|-------|------|-------------|
| `file` | `multipart/form-data` | Image file to classify |

**Response — 200 OK**

```json
{
  "label": "PNEUMONIA",
  "probability": 0.9312,
  "confidence": 0.9312,
  "certainty_tier": "HIGH",
  "clinical_note": "RESEARCH USE ONLY. ...",
  "model_version": "0.1.0"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | `"PNEUMONIA"` or `"NORMAL"` |
| `probability` | float | Raw sigmoid output in `[0, 1]` |
| `confidence` | float | `max(probability, 1 - probability)` |
| `certainty_tier` | string | `"HIGH"` ≥ 0.85 · `"MODERATE"` ≥ 0.70 · `"LOW"` otherwise |
| `clinical_note` | string | Mandatory research-only disclaimer |
| `model_version` | string | Semantic version of the deployed model |

**Error responses**

| Code | Cause |
|------|-------|
| `400` | Unsupported file extension |
| `422` | File cannot be decoded as an image |

---

### `GET /health`

Check whether the service is up and the model is loaded.

**Response — 200 OK**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 142.5,
  "version": "0.1.0"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"healthy"` when model is ready; `"degraded"` otherwise |
| `model_loaded` | bool | `true` when the classifier singleton is initialised |
| `uptime_seconds` | float | Seconds elapsed since application start |
| `version` | string | Application version |

---

### `GET /info`

Return model architecture and validation metadata.

**Response — 200 OK**

```json
{
  "backbone": "MobileNetV2",
  "input_shape": [224, 224, 3],
  "output_classes": ["NORMAL", "PNEUMONIA"],
  "training_dataset": "Kaggle Chest X-Ray Images (Pneumonia) — Guangzhou Women and Children's Medical Centre",
  "validation_approach": "Stratified 70/20/10 internal split + cross-operator validation on an independent dataset",
  "cross_operator_accuracy": 0.86,
  "cross_operator_sensitivity": 0.964,
  "cross_operator_auc": 0.964
}
```

---

### `GET /performance`

Return all published cross-operator validation metrics.

**Response — 200 OK**

```json
{
  "internal_accuracy": 0.948,
  "internal_sensitivity": 0.896,
  "internal_specificity": 1.0,
  "internal_roc_auc": 0.988,
  "external_accuracy": 0.86,
  "external_sensitivity": 0.964,
  "external_specificity": 0.748,
  "external_roc_auc": 0.964,
  "bootstrap_p_value": 0.978,
  "bootstrap_ci_lower": -0.0115,
  "bootstrap_ci_upper": 0.0099,
  "n_external_samples": 107
}
```

---

## Client Examples

### curl

```bash
# Classify an image
curl -X POST http://localhost:8000/api/v1/classify \
     -F "file=@patient_xray.jpg" \
     | python -m json.tool

# Check health
curl http://localhost:8000/api/v1/health

# Get performance stats
curl http://localhost:8000/api/v1/performance | python -m json.tool
```

### Python (requests)

```python
import requests

BASE = "http://localhost:8000/api/v1"

# Classify
with open("patient_xray.jpg", "rb") as f:
    response = requests.post(f"{BASE}/classify", files={"file": f})
result = response.json()
print(result["label"], result["certainty_tier"])

# Health check
print(requests.get(f"{BASE}/health").json())
```

### Python (httpx — async)

```python
import asyncio
import httpx

async def classify(image_path: str) -> dict:
    async with httpx.AsyncClient() as client:
        with open(image_path, "rb") as f:
            resp = await client.post(
                "http://localhost:8000/api/v1/classify",
                files={"file": f},
            )
        resp.raise_for_status()
        return resp.json()

result = asyncio.run(classify("patient_xray.jpg"))
print(result)
```

---

## Starting the Server

```bash
# Development
uvicorn radiolens.api.server:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn radiolens.api.server:app --host 0.0.0.0 --port 8000 --workers 2

# Via CLI entry point
radiolens-serve
```

Set the model path before starting:

```bash
export RADIOLENS_MODEL_WEIGHTS_PATH=model/best_weights.keras
```
