# Deployment

This document covers running radiolens in production using Docker Compose,
and deploying to cloud platforms.

---

## Prerequisites

- Docker 24+ and Docker Compose v2
- A trained model file at `model/best_weights.keras` (see
  [Training Walkthrough](../examples/training-walkthrough/))
- (Optional) a `.env` file with overrides

---

## Docker Compose (Recommended)

The `docker/` directory provides two optimised images:

| Service | Image | Port |
|---------|-------|------|
| `api` | `Dockerfile.api` | `8000` |
| `dashboard` | `Dockerfile.dashboard` | `8501` |

### Start both services

```bash
# From the radiolens/ root
cp .env.example .env
# Edit .env — at minimum set RADIOLENS_MODEL_WEIGHTS_PATH

docker compose -f docker/docker-compose.yml up --build
```

API:       http://localhost:8000/docs
Dashboard: http://localhost:8501

### Start only the API

```bash
docker compose -f docker/docker-compose.yml up --build api
```

### Environment Variables in Docker

Pass variables via the `.env` file or inline:

```bash
RADIOLENS_MODEL_WEIGHTS_PATH=model/best_weights.keras \
docker compose -f docker/docker-compose.yml up api
```

---

## Manual (without Docker)

### Install

```bash
pip install "radiolens[api,dashboard]"
```

### API server

```bash
export RADIOLENS_MODEL_WEIGHTS_PATH=model/best_weights.keras
uvicorn radiolens.api.server:app --host 0.0.0.0 --port 8000
```

Or via the CLI entry point:

```bash
radiolens-serve
```

### Streamlit dashboard

```bash
export RADIOLENS_MODEL_WEIGHTS_PATH=model/best_weights.keras
streamlit run app/dashboard.py --server.port 8501
```

---

## Cloud Platforms

### Render

1. Connect your repository.
2. Set **Start Command** to `radiolens-serve`.
3. Add environment variable `RADIOLENS_MODEL_WEIGHTS_PATH`.
4. Render injects `PORT` automatically; radiolens reads it via the
   `AliasChoices("RADIOLENS_API_PORT", "PORT")` alias.

### Streamlit Cloud

1. Point at `app/dashboard.py`.
2. Add secrets in the Streamlit Cloud UI matching the `RADIOLENS_*` variables.

### Railway / Fly.io

Use the provided `Dockerfile.api` as the build target. Both platforms
inject `PORT`; no additional configuration is required.

---

## Health Checks

The `/api/v1/health` endpoint is suitable as a container health check:

```yaml
# docker-compose.yml snippet
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

---

## Model Weight Distribution

The model weights file is **not** included in the repository (it is
`.gitignore`d). Options for providing it at runtime:

| Method | Notes |
|--------|-------|
| Hugging Face Hub | `huggingface-cli download ayushirathour/chest-xray-pneumonia-detection best_chest_xray_model.h5` |
| Mounted volume | Pass `-v /path/to/model:/app/model` in `docker run` |
| Object storage | Download in `docker-entrypoint.sh` from S3 / GCS at startup |
| Baked into image | Copy into a custom `Dockerfile` layer (increases image size) |

---

## Security Considerations

- The default `RADIOLENS_CORS_ALLOW_ORIGINS=["*"]` is permissive. Restrict
  it in production to your front-end origin.
- `RADIOLENS_MAX_UPLOAD_BYTES` (default 10 MB) limits upload size; adjust
  for DICOM files which can be larger.
- Run containers as a non-root user. The provided Dockerfiles use a
  `appuser` non-root user.
