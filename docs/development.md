# Development Guide

---

## Prerequisites

- Python 3.10 or later
- Git
- (Optional) Docker 24+ for containerised development

---

## Setup

```bash
git clone https://github.com/your-org/radiolens.git
cd radiolens

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -e ".[dev,api,dashboard]"
pre-commit install
```

The editable install (`-e`) means changes to `src/radiolens/` take effect
immediately without reinstalling.

---

## Running Tests

```bash
# Fast unit tests only (< 30 s, no GPU required)
make test-fast

# Full suite with coverage report
make test

# Single module
pytest tests/unit/test_clinical_metrics.py -v

# All unit tests, stop on first failure
pytest tests/unit/ -x -v

# Integration tests (requires model weights)
pytest tests/integration/ -v -m integration
```

Coverage gate is 65% on fast tests (`-m "not slow"`). TF-dependent tests
(detector unit tests, API integration) skip locally without TensorFlow but
run in CI where `pip install -e ".[api,dev]"` installs it.
`visualizer.py` and `dicom_reader.py` are excluded from measurement.

### Test Markers

| Marker | Description |
|--------|-------------|
| `slow` | Tests requiring GPU or large datasets — skip with `-m "not slow"` |
| `integration` | End-to-end tests that spin up the API server |

---

## Linting and Type Checking

```bash
# Lint and auto-fix
make lint

# Type check
make typecheck

# Run both
make check
```

Pre-commit runs ruff and mypy automatically on every `git commit`. To run
manually:

```bash
pre-commit run --all-files
```

---

## Project Structure

```
radiolens/
├── src/radiolens/          # Package source (importable as `radiolens`)
│   ├── config.py
│   ├── api/
│   ├── core/
│   ├── data/
│   ├── evaluation/
│   └── training/
├── app/
│   └── dashboard.py        # Streamlit UI
├── docker/                 # Dockerfiles and docker-compose
├── docs/                   # This documentation
├── examples/               # Runnable example scripts
├── model/                  # .gitignored — place weights here
├── scripts/                # CLI entry points
├── tests/
│   ├── conftest.py         # Shared fixtures
│   ├── unit/               # Pure unit tests
│   └── integration/        # API and training integration tests
├── .env.example
├── .pre-commit-config.yaml
├── Makefile
└── pyproject.toml
```

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make install` | Editable install with all extras |
| `make test` | Full pytest suite with coverage |
| `make test-fast` | Unit tests only (`-m "not slow"`) |
| `make lint` | `ruff check --fix` |
| `make typecheck` | `mypy src/` |
| `make check` | lint + typecheck |
| `make serve` | Start the API server (requires `.env`) |
| `make dashboard` | Start Streamlit (requires `.env`) |
| `make docker-up` | `docker compose up --build` |
| `make docker-down` | `docker compose down` |

---

## Adding a New Endpoint

1. Add request/response models to `src/radiolens/api/contracts.py`.
2. Add the route handler to `src/radiolens/api/endpoints.py`.
3. Wire any new dependencies in `src/radiolens/api/providers.py`.
4. Write integration tests in `tests/integration/test_inference_api.py`.

---

## Coding Standards

- Line length: 79 characters (ruff enforced)
- Type annotations required on all public functions and methods
- Docstrings: Google-style, all public API surface
- Structured logging via `structlog` — no bare `print()` in library code
- No bare `except` clauses; catch specific exception types

---

## Release Process

1. Update `CHANGELOG.md` with the new version section.
2. Bump the version in `pyproject.toml`.
3. Create and push a tag: `git tag v0.2.0 && git push --tags`.
4. The `release.yml` GitHub Actions workflow builds and publishes to PyPI.
