.PHONY: help install install-dev lint typecheck test test-fast \
        train evaluate serve clean docker-build docker-up

PYTHON ?= python
SRC     = src/radiolens

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and API/dashboard extras
	pip install -e ".[api,dashboard]"

install-dev:  ## Install all extras including dev tools
	pip install -e ".[api,dashboard,dev]"
	pre-commit install

lint:  ## Run ruff linter
	ruff check $(SRC) tests scripts app
	ruff format --check $(SRC) tests scripts app

typecheck:  ## Run mypy type checker
	mypy $(SRC)

test:  ## Run full test suite with coverage
	pytest tests/

test-fast:  ## Run tests excluding slow model tests
	pytest tests/ -m "not slow"

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v

test-integration:  ## Run integration tests only
	pytest tests/integration/ -v

format:  ## Auto-format with ruff
	ruff format $(SRC) tests scripts app
	ruff check --fix $(SRC) tests scripts app

prepare:  ## Balance dataset (usage: make prepare SOURCE=./raw OUTPUT=./data)
	$(PYTHON) scripts/prepare_dataset.py \
		--source-dir $(SOURCE) --output-dir $(OUTPUT)

train:  ## Train the classifier (usage: make train DATA=./data)
	$(PYTHON) scripts/run_training.py \
		--data-dir $(DATA) --checkpoint-dir ./model

evaluate:  ## Evaluate trained model (usage: make evaluate MODEL=./model/best.keras TEST=./data/test)
	$(PYTHON) scripts/run_evaluation.py \
		--model-path $(MODEL) --test-dir $(TEST) --output-dir ./results

serve:  ## Start the API server locally
	$(PYTHON) -m radiolens.api.server

dashboard:  ## Launch Streamlit dashboard
	streamlit run app/dashboard.py

docker-build:  ## Build all Docker images
	docker compose -f docker/docker-compose.yml build

docker-up:  ## Start full stack with Docker Compose
	docker compose -f docker/docker-compose.yml up

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage coverage.xml dist/ build/ .mypy_cache/ .ruff_cache/
