# ============================================================================
# Genova – Makefile
# ============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Variables ───────────────────────────────────────────────────────────────
PYTHON      := python
POETRY      := poetry
DOCKER      := docker
COMPOSE     := docker compose
IMAGE_NAME  := genova
IMAGE_TAG   := latest
CONFIG      := configs/train.yaml

# ── Help ────────────────────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mGenova Makefile targets:\033[0m\n\n"} \
		/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# ── Installation ────────────────────────────────────────────────────────────
.PHONY: install
install: ## Install all dependencies (including dev)
	$(POETRY) install --no-interaction
	$(POETRY) run pre-commit install

.PHONY: install-prod
install-prod: ## Install production dependencies only
	$(POETRY) install --only main --no-interaction

# ── Code quality ────────────────────────────────────────────────────────────
.PHONY: lint
lint: ## Run all linters (black check, ruff, isort check)
	$(POETRY) run black --check --diff genova/ tests/
	$(POETRY) run ruff check genova/ tests/
	$(POETRY) run isort --check-only --diff genova/ tests/

.PHONY: format
format: ## Auto-format code (black, ruff fix, isort)
	$(POETRY) run isort genova/ tests/
	$(POETRY) run black genova/ tests/
	$(POETRY) run ruff check --fix genova/ tests/

# ── Testing ─────────────────────────────────────────────────────────────────
.PHONY: test
test: ## Run test suite (excludes slow and GPU tests)
	$(POETRY) run pytest -m "not slow and not gpu"

.PHONY: test-all
test-all: ## Run full test suite including slow tests
	$(POETRY) run pytest

.PHONY: test-cov
test-cov: ## Run tests with HTML coverage report
	$(POETRY) run pytest -m "not slow and not gpu" --cov-report=html:htmlcov
	@echo "Coverage report: htmlcov/index.html"

# ── Docker ──────────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build: ## Build the Docker image
	$(DOCKER) build \
		-f deploy/docker/Dockerfile \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		.

.PHONY: docker-up
docker-up: ## Start all services via docker compose
	$(COMPOSE) up -d

.PHONY: docker-down
docker-down: ## Stop all services
	$(COMPOSE) down

.PHONY: docker-logs
docker-logs: ## Tail logs from all services
	$(COMPOSE) logs -f

# ── Training ────────────────────────────────────────────────────────────────
.PHONY: train
train: ## Launch training run (set CONFIG=path/to/config.yaml to override)
	$(POETRY) run python -m genova.cli.main train --config $(CONFIG)

# ── Serving ─────────────────────────────────────────────────────────────────
.PHONY: serve
serve: ## Start the API server locally
	$(POETRY) run uvicorn genova.api.app:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--reload-dir genova

# ── CI/CD ──────────────────────────────────────────────────────────────────
.PHONY: ci
ci: lint test build ## Run full CI pipeline locally (lint + test + build)

.PHONY: build
build: ## Build Python package (wheel + sdist)
	$(POETRY) build
	@echo "Artifacts in dist/"

.PHONY: typecheck
typecheck: ## Run mypy type checking
	$(POETRY) run mypy genova/ --ignore-missing-imports --no-error-summary || true

.PHONY: security-scan
security-scan: ## Run security scans (bandit + pip-audit)
	$(POETRY) run pip-audit 2>/dev/null || echo "pip-audit not installed, skipping"
	$(POETRY) run bandit -r genova/ -c pyproject.toml 2>/dev/null || echo "bandit not installed, skipping"

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	$(PYTHON) scripts/ci/run_benchmarks.py --output benchmark_results.json

.PHONY: smoke-test
smoke-test: ## Run smoke tests against deployed service (set URL=http://...)
	bash scripts/ci/run_smoke_tests.sh $${URL:-http://localhost:8000}

.PHONY: load-test
load-test: ## Run load tests against API (set URL=http://...)
	$(PYTHON) scripts/ci/load_test.py --host $${URL:-http://localhost:8000} --users 10 --duration 60

.PHONY: check-export
check-export: ## Verify model export (checkpoint + TorchScript + ONNX)
	$(PYTHON) scripts/ci/check_model_export.py

.PHONY: docker-scan
docker-scan: docker-build ## Scan Docker image for vulnerabilities
	$(DOCKER) run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy:latest image --severity HIGH,CRITICAL $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: docker-push
docker-push: ## Push Docker image to registry (set REGISTRY=ghcr.io/org)
	$(DOCKER) tag $(IMAGE_NAME):$(IMAGE_TAG) $${REGISTRY:-ghcr.io/genova-genomics}/$(IMAGE_NAME):$(IMAGE_TAG)
	$(DOCKER) push $${REGISTRY:-ghcr.io/genova-genomics}/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: helm-lint
helm-lint: ## Lint Helm chart
	helm lint deploy/helm/

.PHONY: helm-template
helm-template: ## Render Helm templates (dry-run)
	helm template genova deploy/helm/ -f deploy/helm/values.yaml

.PHONY: deploy-staging
deploy-staging: ## Deploy to staging (set KUBECONFIG)
	helm upgrade --install genova-staging deploy/helm/ \
		-f deploy/helm/values-staging.yaml \
		--namespace genova-staging --create-namespace \
		--wait --timeout 5m

.PHONY: deploy-production
deploy-production: ## Deploy to production (set KUBECONFIG)
	@echo "⚠ Deploying to PRODUCTION. Press Ctrl+C within 5s to cancel."
	@sleep 5
	helm upgrade --install genova deploy/helm/ \
		-f deploy/helm/values-production.yaml \
		--namespace genova --create-namespace \
		--wait --timeout 10m

.PHONY: release
release: ## Create a release tag (set VERSION=x.y.z)
	@test -n "$${VERSION}" || (echo "Usage: make release VERSION=1.0.0" && exit 1)
	git tag -a "v$${VERSION}" -m "Release v$${VERSION}"
	@echo "Tag v$${VERSION} created. Push with: git push origin v$${VERSION}"

# ── Pipeline ───────────────────────────────────────────────────────────────
.PHONY: pipeline-small
pipeline-small: ## Run full pipeline with small config (4L/256d, ~3M params)
	bash scripts/run_full_pipeline.sh --small

.PHONY: pipeline-large
pipeline-large: ## Run full pipeline with large config (12L/768d, ~85M params)
	bash scripts/run_full_pipeline.sh --large

.PHONY: pipeline-mamba
pipeline-mamba: ## Run full pipeline with Mamba config (12L SSM, ~85M params)
	bash scripts/run_full_pipeline.sh --mamba

.PHONY: data-quality
data-quality: ## Run data quality report on reference genome
	$(PYTHON) scripts/run_data_quality.py \
		--fasta data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
		--output reports/data_quality.md \
		--format markdown

.PHONY: model-card
model-card: ## Generate model card for trained model (set MODEL=path/to/model.pt)
	$(PYTHON) scripts/generate_model_card.py \
		--model-path $${MODEL:-outputs/genova_small/best_model.pt} \
		--output $${MODEL_CARD:-outputs/genova_small/MODEL_CARD.md}

.PHONY: benchmarks
benchmarks: ## Run full benchmark suite (set MODEL=path/to/model.pt)
	$(PYTHON) scripts/run_benchmarks_suite.py \
		--model-path $${MODEL:-outputs/genova_small/best_model.pt} \
		--output-dir results/benchmarks \
		--tasks all

# ── Cleanup ─────────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ coverage.xml junit.xml benchmark_results.json
