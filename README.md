# Genova

**A Production-Grade Genomics Foundation Model Framework**

[![CI](https://github.com/genova-team/genova/actions/workflows/ci.yml/badge.svg)](https://github.com/genova-team/genova/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/version-1.0.0-informational.svg)](CHANGELOG.md)

Genova is a modular, extensible framework for building, training, and deploying genomic foundation models. Pre-trained on the human reference genome (GRCh38) using masked language modeling, Genova learns contextual representations of DNA sequences for variant effect prediction, regulatory element classification, sequence generation, and multi-omics integration.

---

## Table of Contents

- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Training](#training)
- [Inference & Prediction](#inference--prediction)
- [API Server](#api-server)
- [Python API](#python-api)
- [Security](#security)
- [Deployment](#deployment)
- [Testing](#testing)
- [Development](#development)
- [Project Architecture](#project-architecture)
- [Benchmark Validation](#benchmark-validation)
- [Documentation](#documentation)
- [License](#license)
- [Citation](#citation)
- [References](#references)

---

## Key Features

- **Multiple architectures**: Transformer (with GQA, MQA, Flash Attention, RoPE, ALiBi, SwiGLU) and Mamba SSM
- **Scalable training**: DDP, FSDP, mixed precision, EMA, gradient checkpointing
- **Clinical applications**: variant pathogenicity prediction (86.67% accuracy on ClinVar), TF binding, chromatin state, 3D genome
- **Generative**: autoregressive, diffusion (D3PM), beam search, guided generation, infilling
- **Explainability**: SHAP, integrated gradients, SmoothGrad, attention rollout, motif discovery
- **Uncertainty**: MC Dropout, Bayesian layers, deep ensembles, conformal prediction
- **Multi-omics**: DNA + methylation + RNA-seq fusion with cross-modal attention
- **Population-aware**: population embeddings, allele frequency encoding, bias auditing
- **Serving infrastructure**: FastAPI + WebSocket + gRPC, Prometheus, OpenTelemetry, auth, rate limiting
- **Deployable**: Docker (GPU), Helm charts, Kubernetes, ONNX/TorchScript export, INT8 quantization
- **Security by default**: Authentication and rate limiting enabled out of the box

## Project Stats

| Metric | Value |
|--------|-------|
| Python files | 149 |
| Lines of code | 55,894 |
| Modules | 113 |
| Tests | 777 passing |
| CI/CD pipelines | 5 |
| Trained models | 3 (852K, 3.3M, 86M params) |
| Coverage threshold | 65% (core modules) |
| Benchmark validation | ClinVar, OMIM, BEND |

---

## Requirements

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **OS**: Linux (recommended), macOS, WSL2
- **GPU** (optional but recommended): NVIDIA GPU with CUDA 12.1+ for training and fast inference
- **RAM**: 16 GB minimum (32 GB+ recommended for large models)
- **Disk**: ~500 MB for the framework, ~3 GB for full genome data (GRCh38)

### System dependencies (Linux)

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev
```

### System dependencies (macOS)

```bash
brew install xz bzip2 curl
```

---

## Installation

### Option 1: Poetry (recommended)

```bash
# Clone the repository
git clone https://github.com/genova-genomics/genova.git
cd genova

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies (including dev tools)
poetry install

# Or install production dependencies only
poetry install --only main

# Verify the installation
poetry run genova --version
```

### Option 2: pip

```bash
# Clone the repository
git clone https://github.com/genova-genomics/genova.git
cd genova

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install Genova in editable mode
pip install -e .

# Verify the installation
genova --version
```

### Option 3: pip (production only)

```bash
pip install -r requirements.txt
pip install .
```

### Optional: Mamba SSM support (requires CUDA)

```bash
poetry install --extras mamba
# or
pip install mamba-ssm causal-conv1d
```

### Post-installation setup

```bash
# Set up pre-commit hooks (development)
poetry run pre-commit install

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys and settings
```

---

## Quick Start

### 1. Download genome data

```bash
# Single chromosome for testing (~12 MB)
mkdir -p data/reference
curl -L -o data/reference/chr22.fa.gz \
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
gunzip data/reference/chr22.fa.gz

# Full genome — all 22 autosomes (~3 GB)
bash scripts/download_data.sh data/
```

### 2. Train a small model (CPU, ~30 min)

```bash
python scripts/train_on_real_data.py \
    --train-fasta data/reference/chr22.fa \
    --val-fasta data/reference/chr21.fa \
    --output-dir outputs/my_model \
    --d-model 128 --n-layers 4 --n-heads 4 \
    --epochs 2 --batch-size 32
```

### 3. Run predictions

```bash
# Predict variant effects
genova predict --vcf variants.vcf --reference genome.fa --output results.csv

# Extract sequence embeddings
genova embed --input sequences.fa --output embeddings.npy
```

### 4. Start the API server

```bash
# Production mode (auth enabled, requires API key)
genova serve --model-path outputs/my_model/best_model.pt --port 8000

# Development mode (auth disabled)
GENOVA_AUTH_ENABLED=0 GENOVA_RATE_LIMIT_ENABLED=0 \
    genova serve --model-path outputs/my_model/best_model.pt --port 8000
```

---

## CLI Reference

Genova provides a full-featured CLI via `genova` (or `poetry run genova`).

```
genova [COMMAND] [OPTIONS]
```

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `train` | Train a genomic foundation model | `genova train --config configs/train_small.yaml` |
| `predict` | Predict variant effects from a VCF file | `genova predict --vcf input.vcf --reference ref.fa --output results.csv` |
| `serve` | Start the REST API server | `genova serve --model-path ./checkpoints/best --port 8000` |
| `embed` | Extract sequence embeddings to a file | `genova embed --input sequences.fa --output embeddings.npy` |
| `evaluate` | Evaluate a model on test data | `genova evaluate --model-path ./checkpoints/best --test-data ./data/test` |
| `preprocess` | Preprocess FASTA into training-ready format | `genova preprocess --fasta genome.fa --output-dir ./data` |
| `--version` | Print version and exit | `genova --version` |

### Train options

```bash
genova train \
    --config configs/default.yaml \           # Path to YAML config file
    --set training.lr=3e-4 \                  # Override any config value
    --set training.epochs=10 \
    --set model.d_model=256
```

### Predict options

```bash
genova predict \
    --vcf input.vcf \                         # Input VCF file
    --reference data/reference/chr22.fa \      # Reference FASTA
    --model-path outputs/my_model/best_model.pt \  # Trained model
    --output results.csv                       # Output predictions
```

### Serve options

```bash
genova serve \
    --model-path outputs/my_model/best_model.pt \  # Trained model
    --host 0.0.0.0 \                               # Bind address
    --port 8000                                     # Port
```

---

## Training

### Using predefined configurations

```bash
# Small model — 4 layers, 128d, 852K params (development/testing)
genova train --config configs/train_small.yaml

# Large model — 12 layers, 768d, 86M params (production, GPU recommended)
genova train --config configs/train_large.yaml

# Mamba SSM — 12 layers, 768d, 55M params (long sequences 100K+ bp, requires CUDA)
genova train --config configs/train_mamba.yaml
```

### Using Makefile shortcuts

```bash
make pipeline-small    # Full pipeline with small config
make pipeline-large    # Full pipeline with large config (GPU recommended)
make pipeline-mamba    # Full pipeline with Mamba config
```

### Custom training with config overrides

```bash
genova train \
    --config configs/default.yaml \
    --set model.arch=transformer \
    --set model.d_model=512 \
    --set model.n_layers=8 \
    --set model.n_heads=8 \
    --set training.lr=1e-4 \
    --set training.epochs=20 \
    --set training.mixed_precision=bf16 \
    --set data.batch_size=64
```

### Distributed training (multi-GPU)

```bash
# DDP with 4 GPUs
torchrun --nproc_per_node=4 -m genova.training.train \
    --config configs/train_large.yaml

# FSDP (for models that don't fit on a single GPU)
genova train --config configs/train_large.yaml \
    --set training.fsdp=true
```

### Model configurations

| Config | Layers | Hidden | Heads | Params | Use Case |
|--------|--------|--------|-------|--------|----------|
| Small | 4 | 128 | 4 | 852K | Development, testing |
| Medium | 4 | 256 | 4 | 3.3M | Quick experiments |
| Full | 12 | 768 | 12 | 86M | Production |
| Mamba | 12 | 768 | -- | 55M | Long sequences (100K+ bp) |

### Training results

Trained on real human genomic data (GRCh38):

| Model | Data | Val PPL | Val Acc | Variant Acc |
|-------|------|---------|---------|-------------|
| Genova-4L-128d | chr22 (50M bp) | **3.55** | 39.9% | -- |
| Genova-4L-256d | 5 chr (337M bp) | **3.56** | 39.9% | **86.7%** |
| Genova-12L-768d | 20 chr (2.78B bp) | 3.74* | 32.3%* | -- |
| DNABERT (ref.) | Full genome | ~4.2 | ~35% | -- |

*Undertrained on CPU (3K steps). Needs GPU for full convergence.

---

## Inference & Prediction

### Variant effect prediction (CLI)

```bash
# From a VCF file
genova predict \
    --vcf patients/sample.vcf \
    --reference data/reference/hg38.fa \
    --model-path outputs/my_model/best_model.pt \
    --output results/predictions.csv

# Output columns: chrom, pos, ref, alt, score, label, confidence
```

### Embedding extraction (CLI)

```bash
# Extract embeddings from a FASTA file
genova embed \
    --input sequences.fa \
    --model-path outputs/my_model/best_model.pt \
    --output embeddings.npy

# Load in Python
import numpy as np
embeddings = np.load("embeddings.npy")
```

### Model evaluation (CLI)

```bash
genova evaluate \
    --model-path outputs/my_model/best_model.pt \
    --test-data data/test \
    --output results/evaluation.json
```

---

## API Server

### Starting the server

```bash
# Production (auth + rate limiting enabled by default)
genova serve --model-path outputs/my_model/best_model.pt --port 8000

# Development (security disabled)
GENOVA_AUTH_ENABLED=0 GENOVA_RATE_LIMIT_ENABLED=0 \
    genova serve --model-path outputs/my_model/best_model.pt --port 8000

# Via uvicorn directly
uvicorn genova.api.server:create_app --factory --host 0.0.0.0 --port 8000

# Via Make
make serve
```

### API endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Health check and model status |
| `GET` | `/model/info` | No | Model metadata and configuration |
| `POST` | `/predict_variant` | Yes | Variant pathogenicity prediction |
| `POST` | `/predict_expression` | Yes | Gene expression prediction |
| `POST` | `/predict_methylation` | Yes | Methylation beta value prediction |
| `POST` | `/embed` | Yes | Sequence embedding extraction |
| `GET` | `/metrics` | No | Prometheus metrics |
| `WS` | `/ws/generate` | Yes | Streaming sequence generation |

### API usage examples

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Predict variant effect (with API key)
curl -X POST http://localhost:8000/predict_variant \
    -H "Content-Type: application/json" \
    -H "X-API-Key: your-api-key" \
    -d '{
        "variants": [{"chrom": "chr17", "pos": 7577121, "ref": "C", "alt": "T"}],
        "reference_sequence": "ACGTACGTACGTACGT",
        "window_size": 512
    }'

# Extract embeddings
curl -X POST http://localhost:8000/embed \
    -H "Content-Type: application/json" \
    -H "X-API-Key: your-api-key" \
    -d '{"sequences": ["ATCGATCGATCGATCG"], "pooling": "mean"}'
```

```python
import httpx

# Python client example
client = httpx.Client(
    base_url="http://localhost:8000",
    headers={"X-API-Key": "your-api-key"},
)

# Variant prediction
response = client.post("/predict_variant", json={
    "variants": [{"ref": "A", "alt": "G", "sequence": "ATCGATCGATCG"}],
})
print(response.json())

# Embedding extraction
response = client.post("/embed", json={
    "sequences": ["ATCGATCGATCGATCG"],
    "pooling": "mean",
})
embedding = response.json()["embeddings"][0]["embedding"]
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GENOVA_AUTH_ENABLED` | `1` | Enable API key / JWT authentication |
| `GENOVA_RATE_LIMIT_ENABLED` | `1` | Enable rate limiting |
| `GENOVA_RATE_LIMIT_RPM` | `60` | Requests per minute per key |
| `GENOVA_RATE_LIMIT_BACKEND` | `memory` | `memory` or `redis` |
| `GENOVA_REDIS_URL` | `redis://localhost:6379/0` | Redis URL for rate limiting |
| `GENOVA_API_KEYS` | -- | Comma-separated API keys |
| `GENOVA_API_KEYS_FILE` | -- | Path to file with one key per line |
| `GENOVA_JWT_SECRET` | -- | JWT signing secret |
| `GENOVA_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `GENOVA_METRICS_ENABLED` | `0` | Enable Prometheus metrics |
| `GENOVA_TRACING_ENABLED` | `0` | Enable OpenTelemetry tracing |
| `GENOVA_REQUEST_LOGGING_ENABLED` | `0` | Enable structured request logging |

---

## Python API

### Basic usage

```python
import torch
from genova.data.tokenizer import GenomicTokenizer
from genova.models.model_factory import create_model
from genova.utils.config import ModelConfig

# Create model
config = ModelConfig(arch="transformer", d_model=256, n_layers=4,
                     n_heads=4, d_ff=1024, vocab_size=10)
model = create_model(config, task="mlm")

# Tokenize
tok = GenomicTokenizer(mode="nucleotide")
tok.build_vocab()

# Forward pass
sequence = "ATCGATCGATCGATCGATCG"
input_ids = torch.tensor([tok.encode(sequence)])
output = model(input_ids)
print(output["logits"].shape)  # (1, seq_len, vocab_size)
```

### Variant prediction

```python
from genova.perturbation.variant_simulator import VariantSimulator

sim = VariantSimulator(model, tokenizer=tok)
effects = sim.saturate_snps("ATCGATCGATCGATCG")
for e in effects[:5]:
    print(f"  pos={e.position} {e.ref_allele}>{e.alt_allele} effect={e.effect_size:.4f}")
```

### Sequence generation

```python
from genova.generative.autoregressive import AutoregressiveGenerator

gen = AutoregressiveGenerator(model, tokenizer=tok)
result = gen.generate(num_sequences=5, max_length=100, temperature=0.8, top_k=10)
```

### Explainability

```python
from genova.explainability.integrated_gradients import IntegratedGradientsExplainer

explainer = IntegratedGradientsExplainer(model, tok)
attributions = explainer.explain(sequence, n_steps=100)
```

### Uncertainty quantification

```python
from genova.uncertainty.mc_dropout import MCDropoutPredictor

mc = MCDropoutPredictor(model, n_forward_passes=30)
result = mc.predict_with_uncertainty(input_ids)
print(f"Mean: {result['mean'].shape}, Variance: {result['variance'].shape}")
```

### Inference engine (programmatic)

```python
from genova.api.inference import InferenceEngine

engine = InferenceEngine(
    model_path="outputs/my_model",
    device="cuda",      # or "cpu", "auto"
    max_batch_size=64,
)
engine.load()

# Embed sequences
embeddings = engine.embed(["ACGTACGTACGT", "GGCCAATTGGCC"], pooling="mean")

# Predict variants
results = engine.predict_variant(
    ref_sequences=["NNNNNACGTNNNNN"],
    alt_sequences=["NNNNNACGGNNNNN"],
)
print(results)  # [{"score": 0.73, "label": "pathogenic", "confidence": 0.46}]

# Clean up
engine.unload()
```

---

## Security

Genova 1.0 ships with **security enabled by default**:

- **Authentication**: API key or JWT-based auth is on by default. Configure keys via `GENOVA_API_KEYS` or `GENOVA_API_KEYS_FILE`.
- **Rate limiting**: 60 RPM per key by default. Supports Redis backend for multi-instance deployments.
- **CORS**: Restricted origins via `GENOVA_CORS_ORIGINS`. Set to your domain(s) in production.
- **Container security**: Non-root user, read-only filesystem, dropped capabilities.

### Generate an API key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Configure for production

```bash
cp .env.example .env
# Edit .env:
#   GENOVA_API_KEYS=your-generated-key
#   GENOVA_CORS_ORIGINS=https://your-domain.com
```

### Disable security for local development

```bash
export GENOVA_AUTH_ENABLED=0
export GENOVA_RATE_LIMIT_ENABLED=0
```

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## Deployment

### Docker

```bash
# Build GPU image
make docker-build

# Run all services (API, TensorBoard)
make docker-up

# View logs
make docker-logs

# Stop
make docker-down

# Scan for vulnerabilities
make docker-scan
```

### Kubernetes

```bash
# Deploy to staging
make deploy-staging

# Deploy to production (with 5-second confirmation delay)
make deploy-production

# Helm
helm install genova deploy/helm/ -f deploy/helm/values-production.yaml

# Lint Helm chart
make helm-lint
```

### Model export (ONNX / TorchScript)

```bash
make check-export
```

---

## Testing

```bash
# Run all tests (excludes slow and GPU-only)
make test
# or
python -m pytest tests/ -m "not slow and not gpu"

# Run specific test suites
python -m pytest tests/unit/ -v                # Unit tests
python -m pytest tests/integration/ -v         # Integration tests
python -m pytest tests/truthset/ -v            # Biological correctness tests
python -m pytest tests/benchmark/ -v           # Benchmark validation tests

# Run with coverage report (65% threshold on core modules)
make test-cov

# Run full test suite including slow tests
make test-all

# Full local CI (lint + test + build)
make ci
```

---

## Development

### Setup

```bash
# Install all dependencies including dev tools
make install

# This runs:
#   poetry install --no-interaction
#   poetry run pre-commit install
```

### Code quality

```bash
# Auto-format code (black + isort + ruff fix)
make format

# Check formatting and lint (without modifying)
make lint

# Type check (strict mypy)
make typecheck

# Security scan (pip-audit + bandit)
make security-scan
```

### Makefile targets

| Target | Description |
|--------|-------------|
| `make install` | Install all dependencies (including dev) |
| `make install-prod` | Install production dependencies only |
| `make format` | Auto-format code (black, isort, ruff) |
| `make lint` | Run all linters |
| `make typecheck` | Run mypy type checking |
| `make test` | Run test suite (excludes slow/GPU) |
| `make test-all` | Run full test suite |
| `make test-cov` | Run tests with HTML coverage report |
| `make ci` | Full local CI (lint + test + build) |
| `make build` | Build Python package (wheel + sdist) |
| `make security-scan` | Run security scans |
| `make benchmark` | Run performance benchmarks |
| `make docker-build` | Build Docker image |
| `make docker-up` | Start services via docker compose |
| `make docker-down` | Stop services |
| `make docker-scan` | Scan Docker image for vulnerabilities |
| `make serve` | Start API server locally (with reload) |
| `make train` | Launch training run |
| `make pipeline-small` | Full pipeline with small config |
| `make pipeline-large` | Full pipeline with large config |
| `make pipeline-mamba` | Full pipeline with Mamba config |
| `make deploy-staging` | Deploy to staging via Helm |
| `make deploy-production` | Deploy to production via Helm |
| `make clean` | Remove build artifacts and caches |

---

## Project Architecture

```
genova/
├── data/                  # Data pipeline
│   ├── tokenizer.py       #   Nucleotide, k-mer, BPE tokenizers
│   ├── genome_dataset.py  #   Streaming FASTA dataset with MLM masking
│   ├── bpe_tokenizer.py   #   Byte-pair encoding for genomics
│   ├── dataloader.py      #   DataLoader factory with DDP support
│   ├── long_sequence.py   #   100kb+ sequence handling
│   ├── preprocessing.py   #   Genome-to-Parquet pipeline
│   └── quality_report.py  #   FASTA quality analysis
│
├── models/                # Model architectures
│   ├── transformer.py     #   Transformer (GQA, MQA, Flash Attn, SwiGLU, RMSNorm)
│   ├── mamba_model.py     #   Mamba SSM (linear complexity, 100K-1M tokens)
│   ├── embeddings.py      #   Token + position (RoPE, ALiBi, sinusoidal, learned)
│   ├── multi_task.py      #   Multi-task heads (MLM + expression + methylation)
│   ├── model_factory.py   #   create_model() with torch.compile support
│   ├── quantization.py    #   INT8 dynamic/static quantization, QAT
│   ├── pruning.py         #   Attention head + FFN pruning
│   └── export.py          #   ONNX, TorchScript, TensorRT export
│
├── training/              # Training infrastructure
│   ├── trainer.py         #   GenovaTrainer (DDP, AMP, checkpointing)
│   ├── fsdp.py            #   Fully Sharded Data Parallel
│   ├── distributed.py     #   DDP utilities
│   ├── scheduler.py       #   Cosine, warm restarts, one-cycle, polynomial
│   ├── ema.py             #   Exponential moving average
│   ├── distillation.py    #   Knowledge distillation (logit + feature)
│   ├── curriculum.py      #   Curriculum learning with difficulty scoring
│   ├── active_learning.py #   Uncertainty/diversity-based sample selection
│   ├── semi_supervised.py #   Pseudo-labeling, FixMatch, MixMatch
│   ├── differential_privacy.py  # DP-SGD training
│   └── hpo.py             #   Optuna hyperparameter optimization
│
├── evaluation/            # Evaluation & downstream tasks
│   ├── variant_predictor.py    # VCF parsing, pathogenicity prediction
│   ├── structural_variants.py  # SV/CNV prediction
│   ├── tf_binding.py           # TF binding site prediction
│   ├── chromatin.py            # Chromatin state + histone marks
│   ├── genome_3d.py            # TAD boundaries, Hi-C, compartments
│   ├── epi_interaction.py      # Enhancer-promoter interaction
│   ├── metrics.py              # AUROC, AUPRC, perplexity, correlation
│   ├── cross_validation.py     # K-fold, stratified, chromosome-level CV
│   ├── statistical_tests.py    # Bootstrap CI, McNemar, DeLong, FDR
│   └── bias_audit.py           # Population, GC, chromosome bias analysis
│
├── explainability/        # Model interpretation
│   ├── shap_explainer.py       # SHAP (Deep + Kernel explainer)
│   ├── integrated_gradients.py # IG, DeepLIFT, SmoothGrad
│   ├── attention_analysis.py   # Attention rollout, head importance
│   └── visualization.py        # Sequence importance plots
│
├── generative/            # Sequence generation
│   ├── autoregressive.py  #   Top-k, nucleus, temperature sampling + KV-cache
│   ├── beam_search.py     #   Beam search with constraints + n-gram blocking
│   ├── diffusion.py       #   D3PM discrete diffusion
│   ├── guided_generation.py #  Classifier-free/guided + constrained generation
│   ├── infilling.py       #   Fill-in-the-middle generation
│   └── evaluation.py      #   Generation quality metrics
│
├── uncertainty/           # Uncertainty quantification
│   ├── mc_dropout.py      #   Monte Carlo Dropout
│   ├── bayesian.py        #   Bayesian layers + wrapper
│   ├── ensemble.py        #   Deep + snapshot ensemble
│   ├── conformal.py       #   Conformal prediction (classification + regression)
│   └── calibration.py     #   Temperature scaling, ECE, reliability diagrams
│
├── contrastive/           # Self-supervised learning
│   ├── augmentations.py   #   5 genomic augmentations
│   ├── contrastive_model.py #  SimCLR / InfoNCE
│   └── contrastive_trainer.py
│
├── population/            # Population-aware genomics
│   ├── population_model.py    # Population embeddings + variant prediction
│   └── frequency_encoder.py   # Allele frequency encoding (gnomAD/QGP)
│
├── multiomics/            # Multi-omics integration
│   ├── data_fusion.py     #   Cross-modal attention fusion
│   ├── multiomics_model.py #  Methylation + RNA-seq + DNA encoder
│   └── ont_methylation.py #   ONT bedMethyl processing
│
├── evolutionary/          # Evolutionary modeling
│   ├── multi_species.py   #   Multi-species training
│   ├── conservation.py    #   Conservation-weighted loss
│   └── transfer_learning.py #  Cross-species transfer with gradient reversal
│
├── motif/                 # Regulatory motif discovery
│   ├── motif_discovery.py #   PWM construction, genome scanning
│   ├── motif_clustering.py #  JASPAR comparison
│   └── visualization.py   #   Sequence logos
│
├── perturbation/          # In silico perturbation
│   ├── variant_simulator.py   # Saturation mutagenesis
│   ├── sensitivity_map.py     # Per-position sensitivity
│   └── causal_inference.py    # Epistatic interactions
│
├── latent/                # Latent space analysis
│   ├── embedding_analyzer.py  # UMAP, clustering, annotation
│   └── visualization.py       # Publication-quality plots
│
├── benchmark/             # Benchmarking suite
│   ├── benchmark_suite.py     # Orchestrator
│   ├── tasks.py               # Promoter, enhancer, variant, splice tasks
│   ├── standard_benchmarks.py # BEND + Nucleotide Transformer tasks
│   └── comparison.py          # Statistical comparison + LaTeX tables
│
├── api/                   # API & serving
│   ├── server.py          #   FastAPI with 6 REST endpoints
│   ├── websocket.py       #   WebSocket streaming
│   ├── grpc_server.py     #   gRPC interface
│   ├── batch_queue.py     #   Dynamic batching
│   ├── inference.py       #   InferenceEngine
│   ├── schemas.py         #   Pydantic models
│   ├── security.py        #   API key + JWT auth + rate limiting (on by default)
│   ├── monitoring.py      #   Prometheus metrics
│   ├── tracing.py         #   OpenTelemetry
│   └── logging_middleware.py # Structured request logging
│
├── cli/                   # Command-line interface
│   └── main.py            #   Typer CLI: train, predict, serve, evaluate, embed
│
└── utils/                 # Utilities
    ├── config.py          #   YAML config with dataclasses
    ├── cache.py           #   Memory + disk embedding cache
    ├── registry.py        #   Model registry (local, MLflow, HF Hub)
    ├── model_card.py      #   Auto model card generation
    ├── paper_figures.py   #   Publication-quality figures
    ├── paper_tables.py    #   LaTeX table generation
    ├── logging.py         #   loguru setup
    ├── device.py          #   GPU/CPU management
    └── reproducibility.py #   Seed management
```

---

## Benchmark Validation

Genova validates variant prediction accuracy against independent benchmarks:

| Benchmark | Metric | Threshold | Status |
|-----------|--------|-----------|--------|
| ClinVar pathogenic/benign SNPs | AUROC | >= 0.80 | Validated |
| OMIM disease-causing variants | Precision@90%Recall | >= 0.70 | Validated |
| BEND promoter detection | AUROC | >= 0.75 | Validated |
| Nucleotide Transformer tasks | Average AUROC | >= 0.70 | Validated |
| Transition/transversion ratio | Ti/Tv in [2.0, 2.2] | Biologically plausible | Validated |

Run benchmark validation:
```bash
python -m pytest tests/benchmark/ -v -m benchmark
python -m pytest tests/truthset/ -v -m truthset
```

---

## CI/CD

5 GitHub Actions pipelines:

| Pipeline | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | Push/PR | Lint, security audit, test (Python 3.10-3.13), build, Docker scan |
| `cd.yml` | Tag `v*` | Release, deploy staging, deploy production |
| `security.yml` | Push + weekly | Dependency scan, SAST, container scan, secret scan |
| `docs.yml` | Push to main | Build API docs, deploy to GitHub Pages |
| `performance.yml` | Push/PR + weekly | Benchmarks, load tests, PR comment |

---

## Documentation

| Document | Location |
|----------|----------|
| Changelog | [CHANGELOG.md](CHANGELOG.md) |
| Security Policy | [SECURITY.md](SECURITY.md) |
| Environment Config | [.env.example](.env.example) |
| Methodology (1,246 lines) | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) |
| Execution Guide | [docs/EXECUTION_GUIDE.md](docs/EXECUTION_GUIDE.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Model Card | [outputs/genova_expanded/MODEL_CARD.md](outputs/genova_expanded/MODEL_CARD.md) |
| Quickstart Notebook | [docs/notebooks/01_quickstart.ipynb](docs/notebooks/01_quickstart.ipynb) |
| Training Notebook | [docs/notebooks/02_training.ipynb](docs/notebooks/02_training.ipynb) |
| Variant Analysis Notebook | [docs/notebooks/03_variant_analysis.ipynb](docs/notebooks/03_variant_analysis.ipynb) |
| Generation Notebook | [docs/notebooks/04_generation.ipynb](docs/notebooks/04_generation.ipynb) |

---

## License

Apache 2.0

---

## Citation

```bibtex
@software{genova2026,
  title = {Genova: A Production-Grade Genomics Foundation Model},
  year = {2026},
  url = {https://github.com/genova-genomics/genova}
}
```

---

## References

- Ji et al. (2021). DNABERT. *Bioinformatics*.
- Avsec et al. (2021). Enformer. *Nature Methods*.
- Nguyen et al. (2024). Evo. *Science*.
- Dalla-Torre et al. (2023). Nucleotide Transformer. *Nature Methods*.
- Gu & Dao (2023). Mamba. *arXiv*.
