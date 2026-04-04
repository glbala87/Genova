# Genova Execution Guide

## Prerequisites

- Python 3.10+
- pip or Poetry
- ~10GB disk space (genome data)
- Optional: CUDA GPU (for large model training)

---

## 1. Installation

### Option A: pip (quickest)
```bash
cd Genova
pip install -r requirements.txt
pip install -e .
```

### Option B: Poetry
```bash
cd Genova
poetry install
```

### Option C: Docker
```bash
docker compose up dev
```

### Verify installation
```bash
genova --help
python -c "import genova; print(genova.__version__)"
```

---

## 2. Download Genome Data

### Download all 22 human chromosomes (~3GB)
```bash
bash scripts/download_data.sh data/
```

### Or download individual chromosomes (smaller)
```bash
# Just chr22 for quick testing (~12MB)
curl -L -o data/reference/chr22.fa.gz \
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
gunzip data/reference/chr22.fa.gz
```

---

## 3. Train a Model

### Quick start (small model, single chromosome, CPU)
```bash
python scripts/train_on_real_data.py \
    --train-fasta data/reference/chr22.fa \
    --val-fasta data/reference/chr21.fa \
    --output-dir outputs/my_model \
    --d-model 128 --n-layers 4 --n-heads 4 \
    --epochs 2 --batch-size 32 --lr 5e-4
```

### Medium model (5 chromosomes)
```bash
# Build training set
cat data/reference/chr{17..20}.fa data/reference/chr22.fa > data/reference/train.fa

python scripts/train_on_real_data.py \
    --train-fasta data/reference/train.fa \
    --val-fasta data/reference/chr21.fa \
    --output-dir outputs/genova_medium \
    --d-model 256 --n-layers 4 --n-heads 4 \
    --epochs 3 --batch-size 16 --lr 3e-4
```

### Full model (all chromosomes, GPU recommended)
```bash
# Build full training set
cat data/reference/chr{1..20}.fa > data/reference/full_train.fa

python scripts/train_full_model.py
```

### Using Makefile shortcuts
```bash
make pipeline-small    # 4L/256d, quick
make pipeline-large    # 12L/768d, production
make pipeline-mamba    # 12L Mamba SSM
```

---

## 4. Use a Trained Model

### 4.1 Predict Variant Effects

```bash
# From CLI
genova predict \
    --vcf data/variants/clinvar_chr22.vcf \
    --reference data/reference/chr22.fa \
    --model-path outputs/my_model/best_model.pt \
    --output results.csv
```

```python
# From Python
import torch
from genova.data.tokenizer import GenomicTokenizer
from genova.models.model_factory import create_model
from genova.utils.config import ModelConfig

# Load model
checkpoint = torch.load("outputs/my_model/best_model.pt", weights_only=False)
config = ModelConfig(**{k: v for k, v in checkpoint["config"].items()
                        if k in ModelConfig.__dataclass_fields__})
model = create_model(config, task="mlm")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load tokenizer
tok = GenomicTokenizer(mode="nucleotide")
tok.build_vocab()

# Predict on a sequence
sequence = "ATCGATCGATCGATCGATCGATCGATCGATCG"
input_ids = torch.tensor([tok.encode(sequence)])

with torch.no_grad():
    output = model(input_ids)
    logits = output["logits"]        # (1, seq_len, vocab_size)
    probs = torch.softmax(logits, -1)  # prediction probabilities
```

### 4.2 Extract Embeddings

```python
# Get sequence embeddings
with torch.no_grad():
    hidden = model.transformer(input_ids, output_hidden_states=True)
    embeddings = hidden["last_hidden_state"]  # (1, seq_len, d_model)

    # Pool to single vector
    seq_embedding = embeddings.mean(dim=1)    # (1, d_model)
```

```bash
# From CLI
genova embed \
    --input sequences.fa \
    --model-path outputs/my_model/best_model.pt \
    --output embeddings.npy
```

### 4.3 Generate Sequences

```python
from genova.generative.autoregressive import AutoregressiveGenerator

generator = AutoregressiveGenerator(model, tokenizer=tok)

# Generate with different strategies
result = generator.generate(
    num_sequences=5,
    max_length=100,
    temperature=0.8,
    top_k=10
)
```

### 4.4 Explain Predictions (SHAP)

```python
from genova.explainability.shap_explainer import GenomicSHAPExplainer

explainer = GenomicSHAPExplainer(model, tok)
attributions = explainer.explain(sequence)
# attributions = per-nucleotide importance scores
```

### 4.5 Saturation Mutagenesis

```python
from genova.perturbation.variant_simulator import VariantSimulator

simulator = VariantSimulator(model, tokenizer=tok)
effects = simulator.saturate_snps("ATCGATCGATCGATCG")
# effects = list of VariantEffect for every possible SNP
```

---

## 5. Serve as API

### Start API server
```bash
# CLI
genova serve --model-path outputs/my_model/best_model.pt --port 8000

# Or directly
uvicorn genova.api.server:create_app --factory --host 0.0.0.0 --port 8000

# Or Docker
docker compose up api
```

### API endpoints
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Predict variant effect
curl -X POST http://localhost:8000/predict_variant \
    -H "Content-Type: application/json" \
    -d '{
        "reference_sequence": "ATCGATCG",
        "variants": [{"position": 4, "ref": "A", "alt": "G"}]
    }'

# Get embedding
curl -X POST http://localhost:8000/embed \
    -H "Content-Type: application/json" \
    -d '{"sequence": "ATCGATCGATCGATCG", "pooling": "mean"}'

# Predict expression
curl -X POST http://localhost:8000/predict_expression \
    -H "Content-Type: application/json" \
    -d '{"sequence": "ATCGATCGATCGATCG"}'
```

---

## 6. Run Benchmarks

```bash
# Run all benchmarks
python scripts/run_immediate_steps.py

# Or individually
make benchmarks MODEL=outputs/my_model/best_model.pt

# Generate model card
make model-card MODEL=outputs/my_model/best_model.pt

# Data quality report
make data-quality
```

---

## 7. Run Tests

```bash
# All 522 tests
make test

# Or specific suites
python -m pytest tests/unit/ -v             # 221 unit tests
python -m pytest tests/integration/ -v      # 204 integration tests
python -m pytest tests/truthset/ -v         # 97 biological truthset tests

# With coverage
make test-cov
```

---

## 8. Development Workflow

```bash
# Format code
make format

# Lint
make lint

# Type check
make typecheck

# Full CI locally
make ci

# Security scan
make security-scan
```

---

## 9. Deploy to Kubernetes

### Staging
```bash
make deploy-staging
```

### Production
```bash
make deploy-production
```

### Docker
```bash
# Build
make docker-build

# Run all services
make docker-up

# View logs
make docker-logs
```

---

## 10. Advanced Usage

### Train with BPE tokenizer
```python
from genova.data.tokenizer import create_tokenizer

tok = create_tokenizer("bpe")
tok.train(["ATCGATCG" * 1000], vocab_size=512)
tok.save("my_bpe_tokenizer.json")
```

### Train with population-aware model
```python
from genova.population.population_model import PopulationAwareEncoder

encoder = PopulationAwareEncoder(backbone=model.transformer, d_model=256)
output = encoder(input_ids, population_labels=["EUR"])
```

### Uncertainty estimation
```python
from genova.uncertainty.mc_dropout import MCDropoutPredictor

mc = MCDropoutPredictor(model, n_forward_passes=30)
result = mc.predict_with_uncertainty(input_ids)
print(f"Mean: {result['mean']}, Variance: {result['variance']}")
```

### Conformal prediction
```python
from genova.uncertainty.conformal import ConformalPredictor

cp = ConformalPredictor()
cp.calibrate(cal_scores, cal_labels, alpha=0.1)  # 90% coverage
prediction_sets = cp.predict_set(test_scores)
```

### Knowledge distillation
```python
from genova.training.distillation import DistillationTrainer

trainer = DistillationTrainer(
    teacher=large_model,
    student=small_model,
    temperature=4.0,
    alpha=0.5
)
trainer.train_step(input_ids, labels)
```

### Active learning
```python
from genova.training.active_learning import ActiveLearner

learner = ActiveLearner(model, strategy="uncertainty")
selected = learner.select_samples(unlabeled_pool, budget=100)
```

### Model quantization
```python
from genova.models.quantization import quantize_dynamic, compare_model_sizes

quantized = quantize_dynamic(model)
sizes = compare_model_sizes(model, quantized)
print(f"Compression: {sizes['compression_ratio']:.1f}x")
```

### Export to ONNX
```python
from genova.models.export import export_onnx, export_torchscript

export_torchscript(model, sample_input, "model.pt")
export_onnx(model, sample_input, "model.onnx")
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt && pip install -e .` |
| Download data | `bash scripts/download_data.sh data/` |
| Train (quick) | `python scripts/train_on_real_data.py --train-fasta chr22.fa ...` |
| Train (full) | `make pipeline-large` |
| Predict | `genova predict --vcf input.vcf --reference ref.fa` |
| Serve API | `genova serve --model-path best_model.pt` |
| Embed | `genova embed --input seqs.fa --output embeddings.npy` |
| Test | `make test` |
| Lint | `make lint` |
| Docker | `make docker-up` |
| Deploy | `make deploy-staging` |
| Benchmark | `make benchmarks` |
