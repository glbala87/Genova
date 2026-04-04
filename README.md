# Genova

**A Production-Grade Genomics Foundation Model**

Genova is a modular, extensible framework for building, training, and deploying genomic foundation models. Pre-trained on the human reference genome (GRCh38) using masked language modeling, Genova learns contextual representations of DNA sequences for variant effect prediction, regulatory element classification, sequence generation, and multi-omics integration.

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
- **Production-ready**: FastAPI + WebSocket + gRPC, Prometheus, OpenTelemetry, auth, rate limiting
- **Deployable**: Docker (GPU), Helm charts, Kubernetes, ONNX/TorchScript export, INT8 quantization

## Project Stats

| Metric | Value |
|--------|-------|
| Python files | 149 |
| Lines of code | 55,894 |
| Modules | 113 |
| Tests | 522 passing |
| CI/CD pipelines | 5 |
| Trained models | 3 (852K, 3.3M, 86M params) |
| Production checks | 27/27 PASS |

---

## Quick Start

### Install

```bash
git clone https://github.com/genova-genomics/genova.git
cd genova
pip install -r requirements.txt
pip install -e .
```

### Download genome data

```bash
# Single chromosome for testing (~12MB)
mkdir -p data/reference
curl -L -o data/reference/chr22.fa.gz \
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
gunzip data/reference/chr22.fa.gz

# Or all 22 autosomes (~3GB)
bash scripts/download_data.sh data/
```

### Train a model

```bash
# Small model (CPU, ~30 min)
python scripts/train_on_real_data.py \
    --train-fasta data/reference/chr22.fa \
    --val-fasta data/reference/chr21.fa \
    --output-dir outputs/my_model \
    --d-model 128 --n-layers 4 --n-heads 4 \
    --epochs 2 --batch-size 32

# Full model (GPU recommended)
make pipeline-large
```

### Use the trained model

```bash
# Predict variant effects
genova predict --vcf variants.vcf --reference genome.fa --output results.csv

# Extract embeddings
genova embed --input sequences.fa --output embeddings.npy

# Start API server
genova serve --model-path outputs/my_model/best_model.pt --port 8000
```

---

## Architecture

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
│   ├── security.py        #   API key + JWT auth + rate limiting
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

## Model Configurations

| Config | Layers | Hidden | Heads | Params | Use Case |
|--------|--------|--------|-------|--------|----------|
| Small | 4 | 128 | 4 | 852K | Development, testing |
| Medium | 4 | 256 | 4 | 3.3M | Quick experiments |
| Full | 12 | 768 | 12 | 86M | Production |
| Mamba | 12 | 768 | — | 55M | Long sequences (100K+ bp) |

```bash
# Use predefined configs
genova train --config configs/train_small.yaml
genova train --config configs/train_large.yaml
genova train --config configs/train_mamba.yaml
```

---

## Training Results

Trained on real human genomic data (GRCh38):

| Model | Data | Val PPL | Val Acc | Variant Acc |
|-------|------|---------|---------|-------------|
| Genova-4L-128d | chr22 (50M bp) | **3.55** | 39.9% | — |
| Genova-4L-256d | 5 chr (337M bp) | **3.56** | 39.9% | **86.7%** |
| Genova-12L-768d | 20 chr (2.78B bp) | 3.74* | 32.3%* | — |
| DNABERT (ref.) | Full genome | ~4.2 | ~35% | — |

*Undertrained on CPU (3K steps). Needs GPU for full convergence.

---

## API

```bash
# Start server
genova serve --model-path outputs/my_model/best_model.pt

# Endpoints
GET  /health              # Health check
GET  /model/info          # Model metadata
POST /predict_variant     # Variant pathogenicity
POST /predict_expression  # Gene expression
POST /predict_methylation # Methylation prediction
POST /embed               # Sequence embeddings
GET  /metrics             # Prometheus metrics
WS   /ws/generate         # Streaming generation
```

```python
import httpx

# Predict variant effect
response = httpx.post("http://localhost:8000/predict_variant", json={
    "reference_sequence": "ATCGATCGATCG",
    "variants": [{"position": 4, "ref": "A", "alt": "G"}]
})
print(response.json())

# Get embedding
response = httpx.post("http://localhost:8000/embed", json={
    "sequence": "ATCGATCGATCGATCG",
    "pooling": "mean"
})
embedding = response.json()["embedding"]
```

---

## Python API

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

### Variant Prediction
```python
from genova.perturbation.variant_simulator import VariantSimulator

sim = VariantSimulator(model, tokenizer=tok)
effects = sim.saturate_snps("ATCGATCGATCGATCG")
for e in effects[:5]:
    print(f"  pos={e.position} {e.ref_allele}>{e.alt_allele} effect={e.effect_size:.4f}")
```

### Sequence Generation
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

### Uncertainty
```python
from genova.uncertainty.mc_dropout import MCDropoutPredictor

mc = MCDropoutPredictor(model, n_forward_passes=30)
result = mc.predict_with_uncertainty(input_ids)
print(f"Mean: {result['mean'].shape}, Variance: {result['variance'].shape}")
```

---

## Deployment

### Docker
```bash
# Build GPU image
make docker-build

# Run all services (train, api, tensorboard)
make docker-up

# View logs
make docker-logs
```

### Kubernetes
```bash
# Deploy to staging
make deploy-staging

# Deploy to production (with confirmation)
make deploy-production

# Helm
helm install genova deploy/helm/ -f deploy/helm/values-production.yaml
```

---

## Testing

```bash
# All 522 tests
make test

# Specific suites
python -m pytest tests/unit/ -v              # 221 unit tests
python -m pytest tests/integration/ -v       # 204 integration tests
python -m pytest tests/truthset/ -v          # 97 biological correctness tests

# With coverage
make test-cov

# Full CI (lint + test + build)
make ci
```

---

## Development

```bash
# Setup
make install

# Format + lint
make format
make lint

# Type check
make typecheck

# Security scan
make security-scan

# Performance benchmark
make benchmark
```

---

## CI/CD

5 GitHub Actions pipelines:

| Pipeline | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | Push/PR | Lint, test (Python 3.10-3.12), build, Docker scan |
| `cd.yml` | Tag `v*` | Release, deploy staging, deploy production |
| `security.yml` | Push + weekly | Dependency scan, SAST, container scan, secret scan |
| `docs.yml` | Push to main | Build API docs, deploy to GitHub Pages |
| `performance.yml` | Push/PR + weekly | Benchmarks, load tests, PR comment |

---

## Documentation

| Document | Location |
|----------|----------|
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
